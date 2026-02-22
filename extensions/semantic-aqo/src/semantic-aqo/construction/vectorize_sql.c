/*
 * Usage:
 *   const char *q_a  = "SELECT name FROM customers WHERE city = 'Hanoi'";
 *   const char *q_b  = "SELECT name FROM clients WHERE city = 'HCMC'";
 *   const char *q_c  = "SELECT SUM(amount) FROM orders GROUP BY month";
 *
 *   const char *qs[] = { q_a, q_b, q_c };
 *   SQLVector  *vecs = vectorize_sql_queries(qs, 3);
 */

#define _POSIX_C_SOURCE 200809L

#include "vectorize_sql.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

/* ------------------------------------------------------------------ */
/*  Compile-time defaults                                              */
/* ------------------------------------------------------------------ */
#ifndef SENSATE_VOCAB_PATH
#  define SENSATE_VOCAB_PATH  "assets/vocab.bin"
#endif
#ifndef SENSATE_EMBED_PATH
#  define SENSATE_EMBED_PATH  "assets/sense_embeddings.bin"
#endif

#define DEFAULT_NUM_SENSES  3
#define DEFAULT_EMBED_DIM   150
#define DEFAULT_WINDOW      7
#define ATTN_SIGMA          3.0f
#define MAX_TOKEN_LEN       256
#define MAX_ALIASES         32

/* ------------------------------------------------------------------ */
/*  Tiny dynamic string array                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    char  **data;
    int     count;
    int     cap;
} StrArr;

static StrArr *strarr_new(void)
{
    StrArr *a = calloc(1, sizeof(StrArr));
    a->data = malloc(32 * sizeof(char *));
    a->cap  = 32;
    return a;
}
static void strarr_push(StrArr *a, const char *s)
{
    if (a->count >= a->cap) {
        a->cap *= 2;
        a->data = realloc(a->data, a->cap * sizeof(char *));
    }
    a->data[a->count++] = strdup(s);
}
static void strarr_free(StrArr *a)
{
    if (!a) return;
    for (int i = 0; i < a->count; i++) free(a->data[i]);
    free(a->data);
    free(a);
}

/* ------------------------------------------------------------------ */
/*  SQL tokeniser  (mirrors preprocessing_pipeline.py)                */
/*                                                                     */
/*  Produces tokens:  SQL keywords, <TAB>, <COL>, <COL_OUT>, <NUM>,   */
/*  <STR>, <NULL>, <BOOL_TRUE>, <BOOL_FALSE>, <ALIAS_Tx>, punctuation */
/* ------------------------------------------------------------------ */

/* Exact same keyword list as utils/sql_preprocessor.c */
static const char *SQL_KEYWORDS[] = {
    "SELECT", "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
    "ON", "AND", "OR", "NOT", "IN", "IS", "NULL", "AS", "GROUP", "BY", "ORDER",
    "HAVING", "LIMIT", "OFFSET", "UNION", "DISTINCT", "CASE", "WHEN", "THEN",
    "ELSE", "END", "ASC", "DESC", "LIKE", "BETWEEN", "EXISTS", "ALL", "ANY",
    "INSERT", "UPDATE", "DELETE", "VALUES", "SET", "INTO", "CREATE", "DROP",
    "ALTER", "TABLE", "INDEX", "VIEW", "PRIMARY", "KEY", "FOREIGN", "REFERENCES",
    "SUM", "COUNT", "AVG", "MIN", "MAX", "CAST", "COALESCE", "NULLIF",
    NULL
};

static bool is_sql_keyword(const char *s)
{
    for (int i = 0; SQL_KEYWORDS[i]; i++)
        if (strcasecmp(s, SQL_KEYWORDS[i]) == 0) return true;
    return false;
}

static bool is_number_str(const char *s)
{
    if (!s || !*s) return false;
    char *end;
    strtod(s, &end);
    return *end == '\0';
}

/* Simple regex-free tokeniser: splits SQL into words / string-literals /
   numbers / operators / punctuation using a single-pass char scan.      */
static StrArr *raw_tokenize(const char *sql)
{
    StrArr *out = strarr_new();
    const char *p = sql;
    char buf[4096];

    while (*p) {
        /* skip spaces */
        if (isspace((unsigned char)*p)) { p++; continue; }

        /* single-quoted string */
        if (*p == '\'') {
            const char *start = p++;
            while (*p && !(*p == '\'' && *(p+1) != '\'')) p++;
            if (*p) p++;                     /* consume closing quote */
            int len = (int)(p - start);
            if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
            memcpy(buf, start, len); buf[len] = '\0';
            strarr_push(out, buf);
            continue;
        }

        /* double-quoted identifier */
        if (*p == '"') {
            const char *start = p++;
            while (*p && *p != '"') p++;
            if (*p) p++;
            int len = (int)(p - start);
            if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
            memcpy(buf, start, len); buf[len] = '\0';
            strarr_push(out, buf);
            continue;
        }

        /* bracketed identifier [name] */
        if (*p == '[') {
            const char *start = p++;
            while (*p && *p != ']') p++;
            if (*p) p++;
            int len = (int)(p - start);
            if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
            memcpy(buf, start, len); buf[len] = '\0';
            strarr_push(out, buf);
            continue;
        }

        /* multi-char operators */
        if ((p[0]=='<' && p[1]=='>') || (p[0]=='!' && p[1]=='=') ||
            (p[0]=='<' && p[1]=='=') || (p[0]=='>' && p[1]=='=')) {
            buf[0]=p[0]; buf[1]=p[1]; buf[2]='\0';
            strarr_push(out, buf);
            p += 2; continue;
        }

        /* number */
        if (isdigit((unsigned char)*p) ||
            (*p == '.' && isdigit((unsigned char)*(p+1)))) {
            const char *start = p;
            while (isdigit((unsigned char)*p) || *p == '.') p++;
            int len = (int)(p - start);
            if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
            memcpy(buf, start, len); buf[len] = '\0';
            strarr_push(out, buf);
            continue;
        }

        /* identifier / keyword (letters, digits, _, #) */
        if (isalpha((unsigned char)*p) || *p == '_' || *p == '#') {
            const char *start = p;
            while (isalnum((unsigned char)*p) || *p == '_' || *p == '#') p++;
            int len = (int)(p - start);
            if (len >= (int)sizeof(buf)) len = (int)sizeof(buf) - 1;
            memcpy(buf, start, len); buf[len] = '\0';
            strarr_push(out, buf);
            continue;
        }

        /* single-char punctuation / operator */
        buf[0] = *p++; buf[1] = '\0';
        strarr_push(out, buf);
    }
    return out;
}

/* Full preprocessing: classify raw tokens into symbolic ones */
static StrArr *preprocess_sql(const char *sql)
{
    StrArr *raw = raw_tokenize(sql);
    StrArr *out = strarr_new();

    /* --- pass 1: collect table aliases --- */
    char alias_name[MAX_ALIASES][MAX_TOKEN_LEN];
    int  alias_num [MAX_ALIASES];
    int  n_aliases  = 0;
    int  alias_ctr  = 0;

    /* Also collect output aliases from SELECT … AS … (before FROM) */
    char out_alias[MAX_ALIASES][MAX_TOKEN_LEN];
    int  n_out_alias = 0;

    for (int i = 0; i < raw->count; i++) {
        const char *t = raw->data[i];
        /* FROM/JOIN <table> [AS] <alias> */
        if (strcasecmp(t, "FROM") == 0 || strcasecmp(t, "JOIN") == 0) {
            int j = i + 1;                          /* table name */
            if (j >= raw->count) continue;
            int k = j + 1;                          /* possible alias or AS */
            if (k >= raw->count) continue;
            if (strcasecmp(raw->data[k], "AS") == 0) k++;
            if (k < raw->count && !is_sql_keyword(raw->data[k]) &&
                n_aliases < MAX_ALIASES) {
                bool already = false;
                for (int a = 0; a < n_aliases; a++)
                    if (strcasecmp(alias_name[a], raw->data[k]) == 0)
                        { already = true; break; }
                if (!already) {
                    strncpy(alias_name[n_aliases], raw->data[k], MAX_TOKEN_LEN-1);
                    alias_name[n_aliases][MAX_TOKEN_LEN-1] = '\0';
                    alias_num[n_aliases] = ++alias_ctr;
                    n_aliases++;
                }
            }
        }
    }

    /* output aliases: SELECT … AS <name> … FROM */
    bool in_select = false;
    for (int i = 0; i < raw->count; i++) {
        if (strcasecmp(raw->data[i], "SELECT") == 0) { in_select = true; continue; }
        if (strcasecmp(raw->data[i], "FROM")   == 0) { in_select = false; continue; }
        if (in_select && strcasecmp(raw->data[i], "AS") == 0) {
            int j = i + 1;
            if (j < raw->count && !is_sql_keyword(raw->data[j]) &&
                n_out_alias < MAX_ALIASES) {
                strncpy(out_alias[n_out_alias++], raw->data[j], MAX_TOKEN_LEN-1);
            }
        }
    }

    /* --- pass 2: emit symbolic tokens --- */
    bool after_from_join = false;  /* next id token is a table name */
    bool after_table     = false;  /* next non-kw token is the alias */
    bool after_as_sel    = false;  /* next token is output alias in SELECT */
    bool after_as_from   = false;  /* next token is table alias after AS */
    bool sel_phase       = false;

    for (int i = 0; i < raw->count; i++) {
        const char *t = raw->data[i];
        char up[MAX_TOKEN_LEN];
        strncpy(up, t, MAX_TOKEN_LEN-1); up[MAX_TOKEN_LEN-1] = '\0';
        for (char *c = up; *c; c++) *c = (char)toupper((unsigned char)*c);

        /* string literal */
        if (t[0] == '\'' || t[0] == '"') { strarr_push(out, "<STR>"); after_as_sel = false; continue; }

        /* bracketed identifier */
        if (t[0] == '[') {
            if (after_as_sel) { strarr_push(out, "<COL_OUT>"); after_as_sel = false; }
            else               strarr_push(out, "<COL>");
            after_table = false; continue;
        }

        /* number */
        if (is_number_str(t)) { strarr_push(out, "<NUM>"); after_as_sel = false; continue; }

        /* punctuation (single char, not alpha) */
        if (!isalpha((unsigned char)t[0]) && t[0] != '_' && t[0] != '#') {
            strarr_push(out, t);
            after_as_sel = false; after_as_from = false;
            continue;
        }

        /* SELECT / FROM / JOIN keywords */
        /* NOTE: NULL is handled via is_sql_keyword() returning "NULL" (same as original.
         *        TRUE/FALSE are not keywords → fall through to <COL> (same as original). */
        if (strcmp(up, "SELECT") == 0) {
            strarr_push(out, "SELECT"); sel_phase = true;
            after_from_join = after_table = after_as_sel = false; continue;
        }
        if (strcmp(up, "FROM") == 0) {
            strarr_push(out, "FROM"); sel_phase = false;
            after_from_join = true; after_table = after_as_sel = false; continue;
        }
        if (strcmp(up, "JOIN") == 0) {
            strarr_push(out, "JOIN");
            after_from_join = true; after_table = false; continue;
        }

        /* AS keyword */
        if (strcmp(up, "AS") == 0) {
            strarr_push(out, "AS");
            if (sel_phase) after_as_sel  = true;
            else           after_as_from = true;
            continue;
        }

        /* other keywords */
        if (is_sql_keyword(t)) {
            strarr_push(out, up);
            after_as_sel = after_as_from = false;
            if (strcmp(up,"WHERE")==0 || strcmp(up,"GROUP")==0 ||
                strcmp(up,"ORDER")==0 || strcmp(up,"HAVING")==0)
                sel_phase = false;
            continue;
        }

        /* table name right after FROM / JOIN */
        if (after_from_join) {
            strarr_push(out, "<TAB>");
            after_from_join = false; after_table = true; continue;
        }

        /* alias after table name (or after AS in FROM clause) */
        if (after_table || after_as_from) {
            /* find alias number */
            int anum = -1;
            for (int a = 0; a < n_aliases; a++)
                if (strcasecmp(alias_name[a], t) == 0) { anum = alias_num[a]; break; }
            if (anum > 0) {
                char buf[32]; snprintf(buf, sizeof(buf), "<ALIAS_T%d>", anum);
                strarr_push(out, buf);
            } else {
                strarr_push(out, "<COL>");
            }
            after_table = after_as_from = false; continue;
        }

        /* output alias position */
        if (after_as_sel) {
            strarr_push(out, "<COL_OUT>"); after_as_sel = false; continue;
        }

        /* qualified: alias.col  (next two tokens are '.' and something) */
        if (i + 2 < raw->count && strcmp(raw->data[i+1], ".") == 0) {
            int anum = -1;
            for (int a = 0; a < n_aliases; a++)
                if (strcasecmp(alias_name[a], t) == 0) { anum = alias_num[a]; break; }
            char buf[32];
            if (anum > 0) snprintf(buf, sizeof(buf), "<ALIAS_T%d>", anum);
            else          snprintf(buf, sizeof(buf), "<COL>");
            strarr_push(out, buf);
            strarr_push(out, ".");
            strarr_push(out, "<COL>");
            i += 2; continue;
        }

        /* standalone alias token */
        int anum = -1;
        for (int a = 0; a < n_aliases; a++)
            if (strcasecmp(alias_name[a], t) == 0) { anum = alias_num[a]; break; }
        if (anum > 0) {
            char buf[32]; snprintf(buf, sizeof(buf), "<ALIAS_T%d>", anum);
            strarr_push(out, buf); continue;
        }

        /* output alias reuse (ORDER BY / HAVING etc.) */
        bool is_oalias = false;
        for (int a = 0; a < n_out_alias; a++)
            if (strcasecmp(out_alias[a], t) == 0) { is_oalias = true; break; }
        if (is_oalias) { strarr_push(out, "<COL_OUT>"); continue; }

        /* function call: next token is '(' */
        if (i + 1 < raw->count && strcmp(raw->data[i+1], "(") == 0) {
            strarr_push(out, up); continue;
        }

        /* default: column */
        strarr_push(out, "<COL>");
    }

    strarr_free(raw);
    return out;
}

/* ------------------------------------------------------------------ */
/*  Global model state                                                 */
/* ------------------------------------------------------------------ */
typedef struct { char *word; int id; } VocabEntry;   /* id: 0-based model index */

static VocabEntry *g_vocab     = NULL;
static int         g_vocab_n   = 0;
static float      *g_sense_data= NULL;  /* [V × K × D] float32 */
static int         g_K = 0, g_D = 0, g_W = DEFAULT_WINDOW;
static bool        g_loaded    = false;

/* ------------------------------------------------------------------ */
/*  Vocab helpers                                                      */
/* ------------------------------------------------------------------ */
static int vocab_lookup(const char *w)
{
    for (int i = 0; i < g_vocab_n; i++)
        if (strcmp(g_vocab[i].word, w) == 0) return g_vocab[i].id;
    return -1;
}

/* Return float* to [K×D] block for model id (0-based), or NULL */
static const float *sense_block(int model_id)
{
    if (model_id < 0 || model_id >= g_vocab_n) return NULL;
    return g_sense_data + (size_t)model_id * g_K * g_D;
}

/* ------------------------------------------------------------------ */
/*  Binary loaders                                                     */
/* ------------------------------------------------------------------ */
static bool load_vocab_bin(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[vectorize_sql] cannot open %s\n", path); return false; }

    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n <= 0) {
        fprintf(stderr, "[vectorize_sql] vocab.bin: bad header\n"); fclose(f); return false;
    }

    g_vocab   = calloc((size_t)n, sizeof(VocabEntry));
    g_vocab_n = (int)n;

    for (int i = 0; i < (int)n; i++) {
        int32_t wlen = 0;
        if (fread(&wlen, 4, 1, f) != 1 || wlen <= 0 || wlen > 1024) {
            fprintf(stderr, "[vectorize_sql] vocab.bin: bad wlen at %d\n", i);
            fclose(f); return false;
        }
        g_vocab[i].word = malloc((size_t)(wlen + 1));
        if ((int)fread(g_vocab[i].word, 1, (size_t)wlen, f) != wlen) {
            fclose(f); return false;
        }
        g_vocab[i].word[wlen] = '\0';

        int32_t mid = 0;
        if (fread(&mid, 4, 1, f) != 1) { fclose(f); return false; }
        g_vocab[i].id = (int)mid;
    }

    fclose(f);
    fprintf(stderr, "[vectorize_sql] vocab loaded: %d words\n", (int)n);
    return true;
}

static bool load_embeddings_bin(const char *path, int K, int D)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[vectorize_sql] cannot open %s\n", path); return false; }

    int32_t total = 0, fdim = 0;
    if (fread(&total, 4, 1, f) != 1 || fread(&fdim, 4, 1, f) != 1) {
        fprintf(stderr, "[vectorize_sql] sense_embeddings.bin: bad header\n");
        fclose(f); return false;
    }

    if (fdim != D) {
        fprintf(stderr, "[vectorize_sql] dim mismatch: file=%d expected=%d, using %d\n",
                (int)fdim, D, (int)fdim);
        D = (int)fdim;
    }

    int V = g_vocab_n;
    if (V > 0 && total > 0 && (total % V) == 0) K = total / V;

    g_K = K; g_D = D;
    g_sense_data = calloc((size_t)V * K * D, sizeof(float));
    if (!g_sense_data) {
        fprintf(stderr, "[vectorize_sql] OOM for sense_data\n"); fclose(f); return false;
    }

    float *buf = malloc((size_t)D * sizeof(float));
    int loaded = 0;

    for (int r = 0; r < (int)total; r++) {
        int32_t wlen = 0;
        if (fread(&wlen, 4, 1, f) != 1 || wlen <= 0 || wlen > 1024) break;
        char word[1025];
        if ((int)fread(word, 1, (size_t)wlen, f) != wlen) break;
        word[wlen] = '\0';

        int32_t sense_id = 0;
        if (fread(&sense_id, 4, 1, f) != 1) break;
        if ((int)fread(buf, sizeof(float), (size_t)D, f) != D) break;

        int mid = vocab_lookup(word);
        if (mid < 0 || mid >= V || sense_id < 0 || sense_id >= K) continue;

        memcpy(g_sense_data + ((size_t)mid * K + sense_id) * D,
               buf, (size_t)D * sizeof(float));
        loaded++;
    }

    free(buf);
    fclose(f);
    fprintf(stderr, "[vectorize_sql] embeddings loaded: %d / %d records, K=%d D=%d\n",
            loaded, (int)total, K, D);
    return true;
}

/* ------------------------------------------------------------------ */
/*  Attention forward  (port of model/attention.c without palloc)     */
/*                                                                     */
/*  Inputs:                                                            */
/*    cs  [K, D]       center sense embeddings                         */
/*    ctx [M, K, D]    context sense embeddings                        */
/*    rp  [M]          relative positions                              */
/*  Outputs (allocated, caller must free):                             */
/*    u     [M, D]     mean context per position                       */
/*    sc    [K, D]     attention-weighted context per sense            */
/* ------------------------------------------------------------------ */
static bool attention_forward(
        const float *cs, const float *ctx, const int *rp,
        int K, int D, int M, float sigma,
        float **out_u, float **out_sc)
{
    float *u  = calloc((size_t)M * D, sizeof(float));
    float *scores = calloc((size_t)K * M, sizeof(float));
    float *ws = calloc((size_t)K * M, sizeof(float));
    float *sc = calloc((size_t)K * D, sizeof(float));
    if (!u || !scores || !ws || !sc) { free(u); free(scores); free(ws); free(sc); return false; }

    /* u[j,d] = mean over senses of ctx[j,*,d] */
    for (int j = 0; j < M; j++)
        for (int d = 0; d < D; d++) {
            float sum = 0;
            for (int k = 0; k < K; k++) sum += ctx[((size_t)j*K+k)*D+d];
            u[(size_t)j*D+d] = sum / (float)K;
        }

    /* scores[k,j] = dot(cs[k], u[j]) / sqrt(D) + distance_bias */
    float scale = sqrtf((float)D);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < M; j++) {
            float dot = 0;
            for (int d = 0; d < D; d++) dot += cs[(size_t)k*D+d] * u[(size_t)j*D+d];
            float bias = 0;
            if (rp) bias = -0.5f * ((float)abs(rp[j]) / sigma) * ((float)abs(rp[j]) / sigma);
            scores[(size_t)k*M+j] = dot / scale + bias;
        }

    /* softmax over j for each k */
    for (int k = 0; k < K; k++) {
        float mx = scores[(size_t)k*M];
        for (int j = 1; j < M; j++) if (scores[(size_t)k*M+j] > mx) mx = scores[(size_t)k*M+j];
        float sm = 0;
        for (int j = 0; j < M; j++) { ws[(size_t)k*M+j] = expf(scores[(size_t)k*M+j] - mx); sm += ws[(size_t)k*M+j]; }
        /* Match original softmax_rows: uniform fallback on degenerate sum */
        if (sm <= 0.0f || !isfinite(sm)) {
            float uni = 1.0f / (float)M;
            for (int j = 0; j < M; j++) ws[(size_t)k*M+j] = uni;
        } else {
            for (int j = 0; j < M; j++) ws[(size_t)k*M+j] /= sm;
        }
    }

    /* sc[k,d] = sum_j ws[k,j] * u[j,d] */
    for (int k = 0; k < K; k++)
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j < M; j++) s += ws[(size_t)k*M+j] * u[(size_t)j*D+d];
            sc[(size_t)k*D+d] = s;
        }

    free(scores); free(ws);
    *out_u  = u;
    *out_sc = sc;
    return true;
}

/* ------------------------------------------------------------------ */
/*  Gating network  (port of model/gating_network.c without palloc)   */
/*                                                                     */
/*  sense_ctx [K, D]  attention-weighted context per sense            */
/*  cs        [K, D]  center sense embeddings                          */
/*  Returns index of best sense.                                       */
/* ------------------------------------------------------------------ */
static int gating_network(const float *cs, const float *sense_ctx, int K, int D,
                           float *out_emb)
{
    float *scores = calloc((size_t)K, sizeof(float));

    for (int k = 0; k < K; k++) {
        const float *a = cs + (size_t)k*D;
        const float *b = sense_ctx + (size_t)k*D;
        float dot = 0, na = 0, nb = 0;
        for (int d = 0; d < D; d++) { dot += a[d]*b[d]; na += a[d]*a[d]; nb += b[d]*b[d]; }
        float denom = sqrtf(na) * sqrtf(nb);
        scores[k] = (denom > 1e-9f) ? dot / denom : 0.0f;
    }

    /* softmax */
    float mx = scores[0];
    for (int k = 1; k < K; k++) if (scores[k] > mx) mx = scores[k];
    float sm = 0;
    for (int k = 0; k < K; k++) { scores[k] = expf(scores[k] - mx); sm += scores[k]; }
    if (sm > 0) for (int k = 0; k < K; k++) scores[k] /= sm;

    /* argmax */
    int best = 0;
    for (int k = 1; k < K; k++) if (scores[k] > scores[best]) best = k;

    if (out_emb)
        memcpy(out_emb, cs + (size_t)best*D, (size_t)D * sizeof(float));

    free(scores);
    return best;
}

/* ------------------------------------------------------------------ */
/*  Core: vectorize one SQL string → malloc'd float[D], or NULL       */
/* ------------------------------------------------------------------ */
static float *vectorize_single(const char *sql)
{
    /* 1. Preprocess */
    StrArr *tok = preprocess_sql(sql);
    if (!tok || tok->count == 0) { strarr_free(tok); return NULL; }

    int T = tok->count;

    /* 2. Map tokens → model IDs (-1 = UNK) */
    int *mids = malloc((size_t)T * sizeof(int));
    for (int i = 0; i < T; i++) mids[i] = vocab_lookup(tok->data[i]);

    strarr_free(tok);

    /* 3 + 4 + 5 + 6: slide window */
    float *sql_vec = calloc((size_t)g_D, sizeof(float));
    int    n_center = 0;

    for (int ci = 0; ci < T; ci++) {
        int mid = mids[ci];
        if (mid < 0) continue;

        const float *cs = sense_block(mid);
        if (!cs) continue;

        /* collect context */
        int wstart = ci - g_W; if (wstart < 0) wstart = 0;
        int wend   = ci + g_W; if (wend >= T)  wend   = T - 1;

        int M = 0;
        for (int j = wstart; j <= wend; j++) if (j != ci && mids[j] >= 0) M++;

        if (M == 0) {
            /* fallback: sense 0 */
            for (int d = 0; d < g_D; d++) sql_vec[d] += cs[d];
            n_center++; continue;
        }

        /* build ctx [M, K, D] and rel_pos [M] */
        float *ctx  = calloc((size_t)M * g_K * g_D, sizeof(float));
        int   *rp   = malloc((size_t)M * sizeof(int));
        int    mi   = 0;
        for (int j = wstart; j <= wend; j++) {
            if (j == ci || mids[j] < 0) continue;
            const float *sb = sense_block(mids[j]);
            if (sb) memcpy(ctx + (size_t)mi * g_K * g_D, sb, (size_t)g_K * g_D * sizeof(float));
            rp[mi] = j - ci;
            mi++;
        }

        /* attention */
        float *u = NULL, *sense_ctx = NULL;
        bool ok = attention_forward(cs, ctx, rp, g_K, g_D, M, ATTN_SIGMA, &u, &sense_ctx);
        free(ctx); free(rp);
        if (!ok) continue;

        /* gating */
        float *best_emb = malloc((size_t)g_D * sizeof(float));
        gating_network(cs, sense_ctx, g_K, g_D, best_emb);
        free(u); free(sense_ctx);

        for (int d = 0; d < g_D; d++) sql_vec[d] += best_emb[d];
        free(best_emb);
        n_center++;
    }

    free(mids);

    if (n_center == 0) { free(sql_vec); return NULL; }

    /* 7. Average */
    float inv = 1.0f / (float)n_center;
    for (int d = 0; d < g_D; d++) sql_vec[d] *= inv;
    return sql_vec;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */
bool load_sensate_model(const char *vocab_path, const char *emb_path,
                        int num_senses, int embedding_dim, int window_size)
{
    if (g_loaded) return true;

    if (!load_vocab_bin(vocab_path)) return false;
    if (!load_embeddings_bin(emb_path, num_senses, embedding_dim)) return false;

    g_W      = (window_size > 0) ? window_size : DEFAULT_WINDOW;
    g_loaded = true;
    fprintf(stderr, "[vectorize_sql] model ready: V=%d K=%d D=%d W=%d\n",
            g_vocab_n, g_K, g_D, g_W);
    return true;
}

bool load_sensate_model_default(void)
{
    return load_sensate_model(SENSATE_VOCAB_PATH, SENSATE_EMBED_PATH,
                              DEFAULT_NUM_SENSES, DEFAULT_EMBED_DIM, DEFAULT_WINDOW);
}

bool is_model_loaded(void) { return g_loaded; }

void unload_sensate_model(void)
{
    for (int i = 0; i < g_vocab_n; i++) free(g_vocab[i].word);
    free(g_vocab);  g_vocab = NULL;  g_vocab_n = 0;
    free(g_sense_data); g_sense_data = NULL;
    g_K = g_D = 0;
    g_loaded = false;
}

SQLVector *vectorize_sql_queries(const char **queries, int n)
{
    if (!queries || n <= 0) return NULL;
    if (!g_loaded) { fprintf(stderr, "[vectorize_sql] model not loaded\n"); return NULL; }

    SQLVector *out = calloc((size_t)n, sizeof(SQLVector));
    for (int i = 0; i < n; i++) {
        out[i].dim   = g_D;
        out[i].valid = false;
        if (!queries[i]) continue;
        float *v = vectorize_single(queries[i]);
        if (v) { out[i].data = v; out[i].valid = true; }
    }
    return out;
}

void free_sql_vectors(SQLVector *vectors, int n)
{
    if (!vectors) return;
    for (int i = 0; i < n; i++) free(vectors[i].data);
    free(vectors);
}
