/*
 * vectorize_sql.c
 *
 * SQL-to-vector pipeline for the Sensate (multi-sense Word2Vec) model.
 *
 * Pipeline for each query
 * -----------------------
 *  1. preprocess_sql_query()   – normalise SQL into symbolic tokens
 *                                e.g.  SELECT <TAB> <ALIAS_T1> . <COL>
 *  2. model vocab lookup       – map each token → 0-based word_id
 *                                (UNK tokens get id = -1 and are skipped)
 *  3. sliding context window   – for every center position c in [0, T):
 *       * collect context positions  [c-W, c+W] \ {c}  with valid word_ids
 *       * build  center_senses  [K, D]  and  context_senses  [M, K, D]
 *  4. attention_forward()      – compute sense_context [K, D]:
 *                                averaged, distance-biased context per sense
 *  5. gating (inline)          – for each sense k:
 *                                  s[k] = cos( center_senses[k], sense_context[k] )
 *                                q = softmax(s);  best_k = argmax(q)
 *  6. Accumulate best_sense embedding for the center token
 *  7. Average all accumulated center vectors → sql_vector [D]
 *
 * Binary asset formats (little-endian)
 * -------------------------------------
 * vocab.bin:
 *   int32   n_words
 *   for each word:
 *     int32   word_len
 *     char[]  word  (UTF-8, not null-terminated in file)
 *     int32   word_id   (0-based model index)
 *
 * sense_embeddings.bin:
 *   int32   total_records  (= n_vocab * K)
 *   int32   embedding_dim  (= D)
 *   for each record  (outer loop: word_id 0..V-1, inner: sense 0..K-1):
 *     int32   word_len
 *     char[]  word  (UTF-8)
 *     int32   sense_id
 *     float[] embedding  (D × float32)
 */

#include "postgres.h"
#include "utils/palloc.h"

#include "vectorize_sql.h"
#include "sql_preprocessor.h"
#include "model/attention.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ================================================================
 * Compile-time defaults – override with -DSENSATE_xxx=… at build time
 * ================================================================ */
#ifndef SENSATE_VOCAB_PATH
#  ifdef PKGDATADIR
#    define SENSATE_VOCAB_PATH  PKGDATADIR "/semantic_aqo/assets/vocab.bin"
#  else
#    define SENSATE_VOCAB_PATH  "assets/vocab.bin"
#  endif
#endif

#ifndef SENSATE_EMBED_PATH
#  ifdef PKGDATADIR
#    define SENSATE_EMBED_PATH  PKGDATADIR "/semantic_aqo/assets/sense_embeddings.bin"
#  else
#    define SENSATE_EMBED_PATH  "assets/sense_embeddings.bin"
#  endif
#endif

/* Default hyper-parameters (must match training config.yaml) */
#define DEFAULT_NUM_SENSES   3
#define DEFAULT_EMBED_DIM    150
#define DEFAULT_WINDOW_SIZE  7

/* Sigma for the positional distance bias in attention (same as attention.c default) */
#define ATTN_SIGMA  3.0f

/* ================================================================
 * Persistent global model state  (malloc – lives across transactions)
 * ================================================================ */

typedef struct {
    char *word;
    int   word_id;   /* 0-based model index */
} VocabEntry;

typedef struct {
    VocabEntry *entries;
    int         size;
} ModelVocab;

typedef struct {
    float  *data;        /* [vocab_size * num_senses * emb_dim], row-major  */
    int     vocab_size;
    int     num_senses;  /* K */
    int     emb_dim;     /* D */
} ModelEmbeddings;

static ModelVocab       g_vocab  = {NULL, 0};
static ModelEmbeddings  g_embed  = {NULL, 0, 0, 0};
static int              g_window = DEFAULT_WINDOW_SIZE;
static bool             g_loaded = false;

/* ================================================================
 * Internal helpers
 * ================================================================ */

/*
 * model_vocab_lookup
 *   Linear scan through the model vocabulary.
 *   Returns the 0-based word_id, or -1 if not found (UNK).
 *
 *   Performance note: for typical SQL token vocabularies (hundreds of
 *   entries) a linear scan is fast enough and avoids extra data structures.
 *   Replace with a hash table if vocab size becomes large.
 */
static int
model_vocab_lookup(const char *word)
{
    if (!word) return -1;
    for (int i = 0; i < g_vocab.size; i++)
    {
        if (strcmp(g_vocab.entries[i].word, word) == 0)
            return g_vocab.entries[i].word_id;
    }
    return -1;
}

/*
 * get_sense_embeddings
 *   Returns a pointer to the start of the K×D block for word_id in the
 *   flat sense-embedding table.  The layout is:
 *     sense_data[ word_id * K * D + sense_id * D + d ]
 */
static const float *
get_sense_embeddings(int word_id)
{
    if (word_id < 0 || word_id >= g_embed.vocab_size)
        return NULL;
    int K = g_embed.num_senses;
    int D = g_embed.emb_dim;
    return g_embed.data + (size_t)word_id * K * D;
}

/* Cosine similarity between two float vectors of length dim */
static float
cosine_sim_f(const float *a, const float *b, int dim)
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-10f) ? (dot / denom) : 0.0f;
}

/* In-place softmax on a float array of length n */
static void
softmax_f(float *arr, int n)
{
    float maxv = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > maxv) maxv = arr[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        arr[i] = expf(arr[i] - maxv);
        sum   += arr[i];
    }
    if (sum > 0.0f)
        for (int i = 0; i < n; i++)
            arr[i] /= sum;
}

/* Index of maximum value in a float array of length n */
static int
argmax_f(const float *arr, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (arr[i] > arr[best])
            best = i;
    return best;
}

/* ================================================================
 * Binary asset loaders  (malloc – called once at startup)
 * ================================================================ */

static bool
load_vocab_bin(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        elog(WARNING, "vectorize_sql: cannot open vocab file: %s", path);
        return false;
    }

    int32_t n = 0;
    if (fread(&n, sizeof(int32_t), 1, f) != 1 || n <= 0)
    {
        elog(WARNING, "vectorize_sql: vocab.bin: bad record count");
        fclose(f);
        return false;
    }

    g_vocab.entries = (VocabEntry *) malloc((size_t)n * sizeof(VocabEntry));
    if (!g_vocab.entries)
    {
        fclose(f);
        return false;
    }
    memset(g_vocab.entries, 0, (size_t)n * sizeof(VocabEntry));
    g_vocab.size = (int)n;

    for (int i = 0; i < (int)n; i++)
    {
        int32_t wlen = 0;
        if (fread(&wlen, sizeof(int32_t), 1, f) != 1 || wlen <= 0 || wlen > 1024)
        {
            elog(WARNING, "vectorize_sql: vocab.bin: bad word length at entry %d", i);
            fclose(f);
            return false;
        }

        g_vocab.entries[i].word = (char *) malloc((size_t)(wlen + 1));
        if (!g_vocab.entries[i].word)
        {
            fclose(f);
            return false;
        }

        if ((int)fread(g_vocab.entries[i].word, 1, (size_t)wlen, f) != wlen)
        {
            elog(WARNING, "vectorize_sql: vocab.bin: truncated word at entry %d", i);
            fclose(f);
            return false;
        }
        g_vocab.entries[i].word[wlen] = '\0';

        int32_t wid = 0;
        if (fread(&wid, sizeof(int32_t), 1, f) != 1)
        {
            elog(WARNING, "vectorize_sql: vocab.bin: truncated word_id at entry %d", i);
            fclose(f);
            return false;
        }
        g_vocab.entries[i].word_id = (int)wid;
    }

    fclose(f);
    elog(DEBUG1, "vectorize_sql: loaded %d vocab entries from %s", (int)n, path);
    return true;
}

static bool
load_embeddings_bin(const char *path, int num_senses, int emb_dim)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        elog(WARNING, "vectorize_sql: cannot open embeddings file: %s", path);
        return false;
    }

    int32_t total_records = 0, file_dim = 0;
    if (fread(&total_records, sizeof(int32_t), 1, f) != 1 ||
        fread(&file_dim,      sizeof(int32_t), 1, f) != 1)
    {
        elog(WARNING, "vectorize_sql: sense_embeddings.bin: missing header");
        fclose(f);
        return false;
    }

    /* Trust the file's embedding_dim over the caller's hint */
    if (file_dim != emb_dim)
    {
        elog(WARNING,
             "vectorize_sql: embedding dim mismatch – file says %d, expected %d; using %d",
             (int)file_dim, emb_dim, (int)file_dim);
        emb_dim = (int)file_dim;
    }

    int V = g_vocab.size;
    int K = num_senses;

    if (V <= 0)
    {
        elog(WARNING, "vectorize_sql: vocab must be loaded before embeddings");
        fclose(f);
        return false;
    }

    /*
     * Sanity check: total_records should equal V * K.
     * If it doesn't, try to derive K from the record count.
     */
    if (total_records != V * K)
    {
        if (total_records > 0 && V > 0 && (total_records % V) == 0)
        {
            K = total_records / V;
            elog(WARNING,
                 "vectorize_sql: adjusting num_senses from %d to %d based on file contents",
                 num_senses, K);
        }
        else
        {
            elog(WARNING,
                 "vectorize_sql: record count %d doesn't match V=%d × K=%d; loading anyway",
                 (int)total_records, V, K);
        }
    }

    g_embed.vocab_size = V;
    g_embed.num_senses = K;
    g_embed.emb_dim    = emb_dim;

    size_t total_floats = (size_t)V * K * emb_dim;
    g_embed.data = (float *) calloc(total_floats, sizeof(float));
    if (!g_embed.data)
    {
        elog(WARNING, "vectorize_sql: out of memory for sense embeddings");
        fclose(f);
        return false;
    }

    /* Temporary read buffer (one embedding at a time) */
    float *emb_buf = (float *) malloc((size_t)emb_dim * sizeof(float));
    if (!emb_buf)
    {
        fclose(f);
        return false;
    }

    int loaded = 0;
    for (int rec = 0; rec < (int)total_records; rec++)
    {
        /* word string */
        int32_t wlen = 0;
        if (fread(&wlen, sizeof(int32_t), 1, f) != 1 || wlen <= 0 || wlen > 1024)
        {
            elog(WARNING, "vectorize_sql: sense_embeddings.bin: bad word length at rec %d", rec);
            break;
        }
        char word[1025];
        if ((int)fread(word, 1, (size_t)wlen, f) != wlen) break;
        word[wlen] = '\0';

        /* sense_id */
        int32_t sense_id = 0;
        if (fread(&sense_id, sizeof(int32_t), 1, f) != 1) break;

        /* embedding */
        if ((int)fread(emb_buf, sizeof(float), (size_t)emb_dim, f) != emb_dim) break;

        /* Place into sense table via model vocab_id */
        int word_id = model_vocab_lookup(word);
        if (word_id < 0 || word_id >= V || sense_id < 0 || sense_id >= K)
            continue;  /* unknown word or out-of-range sense – skip silently */

        float *dst = g_embed.data + ((size_t)word_id * K + sense_id) * emb_dim;
        memcpy(dst, emb_buf, (size_t)emb_dim * sizeof(float));
        loaded++;
    }

    free(emb_buf);
    fclose(f);
    elog(DEBUG1,
         "vectorize_sql: loaded %d / %d sense embedding records from %s",
         loaded, (int)total_records, path);
    return true;
}

/* ================================================================
 * Public: model lifecycle
 * ================================================================ */

bool
load_sensate_model(const char *vocab_path,
                   const char *emb_path,
                   int         num_senses,
                   int         embedding_dim,
                   int         window_size)
{
    if (g_loaded)
        return true;  /* idempotent */

    if (!load_vocab_bin(vocab_path))
        return false;

    if (!load_embeddings_bin(emb_path, num_senses, embedding_dim))
        return false;

    g_window = (window_size > 0) ? window_size : DEFAULT_WINDOW_SIZE;
    g_loaded = true;

    elog(LOG,
         "vectorize_sql: model loaded – vocab=%d, senses=%d, dim=%d, window=%d",
         g_vocab.size, g_embed.num_senses, g_embed.emb_dim, g_window);
    return true;
}

bool
load_sensate_model_default(void)
{
    return load_sensate_model(SENSATE_VOCAB_PATH,
                              SENSATE_EMBED_PATH,
                              DEFAULT_NUM_SENSES,
                              DEFAULT_EMBED_DIM,
                              DEFAULT_WINDOW_SIZE);
}

bool
is_model_loaded(void)
{
    return g_loaded;
}

void
unload_sensate_model(void)
{
    if (g_vocab.entries)
    {
        for (int i = 0; i < g_vocab.size; i++)
            free(g_vocab.entries[i].word);
        free(g_vocab.entries);
        g_vocab.entries = NULL;
        g_vocab.size    = 0;
    }

    if (g_embed.data)
    {
        free(g_embed.data);
        g_embed.data       = NULL;
        g_embed.vocab_size = 0;
        g_embed.num_senses = 0;
        g_embed.emb_dim    = 0;
    }

    g_loaded = false;
    elog(DEBUG1, "vectorize_sql: model unloaded");
}

/* ================================================================
 * Core: vectorize a single SQL query
 *
 * Returns a palloc'd float[D] vector, or NULL if the query is empty
 * or produces no valid center tokens.
 * ================================================================ */

static float *
vectorize_single(const char *sql)
{
    int K = g_embed.num_senses;
    int D = g_embed.emb_dim;

    if (!sql || !g_loaded || K == 0 || D == 0)
        return NULL;

    /* ---- Step 1: Preprocess SQL → symbolic token array ---- */
    SQLPreprocessingResult *prep = preprocess_sql_query(sql);
    if (!prep || !prep->success || !prep->tokens || prep->tokens->count == 0)
    {
        if (prep)
            free_sql_preprocessing_result(prep);
        return NULL;
    }

    TokenArray *tok_arr = prep->tokens;
    int         T       = (int)tok_arr->count;

    /* ---- Step 2: Map tokens → model vocab_ids  (-1 = UNK) ---- */
    int *ids = (int *) palloc((size_t)T * sizeof(int));
    for (int i = 0; i < T; i++)
        ids[i] = model_vocab_lookup(tok_arr->tokens[i]);

    /* ---- Steps 3-6: Sliding window over center positions ---- */
    float *sql_vec           = (float *) palloc0((size_t)D * sizeof(float));
    int    valid_center_count = 0;

    for (int ci = 0; ci < T; ci++)
    {
        int center_id = ids[ci];
        if (center_id < 0)
            continue;  /* UNK center token – skip */

        const float *center_senses = get_sense_embeddings(center_id);
        if (!center_senses)
            continue;

        /* Collect context positions inside the window, excluding UNK */
        int wstart = ci - g_window;
        int wend   = ci + g_window;
        if (wstart < 0)  wstart = 0;
        if (wend >= T)   wend   = T - 1;

        int M = 0;  /* number of valid context positions */
        for (int j = wstart; j <= wend; j++)
        {
            if (j == ci)      continue;
            if (ids[j] >= 0)  M++;
        }

        /*
         * No context available: fall back to sense-0 of the center word.
         * (Rare for normal SQL; handles single-token edge cases.)
         */
        if (M == 0)
        {
            for (int d = 0; d < D; d++)
                sql_vec[d] += center_senses[d];   /* sense 0 */
            valid_center_count++;
            continue;
        }

        /*
         * Build context_senses [M, K, D] and rel_pos [M] for attention.
         * We only include context positions with a known vocab_id.
         */
        float *ctx_senses = (float *) palloc0((size_t)M * K * D * sizeof(float));
        int   *rel_pos    = (int *)   palloc ((size_t)M * sizeof(int));
        int    mi         = 0;

        for (int j = wstart; j <= wend; j++)
        {
            if (j == ci || ids[j] < 0)
                continue;

            const float *cs = get_sense_embeddings(ids[j]);
            if (cs)
                memcpy(ctx_senses + (size_t)mi * K * D, cs,
                       (size_t)K * D * sizeof(float));
            /* else: zero block already from palloc0 */

            rel_pos[mi] = j - ci;
            mi++;
        }

        /* ---- Step 4: Attention forward ----
         *
         * Inputs:
         *   center_senses  [K, D]
         *   context_senses [M, K, D]
         *   rel_pos        [M]
         *
         * Output:
         *   attn_out->sense_context  [K, D]
         *     = for each sense k: Σ_j( weight[k,j] * u[j] )
         *     where u[j] = mean over senses of context_senses[j,*,*]
         */
        AttentionInput attn_in;
        memset(&attn_in, 0, sizeof(attn_in));
        attn_in.center_id      = center_id;
        attn_in.context_ids    = NULL;          /* not used by attention.c */
        attn_in.rel_pos        = rel_pos;
        attn_in.context_count  = (size_t)M;
        attn_in.num_senses     = K;
        attn_in.dim            = D;
        attn_in.center_senses  = center_senses;
        attn_in.context_senses = ctx_senses;
        attn_in.sigma          = ATTN_SIGMA;

        AttentionOutput *attn_out = attention_forward(&attn_in);
        if (!attn_out)
        {
            pfree(ctx_senses);
            pfree(rel_pos);
            continue;
        }

        /* ---- Step 5: Gating ----
         *
         * For each sense k:
         *   s[k] = cosine_sim( center_senses[k, :],  sense_context[k, :] )
         * q = softmax(s)
         * best_k = argmax(q)
         */
        float *scores = (float *) palloc((size_t)K * sizeof(float));

        for (int k = 0; k < K; k++)
        {
            scores[k] = cosine_sim_f(
                center_senses              + (size_t)k * D,
                attn_out->sense_context    + (size_t)k * D,
                D
            );
        }

        softmax_f(scores, K);
        int best_k = argmax_f(scores, K);

        /* ---- Step 6: Accumulate best sense embedding ---- */
        const float *best_sense = center_senses + (size_t)best_k * D;
        for (int d = 0; d < D; d++)
            sql_vec[d] += best_sense[d];
        valid_center_count++;

        /* Cleanup temporaries */
        pfree(scores);
        attention_free_output(attn_out);
        pfree(ctx_senses);
        pfree(rel_pos);
    }

    pfree(ids);
    free_sql_preprocessing_result(prep);

    if (valid_center_count == 0)
    {
        pfree(sql_vec);
        return NULL;
    }

    /* ---- Step 7: Average center embeddings ---- */
    float inv = 1.0f / (float)valid_center_count;
    for (int d = 0; d < D; d++)
        sql_vec[d] *= inv;

    return sql_vec;  /* palloc'd in the caller's memory context */
}

/* ================================================================
 * Public: vectorize_sql_queries
 * ================================================================ */

SQLVector *
vectorize_sql_queries(const char **queries, int n_queries)
{
    if (!queries || n_queries <= 0)
        return NULL;

    if (!g_loaded)
    {
        elog(WARNING,
             "vectorize_sql_queries: model not loaded – "
             "call load_sensate_model() or load_sensate_model_default() first");
        return NULL;
    }

    int        D       = g_embed.emb_dim;
    SQLVector *results = (SQLVector *) palloc0((size_t)n_queries * sizeof(SQLVector));

    for (int i = 0; i < n_queries; i++)
    {
        results[i].dim   = D;
        results[i].valid = false;
        results[i].data  = NULL;

        if (!queries[i])
            continue;

        float *vec = vectorize_single(queries[i]);
        if (vec)
        {
            results[i].data  = vec;
            results[i].valid = true;
        }
    }

    return results;
}

void
free_sql_vectors(SQLVector *vectors, int n_queries)
{
    if (!vectors)
        return;

    for (int i = 0; i < n_queries; i++)
    {
        if (vectors[i].data)
            pfree(vectors[i].data);
    }
    pfree(vectors);
}
