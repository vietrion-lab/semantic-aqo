#include "../utils/pg_compat.h"
#include "w2v_embedding_extractor.h"
#include <stdint.h>

typedef struct { char *word; int id; } VocabEntry;
static VocabEntry *g_v = NULL;
static float *g_e = NULL;
static int g_size = 0, g_k = 0, g_d = 0;
static bool g_init = false;

bool init_embedding_extractor(const char *v_p, const char *e_p, int k, int d) {
    if (g_init) return true;
    FILE *fv = fopen(v_p, "rb");
    if (!fv) return false;
    int32_t n; fread(&n, 4, 1, fv);
    g_v = palloc0(n * sizeof(VocabEntry));
    for (int i = 0; i < n; i++) {
        int32_t wl; fread(&wl, 4, 1, fv);
        g_v[i].word = palloc(wl + 1);
        fread(g_v[i].word, 1, wl, fv);
        g_v[i].word[wl] = '\0';
        fread(&g_v[i].id, 4, 1, fv);
    }
    fclose(fv);
    FILE *fe = fopen(e_p, "rb");
    if (!fe) return false;
    int32_t tr, fd; fread(&tr, 4, 1, fe); fread(&fd, 4, 1, fe);
    g_k = k; g_d = fd; g_size = n;
    g_e = palloc0((size_t)n * k * fd * sizeof(float));
    float *tmp = palloc(fd * sizeof(float));
    for (int r = 0; r < tr; r++) {
        int32_t wl; fread(&wl, 4, 1, fe); fseek(fe, wl, SEEK_CUR);
        int32_t sid; fread(&sid, 4, 1, fe); fread(tmp, 4, fd, fe);
        int wid = r / k;
        if (wid < n && sid < k) memcpy(g_e + (wid * k + sid) * fd, tmp, fd * sizeof(float));
    }
    pfree(tmp); fclose(fe);
    g_init = true; return true;
}

float* extractor_compute_fair_average(const float *agg_vec, int num_words, int d) {
    if (!agg_vec || num_words <= 0) return NULL;
    float *query_vec = palloc0(d * sizeof(float));
    for (int w = 0; w < num_words; w++) {
        for (int i = 0; i < d; i++) query_vec[i] += agg_vec[w * d + i];
    }
    for (int i = 0; i < d; i++) query_vec[i] /= (float)num_words;
    return query_vec;
}

int extractor_get_word_id(const char *w) {
    for (int i = 0; i < g_size; i++) if (strcmp(g_v[i].word, w) == 0) return i;
    return -1;
}

const float* extractor_get_sense_embeddings(int wid) {
    return (wid < 0 || wid >= g_size) ? NULL : g_e + (wid * g_k * g_d);
}

void free_embedding_extractor() {
    if (!g_init) return;
    for (int i = 0; i < g_size; i++) pfree(g_v[i].word);
    pfree(g_v); pfree(g_e); g_init = false;
}

int extractor_get_dim() { return g_d; }
int extractor_get_num_senses() { return g_k; }
bool extractor_is_loaded() { return g_init; }