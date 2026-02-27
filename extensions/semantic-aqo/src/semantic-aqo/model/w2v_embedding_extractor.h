#ifndef W2V_EMBEDDING_EXTRACTOR_H
#define W2V_EMBEDDING_EXTRACTOR_H

#include <stdbool.h>

bool init_embedding_extractor(const char *v_path, const char *e_path, int k, int d);
void free_embedding_extractor(void);
int extractor_get_word_id(const char *word);
const float* extractor_get_sense_embeddings(int word_id);

/* FAIR AVERAGE: Compute average vector from aggregated word vectors */
float* extractor_compute_fair_average(const float *agg_vec, int num_words, int d);

int extractor_get_dim(void);
int extractor_get_num_senses(void);
bool extractor_is_loaded(void);

#endif