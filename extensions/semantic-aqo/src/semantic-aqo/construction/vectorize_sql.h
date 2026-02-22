/*
 * vectorize_sql.h  –  standalone SQL-to-vector interface
 *
 * No PostgreSQL dependency.  Link against vectorize_sql.c only.
 *
 * Usage:
 *   load_sensate_model(vocab_path, emb_path, K, D, window);
 *   SQLVector *vecs = vectorize_sql_queries(queries, n);
 *   // use vecs[i].data  (float[vecs[i].dim])
 *   free_sql_vectors(vecs, n);
 *   unload_sensate_model();
 */
#ifndef VECTORIZE_SQL_H
#define VECTORIZE_SQL_H

#include <stdbool.h>

typedef struct {
    float *data;   /* malloc'd float[dim]; NULL when valid == false */
    int    dim;
    bool   valid;
} SQLVector;

bool       load_sensate_model(const char *vocab_path, const char *emb_path,
                               int num_senses, int embedding_dim, int window_size);
bool       load_sensate_model_default(void);
bool       is_model_loaded(void);
void       unload_sensate_model(void);

SQLVector *vectorize_sql_queries(const char **queries, int n_queries);
void       free_sql_vectors(SQLVector *vectors, int n_queries);

#endif /* VECTORIZE_SQL_H */
