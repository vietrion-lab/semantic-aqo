#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>

/*
Input tensor layout:
- center_senses: [K, D] row-major
- context_senses: [M, K, D] row-major
  index = ((j * K + k) * D + d)
*/
typedef struct {
    int center_id;
    const int *context_ids;
    const int *rel_pos;          /* length M (for distance bias) */
    size_t context_count;        /* M */

    int num_senses;              /* K */
    int dim;                     /* D */

    const float *center_senses;  /* size K*D */
    const float *context_senses; /* size M*K*D */

    float sigma;                 /* distance bias sigma (>0 to enable) */
} AttentionInput;

typedef struct {
    int num_senses;              /* K */
    int dim;                     /* D */
    size_t context_count;        /* M */

    float *u;            /* [M, D] averaged context vectors */
    float *scores;       /* [K, M] pre-softmax scores */
    float *weights;      /* [K, M] softmax attention */
    float *sense_context;/* [K, D] weighted context per sense */
} AttentionOutput;

typedef struct {
    const AttentionInput *in;
    AttentionOutput *out;
} AttentionResult;

AttentionOutput *attention_forward(const AttentionInput *in);
void attention_free_output(AttentionOutput *out);

AttentionResult *attention_run(const AttentionInput *in);
void attention_free_result(AttentionResult *res);

#endif /* ATTENTION_H */
