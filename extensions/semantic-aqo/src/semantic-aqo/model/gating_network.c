#include "gating_network.h"
#include "utils/pg_compat.h"

#include <math.h>
#include <string.h>

#define IDX_KD(k,d,D) ((size_t)(k) * (size_t)(D) + (size_t)(d))
#define IDX_KM(k,m,M) ((size_t)(k) * (size_t)(M) + (size_t)(m))
#define IDX_MD(m,d,D) ((size_t)(m) * (size_t)(D) + (size_t)(d))

static float dot_product(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += a[i] * b[i];
    return sum;
}

static float vector_norm(const float *vec, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += vec[i] * vec[i];
    return sqrtf(sum);
}

static float cosine_similarity(const float *a, const float *b, int dim) {
    float dot = dot_product(a, b, dim);
    float norm_a = vector_norm(a, dim);
    float norm_b = vector_norm(b, dim);
    
    if (norm_a < 1e-10f || norm_b < 1e-10f) return 0.0f;
    return dot / (norm_a * norm_b);
}

static void softmax(const float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

static int argmax(const float *arr, int size) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int gating_network(
    const float *attention_weights,
    const float *context_u,
    const float *center_senses,
    int num_senses,
    int num_context,
    int embedding_dim,
    float *output_embedding,
    float *output_probs
) {
    int K = num_senses;
    int M = num_context;
    int D = embedding_dim;

    float *context_vectors = (float *)palloc0((size_t)K * (size_t)D * sizeof(float));
    
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < M; j++) {
            float weight = attention_weights[IDX_KM(k, j, M)];
            for (int d = 0; d < D; d++) {
                context_vectors[IDX_KD(k, d, D)] += weight * context_u[IDX_MD(j, d, D)];
            }
        }
    }
    
    float *gating_scores = (float *)palloc((size_t)K * sizeof(float));
    for (int k = 0; k < K; k++) {
        const float *center_k = &center_senses[IDX_KD(k, 0, D)];
        const float *ctx_k = &context_vectors[IDX_KD(k, 0, D)];
        gating_scores[k] = cosine_similarity(center_k, ctx_k, D);
    }
    
    float *gating_probs = (float *)palloc((size_t)K * sizeof(float));
    softmax(gating_scores, gating_probs, K);
    
    int best_sense_idx = argmax(gating_probs, K);
    
    memcpy(output_embedding, &center_senses[IDX_KD(best_sense_idx, 0, D)], D * sizeof(float));
    
    if (output_probs != NULL) {
        memcpy(output_probs, gating_probs, K * sizeof(float));
    }
    
    pfree(context_vectors);
    pfree(gating_scores);
    pfree(gating_probs);
    
    return best_sense_idx;
}

float gating_network_max_score(
    const float *attention_weights,
    const float *context_u,
    const float *center_senses,
    int num_senses,
    int num_context,
    int embedding_dim
) {
    int K = num_senses;
    int M = num_context;
    int D = embedding_dim;

    float *context_vectors = (float *)palloc0((size_t)K * (size_t)D * sizeof(float));
    
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < M; j++) {
            float weight = attention_weights[IDX_KM(k, j, M)];
            for (int d = 0; d < D; d++) {
                context_vectors[IDX_KD(k, d, D)] += weight * context_u[IDX_MD(j, d, D)];
            }
        }
    }
    
    float max_score = -INFINITY;
    for (int k = 0; k < K; k++) {
        const float *center_k = &center_senses[IDX_KD(k, 0, D)];
        const float *ctx_k = &context_vectors[IDX_KD(k, 0, D)];
        float score = cosine_similarity(center_k, ctx_k, D);
        if (score > max_score) max_score = score;
    }
    
    pfree(context_vectors);
    return max_score;
}