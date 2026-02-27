#include "attention.h"
#include "utils/pg_compat.h"

#include <math.h>
#include <stdlib.h>

#define IDX_U(j,d,D)              ((size_t)(j) * (size_t)(D) + (size_t)(d))
#define IDX_KM(k,m,M)             ((size_t)(k) * (size_t)(M) + (size_t)(m))
#define IDX_JKD(j,k,d,K,D)        (((size_t)(j) * (size_t)(K) * (size_t)(D)) + ((size_t)(k) * (size_t)(D)) + (size_t)(d))
#define IDX_KD(k,d,D)             ((size_t)(k) * (size_t)(D) + (size_t)(d))

#define DEFAULT_SIGMA 3.0f

static int validate_input(const AttentionInput *in) {
    if (!in) return 0;
    if (in->num_senses <= 0 || in->dim <= 0) return 0;
    if (in->context_count == 0) return 0;
    if (!in->center_senses || !in->context_senses) return 0;
    return 1;
}

static float distance_bias(int rel_pos, float sigma) {
    if (sigma <= 0.0f) sigma = DEFAULT_SIGMA;
    float x = (float)abs(rel_pos) / sigma;
    return -0.5f * x * x;
}

static int compute_context_average(const AttentionInput *in, AttentionOutput *out) {
    int K = in->num_senses;
    int D = in->dim;
    size_t M = in->context_count;

    for (size_t j = 0; j < M; j++) {
        for (int d = 0; d < D; d++) {
            float sum = 0.0f;
            for (int r = 0; r < K; r++) {
                sum += in->context_senses[IDX_JKD(j, r, d, K, D)];
            }
            out->u[IDX_U(j, d, D)] = sum / (float)K;
        }
    }
    return 1;
}

static int compute_scores(const AttentionInput *in, AttentionOutput *out) {
    int K = in->num_senses;
    int D = in->dim;
    size_t M = in->context_count;

    float scale = sqrtf((float)D);
    if (scale <= 0.0f) return 0;

    for (int k = 0; k < K; k++) {
        for (size_t j = 0; j < M; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                float q = in->center_senses[IDX_KD(k, d, D)];
                float u = out->u[IDX_U(j, d, D)];
                dot += q * u;
            }

            float b = 0.0f;
            if (in->rel_pos) b = distance_bias(in->rel_pos[j], in->sigma);

            out->scores[IDX_KM(k, j, M)] = (dot / scale) + b;
        }
    }
    return 1;
}

static int softmax_rows(AttentionOutput *out) {
    int K = out->num_senses;
    size_t M = out->context_count;

    for (int k = 0; k < K; k++) {
        float maxv = out->scores[IDX_KM(k, 0, M)];

        for (size_t j = 1; j < M; j++) {
            float v = out->scores[IDX_KM(k, j, M)];
            if (v > maxv) maxv = v;
        }

        float sum = 0.0f;
        for (size_t j = 0; j < M; j++) {
            float e = expf(out->scores[IDX_KM(k, j, M)] - maxv);
            out->weights[IDX_KM(k, j, M)] = e;
            sum += e;
        }

        if (sum <= 0.0f || !isfinite(sum)) {
            float uni = 1.0f / (float)M;
            for (size_t j = 0; j < M; j++) out->weights[IDX_KM(k, j, M)] = uni;
        } else {
            for (size_t j = 0; j < M; j++) out->weights[IDX_KM(k, j, M)] /= sum;
        }
    }
    return 1;
}

static int compute_weighted_context(AttentionOutput *out) {
    int K = out->num_senses;
    int D = out->dim;
    size_t M = out->context_count;

    for (int k = 0; k < K; k++) {
        for (int d = 0; d < D; d++) {
            float s = 0.0f;
            for (size_t j = 0; j < M; j++) {
                float a = out->weights[IDX_KM(k, j, M)];
                float u = out->u[IDX_U(j, d, D)];
                s += a * u;
            }
            out->sense_context[IDX_KD(k, d, D)] = s;
        }
    }
    return 1;
}

AttentionOutput *attention_forward(const AttentionInput *in) {
    if (!validate_input(in)) return NULL;

    AttentionOutput *out = (AttentionOutput*) palloc(sizeof(AttentionOutput));
    if (!out) return NULL;

    int K = in->num_senses;
    int D = in->dim;
    size_t M = in->context_count;

    out->num_senses = K;
    out->dim = D;
    out->context_count = M;

    out->u = (float*) palloc0(sizeof(float) * M * D);
    out->scores = (float*) palloc0(sizeof(float) * K * M);
    out->weights = (float*) palloc0(sizeof(float) * K * M);
    out->sense_context = (float*) palloc0(sizeof(float) * K * D);

    if (!out->u || !out->scores || !out->weights || !out->sense_context) {
        attention_free_output(out);
        return NULL;
    }

    if (!compute_context_average(in, out) ||
        !compute_scores(in, out) ||
        !softmax_rows(out) ||
        !compute_weighted_context(out)) {
        attention_free_output(out);
        return NULL;
    }

    return out;
}

void attention_free_output(AttentionOutput *out) {
    if (!out) return;
    pfree(out->u);
    pfree(out->scores);
    pfree(out->weights);
    pfree(out->sense_context);
    pfree(out);
}

AttentionResult *attention_run(const AttentionInput *in) {
    AttentionOutput *out = attention_forward(in);
    if (!out) return NULL;

    AttentionResult *res = (AttentionResult*) palloc(sizeof(AttentionResult));
    if (!res) {
        attention_free_output(out);
        return NULL;
    }

    res->in = in;
    res->out = out;
    return res;
}

void attention_free_result(AttentionResult *res) {
    if (!res) return;
    attention_free_output(res->out);
    pfree(res);
}