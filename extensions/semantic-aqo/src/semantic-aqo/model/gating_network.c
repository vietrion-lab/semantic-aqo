// src/semantic-aqo/model/gating_network.c
// Gating Network for PostgreSQL extension

#include "postgres.h"
#include "fmgr.h"
#include <math.h>

#include "model/gating_network.h"

/**
 * Compute dot product of two vectors
 */
static double dot_product(const double *a, const double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * Compute L2 norm of a vector
 */
static double vector_norm(const double *vec, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

/**
 * Compute cosine similarity between two vectors
 * cos(a, b) = (a · b) / (||a|| * ||b||)
 */
static double cosine_similarity(const double *a, const double *b, int dim) {
    double dot = dot_product(a, b, dim);
    double norm_a = vector_norm(a, dim);
    double norm_b = vector_norm(b, dim);
    
    // Avoid division by zero
    if (norm_a < 1e-10 || norm_b < 1e-10) {
        return 0.0;
    }
    
    return dot / (norm_a * norm_b);
}

/**
 * Apply softmax to an array
 * softmax(x_i) = exp(x_i) / Σ exp(x_j)
 */
static void softmax(const double *input, double *output, int size) {
    // Find max for numerical stability
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/**
 * Find the index of maximum value in an array
 */
static int argmax(const double *arr, int size) {
    int max_idx = 0;
    double max_val = arr[0];
    
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

/**
 * Gating Network Implementation
 * 
 * This implements the gating mechanism for sense selection:
 * 1. Compute sense-specific context vectors: c_k = Σ_j (v_j * α_jk)
 * 2. Compute gating scores: s_k = cos(center_sense_k, c_k)
 * 3. Apply softmax: q = softmax(s)
 * 4. Select best sense: argmax(q)
 * 
 * @param gate_input: Contains attention_score [num_context, num_senses] and 
 *                    context_value [num_context, embedding_dim]
 * @param center_sense_embeddings: [num_senses, embedding_dim]
 * @param num_senses: Number of senses (K)
 * @param num_context: Number of context words (T-1)
 * @param embedding_dim: Dimension of embeddings (D)
 * @param output_embedding: Output buffer [embedding_dim] - the selected sense
 * @param output_probs: Output buffer [num_senses] - gating probabilities (can be NULL)
 * 
 * @return: Index of the selected sense
 */
int gating_network(
    const GateInput *gate_input,
    const double **center_sense_embeddings,
    int num_senses,
    int num_context,
    int embedding_dim,
    double *output_embedding,
    double *output_probs
) {
    // Step 1: Compute sense-specific context vectors
    // c_k = Σ_j (v_j * α_jk) where j ∈ context words, k ∈ senses
    // Result: [num_senses, embedding_dim]
    
    double **context_vectors = (double **)palloc(num_senses * sizeof(double *));
    for (int k = 0; k < num_senses; k++) {
        context_vectors[k] = (double *)palloc0(embedding_dim * sizeof(double));
        
        // For each sense k, compute weighted sum of context values
        for (int j = 0; j < num_context; j++) {
            double weight = gate_input->attention_score[j][k];  // α_jk
            
            for (int d = 0; d < embedding_dim; d++) {
                context_vectors[k][d] += weight * gate_input->context_value[j][d];
            }
        }
    }
    
    // Step 2: Compute gating scores using cosine similarity
    // s_k = cos(center_sense_k, c_k)
    double *gating_scores = (double *)palloc(num_senses * sizeof(double));
    
    for (int k = 0; k < num_senses; k++) {
        gating_scores[k] = cosine_similarity(
            center_sense_embeddings[k],
            context_vectors[k],
            embedding_dim
        );
    }
    
    // Step 3: Apply softmax to get probability distribution
    // q = softmax(s)
    double *gating_probs = (double *)palloc(num_senses * sizeof(double));
    softmax(gating_scores, gating_probs, num_senses);
    
    // Step 4: Select the sense with highest probability
    int best_sense_idx = argmax(gating_probs, num_senses);
    
    // Copy the selected sense embedding to output
    memcpy(output_embedding, center_sense_embeddings[best_sense_idx], 
           embedding_dim * sizeof(double));
    
    // Optionally copy probabilities to output
    if (output_probs != NULL) {
        memcpy(output_probs, gating_probs, num_senses * sizeof(double));
    }
    
    // Cleanup
    for (int k = 0; k < num_senses; k++) {
        pfree(context_vectors[k]);
    }
    pfree(context_vectors);
    pfree(gating_scores);
    pfree(gating_probs);
    
    return best_sense_idx;
}

/**
 * Alternative version that returns the max score instead of embedding
 * This matches Step 7 in the algorithm: μ_cw = max_{1≤k≤K} s_k
 */
double gating_network_max_score(
    const GateInput *gate_input,
    const double **center_sense_embeddings,
    int num_senses,
    int num_context,
    int embedding_dim
) {
    // Compute sense-specific context vectors
    double **context_vectors = (double **)palloc(num_senses * sizeof(double *));
    for (int k = 0; k < num_senses; k++) {
        context_vectors[k] = (double *)palloc0(embedding_dim * sizeof(double));
        
        for (int j = 0; j < num_context; j++) {
            double weight = gate_input->attention_score[j][k];
            
            for (int d = 0; d < embedding_dim; d++) {
                context_vectors[k][d] += weight * gate_input->context_value[j][d];
            }
        }
    }
    
    // Compute gating scores
    double max_score = -INFINITY;
    
    for (int k = 0; k < num_senses; k++) {
        double score = cosine_similarity(
            center_sense_embeddings[k],
            context_vectors[k],
            embedding_dim
        );
        
        if (score > max_score) {
            max_score = score;
        }
    }
    
    // Cleanup
    for (int k = 0; k < num_senses; k++) {
        pfree(context_vectors[k]);
    }
    pfree(context_vectors);
    
    return max_score;
}
