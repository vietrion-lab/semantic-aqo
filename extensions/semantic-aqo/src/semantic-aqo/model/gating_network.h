#ifndef GATING_NETWORK_H
#define GATING_NETWORK_H

/**
 * Gating Network for multi-sense word disambiguation
 * * Computes the best sense embedding for a center word given context.
 * * @param attention_weights: [K, M] softmax attention weights from Attention
 * @param context_u: [M, D] averaged context vectors from Attention
 * @param center_senses: [K, D] all sense embeddings for center word
 * @param num_senses: Number of senses (K)
 * @param num_context: Number of context words (M)
 * @param embedding_dim: Dimension of embeddings (D)
 * @param output_embedding: Output buffer [D] - the selected sense embedding
 * @param output_probs: Optional output buffer [K] - gating probabilities (can be NULL)
 * @return: Index of the selected sense
 */
int gating_network(
    const float *attention_weights,
    const float *context_u,
    const float *center_senses,
    int num_senses,
    int num_context,
    int embedding_dim,
    float *output_embedding,
    float *output_probs
);

/**
 * Alternative version that returns the maximum gating score
 */
float gating_network_max_score(
    const float *attention_weights,
    const float *context_u,
    const float *center_senses,
    int num_senses,
    int num_context,
    int embedding_dim
);

#endif /* GATING_NETWORK_H */