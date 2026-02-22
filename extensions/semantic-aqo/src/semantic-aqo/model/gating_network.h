// src/semantic-aqo/model/gating_network.h

#ifndef GATING_NETWORK_H
#define GATING_NETWORK_H

typedef struct {
    double **attention_score;  // [num_context, num_senses] - attention weights
    double **context_value;    // [num_context, embedding_dim] - averaged context vectors
} GateInput;

/**
 * Gating Network for multi-sense word disambiguation
 * 
 * Computes the best sense embedding for a center word given context.
 * 
 * @param gate_input: Input containing attention scores and context values
 * @param center_sense_embeddings: [num_senses][embedding_dim] - all sense embeddings for center word
 * @param num_senses: Number of senses (K)
 * @param num_context: Number of context words (T-1)
 * @param embedding_dim: Dimension of embeddings (D)
 * @param output_embedding: Output buffer [embedding_dim] - the selected sense embedding
 * @param output_probs: Optional output buffer [num_senses] - gating probabilities (can be NULL)
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
);

/**
 * Alternative version that returns the maximum gating score
 * 
 * @return: The maximum score among all senses
 */
double gating_network_max_score(
    const GateInput *gate_input,
    const double **center_sense_embeddings,
    int num_senses,
    int num_context,
    int embedding_dim
);
#endif /* GATING_NETWORK_H */