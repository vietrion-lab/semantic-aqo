#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <stdint.h>

typedef struct {
    int word_id;
    int sense_id;
    float *embedding;
} SenseEmbedding;

typedef struct {
    SenseEmbedding *embeddings;
    size_t count;
    int num_senses;
    int dim;
} EmbeddingModel;

Vocab* load_vocab_bin(const char *file_path);
void free_vocab(Vocab *v);

EmbeddingModel* load_embeddings_bin(const char *file_path);
void free_embedding_model(EmbeddingModel *model);

#endif // MODEL_LOADER_H