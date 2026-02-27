#include "model_loader.h"
#include "../utils/alloc_compat.h"
#include "../utils/context_extractor.h"
#include <stdio.h>

Vocab* load_vocab_bin(const char *file_path) {
    FILE *f = fopen(file_path, "rb");
    if (!f) return NULL;

    int32_t num_records;
    if (fread(&num_records, sizeof(int32_t), 1, f) != 1) {
        fclose(f); return NULL;
    }

    Vocab *v = create_vocab();
    for (int i = 0; i < num_records; i++) {
        int32_t word_len;
        fread(&word_len, sizeof(int32_t), 1, f);
        
        char *word = (char *)palloc(word_len + 1);
        fread(word, 1, word_len, f);
        word[word_len] = '\0';
        
        int32_t word_id;
        fread(&word_id, sizeof(int32_t), 1, f);
        
        // Assume vocab structure handles insertion properly
        if (v->count >= v->capacity) {
            v->capacity *= 2;
            v->tokens = (char**)repalloc(v->tokens, v->capacity * sizeof(char*));
            v->ids = (int*)repalloc(v->ids, v->capacity * sizeof(int));
        }
        v->tokens[v->count] = word;
        v->ids[v->count] = word_id;
        v->count++;
    }
    
    fclose(f);
    return v;
}

EmbeddingModel* load_embeddings_bin(const char *file_path) {
    FILE *f = fopen(file_path, "rb");
    if (!f) return NULL;

    EmbeddingModel *model = (EmbeddingModel *)palloc(sizeof(EmbeddingModel));
    
    fread(&model->count, sizeof(int32_t), 1, f);
    fread(&model->num_senses, sizeof(int32_t), 1, f);
    fread(&model->dim, sizeof(int32_t), 1, f);

    model->embeddings = (SenseEmbedding *)palloc(model->count * sizeof(SenseEmbedding));

    for (size_t i = 0; i < model->count; i++) {
        int32_t word_len;
        fread(&word_len, sizeof(int32_t), 1, f);
        fseek(f, word_len, SEEK_CUR); // Skip word string, we only need IDs
        
        fread(&model->embeddings[i].word_id, sizeof(int32_t), 1, f);
        fread(&model->embeddings[i].sense_id, sizeof(int32_t), 1, f);
        
        model->embeddings[i].embedding = (float *)palloc(model->dim * sizeof(float));
        fread(model->embeddings[i].embedding, sizeof(float), model->dim, f);
    }

    fclose(f);
    return model;
}

void free_embedding_model(EmbeddingModel *model) {
    if (!model) return;
    for (size_t i = 0; i < model->count; i++) {
        pfree(model->embeddings[i].embedding);
    }
    pfree(model->embeddings);
    pfree(model);
}