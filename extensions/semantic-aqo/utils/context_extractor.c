#include "context_extractor.h"

#include "postgres.h"
#include "utils/palloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define INITIAL_CAPACITY 32


Vocab* create_vocab(void) {
    Vocab *v = (Vocab*) palloc(sizeof(Vocab));
    // Vocab *v = (Vocab*) malloc(sizeof(Vocab));
    if (!v) return NULL;

    v->tokens = (char**) palloc(INITIAL_CAPACITY * sizeof(char*));
    v->ids = (int*) palloc(INITIAL_CAPACITY * sizeof(int));
    // v->tokens = (char**) malloc(INITIAL_CAPACITY * sizeof(char*));
    // v->ids = (int*) malloc(INITIAL_CAPACITY * sizeof(int));

    if (!v->tokens || !v->ids) return NULL;

    v->count = 0;
    v->capacity = INITIAL_CAPACITY;
    return v;
}


static int find_token(Vocab *v, const char *token) {
    for (size_t i = 0; i < v->count; i++) {
        if (strcmp(v->tokens[i], token) == 0)
            return v->ids[i];
    }
    return -1;
}


int get_or_add_token_id(Vocab *v, const char *token) {
    int existing = find_token(v, token);
    if (existing != -1) return existing;

    if (v->count >= v->capacity) {
        size_t new_cap = v->capacity * 2;

        v->tokens = repalloc(v->tokens, sizeof(char*) * new_cap);
        v->ids = repalloc(v->ids, sizeof(int) * new_cap);

        // v->tokens = realloc(v->tokens, sizeof(char*) * new_cap);
        // v->ids = realloc(v->ids, sizeof(int) * new_cap);

        v->capacity = new_cap;
    }

    int new_id = (int)v->count + 1;

    v->tokens[v->count] = pstrdup(token);
    // v->tokens[v->count] = strdup(token);

    v->ids[v->count] = new_id;
    v->count++;

    return new_id;
}


void free_vocab(Vocab *v) {
    if (!v) return;

    for (size_t i = 0; i < v->count; i++) {
        pfree(v->tokens[i]);
        // free(v->tokens[i]);
    }

    pfree(v->tokens);
    pfree(v->ids);
    pfree(v);

    // free(v->tokens);
    // free(v->ids);
    // free(v);
}


IdSequence* tokens_to_ids(char **tokens, size_t token_count, Vocab *vocab) {
    IdSequence *seq = (IdSequence*) palloc(sizeof(IdSequence)); */
    // IdSequence *seq = (IdSequence*) malloc(sizeof(IdSequence));
    if (!seq) return NULL;

    seq->data = (int*) palloc(sizeof(int) * token_count);
    // seq->data = (int*) malloc(sizeof(int) * token_count);

    seq->count = token_count;

    for (size_t i = 0; i < token_count; i++) {
        seq->data[i] = get_or_add_token_id(vocab, tokens[i]);
    }

    return seq;
}


void free_id_sequence(IdSequence *seq) {
    if (!seq) return;

    pfree(seq->data);
    pfree(seq);

    // free(seq->data);
    // free(seq);
}


TrainingPairArray* extract_training_pairs(IdSequence *seq, int window) {
    if (!seq) return NULL;

    TrainingPairArray *arr = palloc(sizeof(TrainingPairArray));
    // TrainingPairArray *arr = malloc(sizeof(TrainingPairArray));

    arr->pairs = palloc(sizeof(TrainingPair) * seq->count);
    // arr->pairs = malloc(sizeof(TrainingPair) * seq->count);

    arr->count = 0;

    for (size_t i = 0; i < seq->count; i++) {
        int center = seq->data[i];
        int max_ctx = window * 2;

        int *contexts = palloc(sizeof(int) * max_ctx);
        // int *contexts = malloc(sizeof(int) * max_ctx);

        size_t ctx_count = 0;

        int start = (int)i - window;
        int end = (int)i + window;

        if (start < 0) start = 0;
        if (end >= seq->count) end = seq->count - 1;

        for (int j = start; j <= end; j++) {
            if (j == i) continue;
            contexts[ctx_count++] = seq->data[j];
        }

        arr->pairs[arr->count].center = center;
        arr->pairs[arr->count].contexts = contexts;
        arr->pairs[arr->count].context_count = ctx_count;
        arr->count++;
    }

    return arr;
}


void free_training_pairs(TrainingPairArray *arr) {
    if (!arr) return;

    for (size_t i = 0; i < arr->count; i++) {
        pfree(arr->pairs[i].contexts);
        // free(arr->pairs[i].contexts);
    }

    pfree(arr->pairs);
    pfree(arr);

    // free(arr->pairs);
    // free(arr);
}


void print_training_pairs(TrainingPairArray *arr) {
    if (!arr) return;

    for (size_t i = 0; i < arr->count; i++) {
        printf("Center: %d | Context: ", arr->pairs[i].center);
        for (size_t j = 0; j < arr->pairs[i].context_count; j++) {
            printf("%d ", arr->pairs[i].contexts[j]);
        }
        printf("\n");
    }
}
