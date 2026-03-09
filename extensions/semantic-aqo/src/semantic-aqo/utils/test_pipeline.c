#include "pg_compat.h"
#include "../model/w2v_embedding_extractor.h"
#include "../model/w2v_encoder.h"
#include "../utils/sql_preprocessor.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("--- SEMANTIC AQO: FINAL INTEGRATION TEST ---\n");

    const char *v_path = "../../../../../multisense-word2vec/output/model/vocab.bin";
    const char *e_path = "../../../../../multisense-word2vec/output/model/sense_embeddings.bin";

    // Khởi tạo khớp với header mới (4 tham số)
    if (!init_embedding_extractor(v_path, e_path, 3, 150)) {
        printf("Error loading model files.\n");
        return 1;
    }

    const char *query = "SELECT u.age FROM users u WHERE u.age >= 21";
    float sigma = 3.0f;

    // Lấy token để in log chi tiết
    SQLPreprocessingResult *prep = preprocess_sql_query(query);
    EncodedQuery *eq = encode_sql_query(query, sigma);

    if (eq && prep) {
        int D = eq->word_dim;
        int N = prep->tokens->count;
        float cw = (N - 1) / 2.0f;

        FILE *f = fopen("out.txt", "w");
        fprintf(f, "SQL: %s\nSIGMA: %.2f | CENTER: %.2f\n\n", query, sigma, cw);

        fprintf(f, "[PER-TOKEN ANALYSIS]\n");
        for (int j = 0; j < N; j++) {
            int wid = extractor_get_word_id(prep->tokens->tokens[j]);
            const float *emb = (wid >= 0) ? extractor_get_word_embedding(wid) : NULL;
            float diff = cw - (float)j;
            float w_j = expf(-(diff * diff) / (2.0f * sigma * sigma));
            fprintf(f, "Token %d: %-10s | Weight: %.4f\n",
        	 j, prep->tokens->tokens[j], w_j);
        }

        fprintf(f, "\n[FINAL QUERY VECTOR]:\n");
        for (int i = 0; i < D; i++) fprintf(f, "%.4f ", eq->aggregate_vector[i]);

        fclose(f);
        printf("Success! Trace written to out.txt\n");
    } else {
        printf("Encoding failed or valid tokens not found.\n");
    }

    if (eq) free_encoded_query(eq);
    if (prep) free_sql_preprocessing_result(prep);
    free_embedding_extractor();
    return 0;
}