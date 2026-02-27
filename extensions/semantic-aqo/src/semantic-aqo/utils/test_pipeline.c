#include "pg_compat.h"
#include "../model/w2v_embedding_extractor.h"
#include "../model/w2v_encoder.h"
#include <stdio.h>

int main() {
    printf("--- PIPELINE FINAL INTEGRATION TEST ---\n");

    const char *v_path = "../../../../../multisense-word2vec/output/model/vocab.bin";
    const char *e_path = "../../../../../multisense-word2vec/output/model/sense_embeddings.bin";

    if (!init_embedding_extractor(v_path, e_path, 3, 150)) {
        printf("Error loading model files.\n");
        return 1;
    }

    const char *query =
        "SELECT u.name, SUM(o.amount) AS total "
        "FROM users u JOIN orders o ON o.user_id = u.id "
        "WHERE u.age >= 21 AND o.status IN ('paid','shipped') "
        "GROUP BY u.name ORDER BY total DESC";

    EncodedQuery *eq = encode_sql_query(query, 7);

    if (eq) {
        /* FAIR AVERAGE: Compute the final Query Vector via Extractor */
        float *final_query_vector = extractor_compute_fair_average(eq->aggregate_vector, eq->num_words, eq->word_dim);

        FILE *f = fopen("out.txt", "w");
        fprintf(f, "SQL: %s\nWords: %d\n\n[QUERY VECTOR (FAIR AVERAGE)]:\n", query, eq->num_words);
        for (int i = 0; i < eq->word_dim; i++) fprintf(f, "%f ", final_query_vector[i]);

        fprintf(f, "\n\n[AGGREGATE EMBEDDINGS (CONCAT)]:\n");
        for (int i = 0; i < eq->num_words * eq->word_dim; i++) {
            fprintf(f, "%f ", eq->aggregate_vector[i]);
            if ((i + 1) % 150 == 0) fprintf(f, "\n");
        }

        fclose(f);
        pfree(final_query_vector);
        free_encoded_query(eq);
        printf("Success! Pipeline processed query. See out.txt\n");
    }

    free_embedding_extractor();
    return 0;
}