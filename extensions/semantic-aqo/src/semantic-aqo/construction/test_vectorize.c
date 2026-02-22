/*
 * test_vectorize.c  –  smoke-test for vectorize_sql_queries()
 *
 * Build & run:
 *   make test
 */

#include "vectorize_sql.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static int failures = 0;

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("[" FAIL "] %s\n", msg); \
            failures++; \
        } else { \
            printf("[" PASS "] %s\n", msg); \
        } \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* L2 norm of a float vector */
static float l2(const float *v, int d)
{
    float s = 0;
    for (int i = 0; i < d; i++) s += v[i] * v[i];
    return sqrtf(s);
}

/* cosine similarity between two float vectors */
static float cosine(const float *a, const float *b, int d)
{
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < d; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-9f) ? dot / denom : 0.0f;
}

/* ------------------------------------------------------------------ */
/*  Test cases                                                         */
/* ------------------------------------------------------------------ */

static void test_model_loading(void)
{
    printf("\n=== Model loading ===\n");
    ASSERT(!is_model_loaded(), "model not loaded before load_sensate_model_default()");

    bool ok = load_sensate_model_default();
    ASSERT(ok, "load_sensate_model_default() succeeds");
    ASSERT(is_model_loaded(), "is_model_loaded() returns true after load");
}

static void test_single_query(void)
{
    printf("\n=== Single query ===\n");

    const char *q = "SELECT id FROM users WHERE age > 18";
    const char *queries[1] = { q };

    SQLVector *vecs = vectorize_sql_queries(queries, 1);
    ASSERT(vecs != NULL, "vectorize_sql_queries() returns non-NULL");
    ASSERT(vecs[0].valid, "first result is valid");
    ASSERT(vecs[0].dim > 0, "embedding dimension > 0");

    if (vecs && vecs[0].valid) {
        float norm = l2(vecs[0].data, vecs[0].dim);
        printf("      query : \"%s\"\n", q);
        printf("      dim   : %d\n", vecs[0].dim);
        printf("      L2    : %.6f\n", norm);
        ASSERT(norm > 0.0f, "embedding L2 norm > 0 (non-zero vector)");
        ASSERT(isfinite(norm), "embedding L2 norm is finite");

        /* Print first 8 dims */
        printf("      vec[0..7]:");
        for (int i = 0; i < 8 && i < vecs[0].dim; i++)
            printf(" %.4f", vecs[0].data[i]);
        printf("\n");
    }

    free_sql_vectors(vecs, 1);
}

static void test_batch_queries(void)
{
    printf("\n=== Batch queries ===\n");

    const char *queries[] = {
        "SELECT name, age FROM employees",
        "SELECT e.name, d.dept FROM employees e JOIN departments d ON d.id = e.dept_id",
        "SELECT COUNT(*) FROM orders WHERE status = 'pending'",
        "SELECT AVG(salary) AS avg_sal FROM staff GROUP BY department",
        "SELECT * FROM products WHERE price BETWEEN 10 AND 100 ORDER BY price ASC",
    };
    int N = (int)(sizeof(queries) / sizeof(queries[0]));

    SQLVector *vecs = vectorize_sql_queries(queries, N);
    ASSERT(vecs != NULL, "batch vectorize returns non-NULL");

    int valid_count = 0;
    for (int i = 0; i < N; i++) {
        if (vecs[i].valid) valid_count++;
        printf("  [%d] valid=%-5s  L2=%.4f  \"%s\"\n",
               i, vecs[i].valid ? "true" : "false",
               vecs[i].valid ? l2(vecs[i].data, vecs[i].dim) : 0.0f,
               queries[i]);
    }
    ASSERT(valid_count == N, "all batch queries produce valid vectors");

    free_sql_vectors(vecs, N);
}

static void test_similar_queries_are_closer(void)
{
    printf("\n=== Similarity ordering ===\n");
    /*
     * Two semantically similar queries should have higher cosine similarity
     * with each other than with a structurally different one.
     */
    const char *q_a  = "SELECT name FROM customers WHERE city = 'Hanoi'";
    const char *q_b  = "SELECT name FROM clients WHERE city = 'HCMC'";   /* similar */
    const char *q_c  = "SELECT SUM(amount) FROM orders GROUP BY month";  /* different */

    const char *qs[] = { q_a, q_b, q_c };
    SQLVector  *vecs = vectorize_sql_queries(qs, 3);

    if (!vecs || !vecs[0].valid || !vecs[1].valid || !vecs[2].valid) {
        printf("  (skipped – one or more queries did not vectorize)\n");
        free_sql_vectors(vecs, 3);
        return;
    }

    int D    = vecs[0].dim;
    float ab = cosine(vecs[0].data, vecs[1].data, D);
    float ac = cosine(vecs[0].data, vecs[2].data, D);

    printf("  cosine(a,b)=%.4f  cosine(a,c)=%.4f\n", ab, ac);
    ASSERT(ab > ac, "similar queries (a,b) are closer than dissimilar (a,c)");

    free_sql_vectors(vecs, 3);
}

static void test_null_and_empty(void)
{
    printf("\n=== Edge cases ===\n");

    /* NULL pointer in array */
    const char *q_null[] = { NULL };
    SQLVector  *v1 = vectorize_sql_queries(q_null, 1);
    ASSERT(v1 != NULL, "NULL query in array → returns array (not crash)");
    ASSERT(!v1[0].valid, "NULL query → valid == false");
    free_sql_vectors(v1, 1);

    /* Empty string */
    const char *q_empty[] = { "" };
    SQLVector  *v2 = vectorize_sql_queries(q_empty, 1);
    ASSERT(v2 != NULL, "empty-string query → returns array");
    ASSERT(!v2[0].valid, "empty-string query → valid == false");
    free_sql_vectors(v2, 1);

    /* Zero count */
    SQLVector  *v3 = vectorize_sql_queries(q_null, 0);
    ASSERT(v3 == NULL, "n_queries=0 → NULL return");
}

static void test_unload_reload(void)
{
    printf("\n=== Unload / reload ===\n");
    unload_sensate_model();
    ASSERT(!is_model_loaded(), "is_model_loaded() false after unload");

    bool ok = load_sensate_model_default();
    ASSERT(ok, "reload after unload succeeds");
    ASSERT(is_model_loaded(), "is_model_loaded() true after reload");
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("========================================\n");
    printf("  vectorize_sql  –  unit tests\n");
    printf("========================================\n");

    test_model_loading();
    test_single_query();
    test_batch_queries();
    test_similar_queries_are_closer();
    test_null_and_empty();
    test_unload_reload();

    unload_sensate_model();

    printf("\n========================================\n");
    if (failures == 0)
        printf("  All tests passed.\n");
    else
        printf("  %d test(s) FAILED.\n", failures);
    printf("========================================\n");

    return failures == 0 ? 0 : 1;
}
