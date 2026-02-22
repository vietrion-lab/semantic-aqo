#!/bin/bash
# Test 3: Verify database tables and counts
# Checks that all data was ingested correctly

set -e

echo "============================================================"
echo "TEST 3: Verify database tables and record counts"
echo "============================================================"

# Connect to PostgreSQL and check table contents
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "
SELECT 
    'sense_vectors_raw' as table_name, 
    COUNT(*) as record_count 
FROM sense_vectors_raw

UNION ALL

SELECT 
    'pq_codebook' as table_name, 
    COUNT(*) as record_count 
FROM pq_codebook

UNION ALL

SELECT 
    'pq_quantization' as table_name, 
    COUNT(*) as record_count 
FROM pq_quantization
ORDER BY table_name;
"

echo ""
echo "--- Sample raw vectors ---"
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "
SELECT id, word, sense_id, octet_length(vector) as vector_bytes 
FROM sense_vectors_raw 
LIMIT 5;
"

echo ""
echo "--- Sample PQ codes ---"
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "
SELECT id, array_length(code_vector, 1) as code_length 
FROM pq_quantization 
LIMIT 5;
"

echo ""
echo "--- Codebook statistics ---"
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "
SELECT 
    subspace_id, 
    COUNT(*) as num_centroids,
    MIN(centroid_id) as min_centroid_id,
    MAX(centroid_id) as max_centroid_id
FROM pq_codebook 
GROUP BY subspace_id 
ORDER BY subspace_id;
"

echo ""
echo "✓ Database verification completed successfully"
