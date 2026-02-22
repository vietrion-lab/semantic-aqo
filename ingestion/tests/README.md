# Test Scripts for Sense Embeddings Search System

Test scripts to verify the PQ-based sense embeddings search system after ingestion.

## Prerequisites

1. ✅ PostgreSQL is running
2. ✅ Data ingested: `cd /workspaces/app/ingestion && source venv/bin/activate && python main.py ingest`

## Vocabulary

The current dataset contains **141 vectors** across **47 unique words** (SQL tokens and special identifiers):

```
(  )  *  +  ,  -  .  /  <  =  >
AND  AS  BETWEEN  BY  CASE  CAST  COUNT  DBO  DISTINCT  ELSE  END
FGETNEARBYOBJEQ  FGETURLFITSCFRAME  FPHOTOTYPEN  FROM  IN  JOIN  LIKE
NOT  ON  OR  ORDER  SELECT  THEN  TOP  WHEN  WHERE
<ALIAS_T1>  <ALIAS_T2>  <ALIAS_T3>  <ALIAS_T4>
<COL>  <COL_OUT>  <NUM>  <STR>  <TAB>
```

## Tests

| Script | Purpose | Description |
|--------|---------|-------------|
| `01-search-word.sh` | Single word search | Searches for `ELSE`, expects exact match (dist=0) |
| `02-search-multiple-words.sh` | Multi-word search | Searches `SELECT`, `FROM`, `WHERE`, `JOIN`, `DISTINCT` |
| `03-verify-database.sh` | DB integrity check | Verifies record counts and table structure |
| `04-performance-benchmark.sh` | Performance | Benchmarks 10 queries, reports avg latency in ms |
| `05-edge-cases.sh` | Edge cases | Tests non-existent word, symbol `*`, valid word `AND` |
| `run-all-tests.sh` | Run all | Executes all tests and reports pass/fail summary |

## Running

```bash
cd /workspaces/app/ingestion
chmod +x tests/*.sh

# Individual
bash tests/01-search-word.sh

# All tests
bash tests/run-all-tests.sh
```

## Expected Output

### 03-verify-database.sh
```
sense_vectors_raw:   141 records
pq_codebook:        2560 records (10 subspaces × 256 centroids)
pq_quantization:     141 records
code_length:         10 (one byte per subspace)
```

### Search results format
```
Top-5 nearest senses for 'ELSE':
============================================================
    1. word=ELSE                 sense_id=0     dist=0.000000
    2. word=ELSE                 sense_id=2     dist=0.001552
    3. word=FPHOTOTYPEN          sense_id=0     dist=0.003346
    ...
```

- **word**: Token from vocabulary
- **sense_id**: Sense index (0, 1, 2... for polysemous words)
- **dist**: Squared L2 distance (lower = more similar)

## Configuration

Search parameters in `config/pq_config.json`:
```json
{
  "m_subspaces": 10,
  "nbits": 8,
  "post_verification_k": 50,
  "final_k": 5
}
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `database "sense_db" does not exist` | Run `python main.py ingest` (auto-creates DB) |
| `function vec_to_bytea does not exist` | Re-run `python main.py ingest` (auto-creates UDFs) |
| `Word not found in database` | Check word exists in vocabulary list above |
| Slow search | Reduce `post_verification_k` in config |
