# Usage Guide — Sense Embeddings Search System

## Overview

A search system for sense embeddings using **Product Quantization (PQ)** with
**post-verification**, backed by PostgreSQL. Vectors are stored as `bytea`,
PQ codes enable fast approximate search, and exact L2 re-ranking ensures
accurate final results.

### Architecture
```
sense_embeddings.bin
        │
        ▼
┌─────────────────┐     ┌──────────────┐
│  PQ Training    │────▶│  codebook.pkl│
│  (FAISS)        │     └──────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        PostgreSQL  (sense_db)       │
│                                     │
│  sense_vectors_raw   141 rows       │ ← full-precision bytea vectors
│  pq_codebook        2560 rows       │ ← 10 subspaces × 256 centroids
│  pq_quantization     141 rows       │ ← 10-element INT[] per vector
└────────┬────────────────────────────┘
         │
    ┌────▼─────────────┐
    │ PQ Search        │  Phase 1: approximate top-50
    └────┬─────────────┘
    ┌────▼─────────────┐
    │ Post-Verify      │  Phase 2: exact re-rank → top-5
    └────┬─────────────┘
         ▼
    [Final Results]
```

---

## Quick Start

```bash
cd /workspaces/app/ingestion
source venv/bin/activate

# 1. Ingest (auto-creates database, UDFs, tables, indexes)
python main.py ingest

# 2. Search
python main.py search SELECT
python main.py search ELSE
```

### What `ingest` does automatically

1. Creates database `sense_db` if it does not exist
2. Creates UDFs `vec_to_bytea` / `bytea_to_vec`
3. Creates tables `sense_vectors_raw`, `pq_codebook`, `pq_quantization`
4. Reads `sense_embeddings.bin` (141 vectors, dim=150)
5. Trains PQ with FAISS (augments training set if < 256 vectors)
6. Imports raw vectors, codebook, and PQ codes
7. Creates B-tree indexes

---

## Vocabulary

The current dataset has **141 vectors** across **47 unique words** (SQL tokens):

```
AND  AS  BETWEEN  BY  CASE  CAST  COUNT  DBO  DISTINCT  ELSE  END
FGETNEARBYOBJEQ  FGETURLFITSCFRAME  FPHOTOTYPEN
FROM  IN  JOIN  LIKE  NOT  ON  OR  ORDER  SELECT  THEN  TOP  WHEN  WHERE
<ALIAS_T1>  <ALIAS_T2>  <ALIAS_T3>  <ALIAS_T4>
<COL>  <COL_OUT>  <NUM>  <STR>  <TAB>
(  )  *  +  ,  -  .  /  <  =  >
```

---

## Search

```bash
python main.py search <WORD>
```

**Example:**
```
$ python main.py search ELSE

Top-5 nearest senses for 'ELSE':
============================================================
    1. word=ELSE                 sense_id=0     dist=0.000000
    2. word=ELSE                 sense_id=2     dist=0.001552
    3. word=FPHOTOTYPEN          sense_id=0     dist=0.003346
    4. word=>                    sense_id=0     dist=0.003421
    5. word=DBO                  sense_id=1     dist=0.003421
```

**Fields:**
- `word` — token from the vocabulary
- `sense_id` — sense index (0, 1, 2… for polysemous words)
- `dist` — squared L2 distance (lower = more similar)

---

## Testing

```bash
chmod +x tests/*.sh

# Run all tests
bash tests/run-all-tests.sh

# Or individually
bash tests/01-search-word.sh          # single word search
bash tests/02-search-multiple-words.sh # multi-word search
bash tests/03-verify-database.sh       # DB integrity check
bash tests/04-performance-benchmark.sh # latency benchmark
bash tests/05-edge-cases.sh            # edge cases
```

See [tests/README.md](tests/README.md) for details.

---

## Configuration

### config/db_config.json
```json
{
  "username": "postgres",
  "password": "postgres",
  "host": "localhost",
  "port": 5432,
  "db_name": "sense_db",
  "batch_size": 10000
}
```

### config/pq_config.json
```json
{
  "m_subspaces": 10,
  "nbits": 8,
  "train_size": 100000,
  "post_verification_k": 50,
  "final_k": 5,
  "use_bytea": true,
  "normalization": false
}
```

| Key | Description |
|-----|-------------|
| `m_subspaces` | Number of PQ subspaces. `dim` must be divisible by this. Divisors of 150: 1,2,3,5,6,10,15,25,30,50,75,150 |
| `nbits` | Bits per sub-quantizer → $2^{nbits}$ centroids per subspace |
| `post_verification_k` | Candidates from PQ phase (phase 1) |
| `final_k` | Final results after exact re-rank (phase 2) |

### Tuning Tips

**Better accuracy** (slower): increase `post_verification_k` and `final_k`

**Better speed** (less accurate): decrease `post_verification_k` and `final_k`

**Changing PQ structure** requires re-ingestion: `m_subspaces`, `nbits`

---

## Database Management

```bash
# Connect
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db
```

### Useful Queries

```sql
-- Record counts
SELECT 'sense_vectors_raw' AS tbl, COUNT(*) FROM sense_vectors_raw
UNION ALL
SELECT 'pq_codebook', COUNT(*) FROM pq_codebook
UNION ALL
SELECT 'pq_quantization', COUNT(*) FROM pq_quantization;

-- List all words
SELECT DISTINCT word FROM sense_vectors_raw ORDER BY word;

-- Codebook structure
SELECT subspace_id, COUNT(*) AS centroids FROM pq_codebook GROUP BY 1 ORDER BY 1;

-- PQ code sample
SELECT id, array_length(code_vector, 1) AS len FROM pq_quantization LIMIT 5;
```

### Backup & Restore

```bash
sudo -u postgres /usr/local/pgsql/bin/pg_dump sense_db > backup.sql
sudo -u postgres /usr/local/pgsql/bin/psql sense_db < backup.sql
```

### Reset

```bash
sudo -u postgres /usr/local/pgsql/bin/psql -c "DROP DATABASE IF EXISTS sense_db;"
python main.py ingest   # re-creates everything
```

---

## How It Works

### Phase 1 — PQ Approximate Search

1. Split query vector $q$ into $m$ sub-vectors: $q_1, q_2, \ldots, q_m$
2. Precompute distance table: $d[j][c] = \|q_j - centroid_{j,c}\|^2$
3. For each stored PQ code $[c_1, c_2, \ldots, c_m]$, compute approximate distance: $\sum_{j=1}^{m} d[j][c_j]$
4. Return top-N candidates sorted by approximate distance

### Phase 2 — Post-Verification

1. Fetch raw vectors (bytea) for the N candidates
2. Compute exact squared L2 distance: $\|q - v\|^2$
3. Re-rank and return top-K

### Why Post-Verification?

PQ gives **approximate** distances that can mis-order results.
Post-verification guarantees **exact** ranking for the final output.
Trade-off: PQ speed + exact search accuracy.

---

## Bytea Storage Notes

Vectors are stored as `bytea` using the `vec_to_bytea` UDF, which calls
PostgreSQL's `float4send` (big-endian / network byte order). When reading
in Python, vectors are decoded with `np.frombuffer(data, dtype='>f4')`
to handle the byte order correctly.

---

## Python API

```python
from config import Configuration
from logger import Logger
from db_schema import create_connection
from pq_trainer import load_codebook
from search import search_pq_with_post_verification
import numpy as np

db_config = Configuration("config/db_config.json")
index_config = Configuration("config/pq_config.json")
logger = Logger("")

con, cur = create_connection(db_config, logger)
codebook = load_codebook("codebook.pkl", logger)

query_vec = np.random.randn(150).astype(np.float32)

results = search_pq_with_post_verification(
    query_vec, codebook, con, cur, index_config, logger,
    top_n=50, final_k=5, use_bytea=True,
)

for r in results:
    print(f"{r['word']}  sense={r['sense_id']}  dist={r['distance']:.6f}")

cur.close()
con.close()
```

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `database "sense_db" does not exist` | First run | Run `python main.py ingest` |
| `function vec_to_bytea does not exist` | UDF missing | Re-run `python main.py ingest` |
| `Dimension 150 must be divisible by m_subspaces` | Bad config | Set `m_subspaces` to a divisor of 150 |
| `Number of training points < clusters` | Small dataset | Pipeline auto-augments; no action needed |
| Word not found | Not in vocabulary | Check vocabulary list above |
| Search returns no results | DB empty | Run ingestion first |

### Debug Commands

```bash
# Check PostgreSQL status
sudo -u postgres /usr/local/pgsql/bin/pg_ctl status -D /usr/local/pgsql/data

# Check data exists
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "SELECT COUNT(*) FROM sense_vectors_raw;"

# Check if a word exists
sudo -u postgres /usr/local/pgsql/bin/psql -d sense_db -c "SELECT * FROM sense_vectors_raw WHERE word = 'SELECT';"
```

---

## Benchmark

With 141 vectors, dim=150, m=10, nbits=8:

| Metric | Value |
|--------|-------|
| Ingest time | ~2s |
| Search latency | ~900ms/query |
| PQ candidates | 50 |
| Final results | 5 |

---

## Deployment Checklist

- [ ] PostgreSQL is running
- [ ] `sense_embeddings.bin` exists in `models/`
- [ ] Config files are set correctly
- [ ] `python main.py ingest` completed successfully
- [ ] `bash tests/run-all-tests.sh` passes
- [ ] Search tested with multiple words

---

## References

- Jégou et al., "Product Quantization for Nearest Neighbor Search" (2011)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PostgreSQL bytea](https://www.postgresql.org/docs/current/datatype-binary.html)
