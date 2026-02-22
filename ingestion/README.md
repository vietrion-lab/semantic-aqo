# Sense Embeddings Ingestion Pipeline

**PQ (Product Quantization) + Post-verification** pipeline for loading custom
`sense_embeddings.bin` into PostgreSQL, following the FREDDY architecture.

## Architecture

```
┌─────────────────────────┐
│  sense_embeddings.bin   │   Binary file with (word, sense_id, embedding)
└────────────┬────────────┘
             │
     ┌───────▼────────┐
     │  Binary Reader  │   sense_embeddings_reader.py
     └───────┬────────┘
             │
    ┌────────▼─────────┐
    │   PQ Trainer     │   pq_trainer.py (FAISS ProductQuantizer)
    │  • Train PQ      │
    │  • Extract CB    │
    │  • Encode vecs   │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   DB Importer    │   db_importer.py + db_schema.py
    │  • Raw vectors   │   → sense_vectors_raw
    │  • Codebook      │   → pq_codebook
    │  • PQ codes      │   → pq_quantization
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   Search Engine  │   search.py
    │  Phase 1: PQ     │   Approximate top-50
    │  Phase 2: PV     │   Exact re-rank → top-5
    └──────────────────┘
```

## Database Schema

```sql
-- 1. Raw vectors (for post-verification)
sense_vectors_raw (
    id SERIAL PRIMARY KEY,
    word VARCHAR(200),
    sense_id INT,
    vector FLOAT4[] | bytea
)

-- 2. PQ Codebook (centroids per subspace)
pq_codebook (
    id SERIAL PRIMARY KEY,
    subspace_id INT,
    centroid_id INT,
    vector FLOAT4[] | bytea
)

-- 3. PQ Quantization codes
pq_quantization (
    id INT REFERENCES sense_vectors_raw(id),
    code_vector INT[]
)
```

## Files

| File                         | Purpose                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `main.py`                    | CLI entry point (`ingest` / `search`)                      |
| `config.py`                  | JSON configuration loader                                  |
| `logger.py`                  | Logging utility                                            |
| `sense_embeddings_reader.py` | Binary parser for `vocab.bin` and `sense_embeddings.bin`   |
| `pq_trainer.py`              | FAISS Product Quantization training, encoding, persistence |
| `db_schema.py`               | PostgreSQL schema creation, indexes, triggers              |
| `db_importer.py`             | Batch insert raw vectors, codebook, PQ codes               |
| `search.py`                  | Two-phase PQ search + post-verification                    |
| `config/db_config.json`      | Database connection settings                               |
| `config/pq_config.json`      | PQ parameters & table names                                |

## Quick Start

```bash
# 1. Setup virtual environment & install dependencies
cd /workspaces/app/ingestion
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the ingestion pipeline (auto-creates database & UDFs)
python main.py ingest

# 3. Search for similar senses
python main.py search SELECT
python main.py search ELSE
```

## Current Dataset

- **141 vectors**, dimension **150**, across **47 unique words**
- Vocabulary consists of SQL tokens and special identifiers:
  `AND`, `AS`, `BETWEEN`, `BY`, `CASE`, `CAST`, `COUNT`, `DBO`, `DISTINCT`,
  `ELSE`, `END`, `FROM`, `IN`, `JOIN`, `LIKE`, `NOT`, `ON`, `OR`, `ORDER`,
  `SELECT`, `THEN`, `TOP`, `WHEN`, `WHERE`, symbols (`*`, `+`, `<`, `>`, etc.),
  and placeholders (`<COL>`, `<TAB>`, `<NUM>`, `<ALIAS_T1>`, etc.)

## Configuration

### `config/pq_config.json`

| Key                   | Description                                    | Current |
| --------------------- | ---------------------------------------------- | ------- |
| `m_subspaces`         | Number of PQ subspaces (dim must be divisible) | 10      |
| `nbits`               | Bits per sub-quantizer (2^nbits centroids)     | 8       |
| `train_size`          | Max vectors used for PQ training               | 100000  |
| `post_verification_k` | Candidates from PQ phase                       | 50      |
| `final_k`             | Final results after exact re-rank              | 5       |
| `use_bytea`           | Store vectors as bytea (FREDDY-style)          | true    |

> **Note:** If the training set is smaller than 2^nbits, the pipeline
> automatically augments it with noisy duplicates so FAISS can train.

## Key Differences from FREDDY

| Aspect      | FREDDY                        | This Pipeline                      |
| ----------- | ----------------------------- | ---------------------------------- |
| Data format | Word2Vec text (word + vector) | Binary (word + sense_id + vector)  |
| Primary key | word (unique)                 | (word, sense_id) or SERIAL id      |
| Raw storage | Optional                      | **Required** for post-verification |
| Search      | PQ only or IVF-ADC            | PQ + mandatory post-verification   |
| DB setup    | Manual                        | Auto-creates DB, UDFs, and tables  |
| Small data  | Requires n ≥ centroids        | Auto-augments training set         |
