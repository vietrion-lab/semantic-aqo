#!/usr/bin/env python3
"""
Main entry point for the sense-embeddings ingestion pipeline.

Usage:
    python main.py ingest                 # full ETL pipeline
    python main.py search <word>          # demo PQ + post-verification search
    python main.py search-vec <vec_file>  # search with a raw float32 vector file

The pipeline:
  1. Read sense_embeddings.bin  (binary → numpy)
  2. Create/reset DB tables
  3. Import raw vectors into  sense_vectors_raw
  4. Train PQ (faiss)          → codebook
  5. Encode all vectors        → PQ codes
  6. Import codebook + codes into  pq_codebook / pq_quantization
  7. Build indexes
"""

import os
import sys
import time

import numpy as np

from config import Configuration
from logger import Logger
from sense_embeddings_reader import read_sense_embeddings, read_vocab
from db_schema import (
    create_connection,
    init_tables,
    create_all_indexes,
    disable_triggers,
    enable_triggers,
)
from db_importer import import_raw_vectors, import_codebook, import_pq_codes
from pq_trainer import (
    train_pq,
    extract_codebook,
    encode_vectors,
    save_codebook,
    load_codebook,
)
from search import search_pq_with_post_verification


# ---------------------------------------------------------------------------
# Config paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "db_config.json")
PQ_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "pq_config.json")


# ===================================================================
# INGEST
# ===================================================================

def run_ingest():
    """Full ETL: read binary → train PQ → push everything to PostgreSQL."""

    # ---- Configuration ----------------------------------------------------
    db_config = Configuration(DB_CONFIG_PATH)
    index_config = Configuration(PQ_CONFIG_PATH)
    logger = Logger(db_config.get_value("log"))

    batch_size = db_config.get_value("batch_size")
    use_bytea = index_config.get("use_bytea", True)
    m_subspaces = index_config.get_value("m_subspaces")
    nbits = index_config.get_value("nbits")
    train_size = index_config.get("train_size", None)
    normalization = index_config.get("normalization", False)

    embeddings_path = index_config.get_value("sense_embeddings_path")
    # Resolve relative paths from script dir
    if not os.path.isabs(embeddings_path):
        embeddings_path = os.path.normpath(os.path.join(SCRIPT_DIR, embeddings_path))

    # ---- Step 1: Read binary data ----------------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 1 — Reading sense_embeddings.bin")
    logger.log(Logger.INFO, "=" * 60)

    t0 = time.time()
    vectors, metadata, dim = read_sense_embeddings(
        embeddings_path, logger, normalization=normalization,
    )
    logger.log(
        Logger.INFO,
        f"Read {len(metadata)} vectors (dim={dim}) in {time.time() - t0:.1f}s",
    )

    # ---- Step 2: DB connection & schema -----------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 2 — Initializing database schema")
    logger.log(Logger.INFO, "=" * 60)

    con, cur = create_connection(db_config, logger)
    init_tables(con, cur, index_config, logger, use_bytea=use_bytea)

    raw_table = index_config.get_value("raw_table_name")
    disable_triggers(raw_table, con, cur)

    # ---- Step 3: Import raw vectors --------------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 3 — Importing raw vectors")
    logger.log(Logger.INFO, "=" * 60)

    t0 = time.time()
    db_ids = import_raw_vectors(
        vectors, metadata, con, cur, index_config, batch_size, logger,
        use_bytea=use_bytea,
    )
    logger.log(Logger.INFO, f"Raw import done in {time.time() - t0:.1f}s")

    enable_triggers(raw_table, con, cur)

    # ---- Step 4: Train PQ ------------------------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 4 — Training Product Quantizer")
    logger.log(Logger.INFO, "=" * 60)

    codebook_file = index_config.get("codebook_file", "")
    codebook = None

    if codebook_file and os.path.isfile(codebook_file):
        codebook = load_codebook(codebook_file, logger)
    else:
        t0 = time.time()
        pq = train_pq(vectors, dim, m_subspaces, nbits, logger, train_size)
        codebook = extract_codebook(pq, m_subspaces, nbits)
        logger.log(Logger.INFO, f"PQ training done in {time.time() - t0:.1f}s")

        export_cb = index_config.get("export_codebook", "")
        if export_cb:
            save_codebook(codebook, export_cb, logger)

    # ---- Step 5: Encode vectors ------------------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 5 — Encoding vectors to PQ codes")
    logger.log(Logger.INFO, "=" * 60)

    t0 = time.time()
    # Re-create PQ from codebook for encoding if we loaded from file
    if codebook_file and os.path.isfile(codebook_file):
        import faiss
        pq = faiss.ProductQuantizer(dim, m_subspaces, nbits)
        # Assign centroids back
        faiss.copy_array_to_vector(codebook.ravel(), pq.centroids)
        pq.is_trained = True

    codes = encode_vectors(pq, vectors)
    logger.log(
        Logger.INFO,
        f"Encoded {len(codes)} vectors in {time.time() - t0:.1f}s  "
        f"(codes shape: {codes.shape})",
    )

    # ---- Step 6: Import codebook + codes ---------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 6 — Importing codebook & PQ codes to DB")
    logger.log(Logger.INFO, "=" * 60)

    t0 = time.time()
    import_codebook(codebook, con, cur, index_config, logger, use_bytea=use_bytea)
    import_pq_codes(codes, db_ids, con, cur, index_config, batch_size, logger)
    logger.log(Logger.INFO, f"DB import done in {time.time() - t0:.1f}s")

    # ---- Step 7: Indexes -------------------------------------------------
    logger.log(Logger.INFO, "=" * 60)
    logger.log(Logger.INFO, "STEP 7 — Creating database indexes")
    logger.log(Logger.INFO, "=" * 60)

    create_all_indexes(index_config, con, cur, logger)

    cur.close()
    con.close()
    logger.log(Logger.INFO, "✅ Ingestion pipeline complete!")


# ===================================================================
# SEARCH (demo)
# ===================================================================

def run_search(query_word):
    """
    Demo: look up a word in the raw table, take its vector, then run
    PQ + post-verification to find similar senses.
    """
    db_config = Configuration(DB_CONFIG_PATH)
    index_config = Configuration(PQ_CONFIG_PATH)
    logger = Logger(db_config.get_value("log"))

    use_bytea = index_config.get("use_bytea", True)
    top_n = index_config.get("post_verification_k", 500)
    final_k = index_config.get("final_k", 10)

    con, cur = create_connection(db_config, logger)

    # Fetch the query vector from the DB itself (always get bytea/vector raw)
    raw_table = index_config.get_value("raw_table_name")
    cur.execute(
        f"SELECT id, word, sense_id, vector "
        f"FROM {raw_table} WHERE word = %s LIMIT 1",
        (query_word,),
    )

    row = cur.fetchone()
    if row is None:
        logger.log(Logger.WARNING, f"Word '{query_word}' not found in database")
        cur.close()
        con.close()
        return

    _, word, sense_id, raw_vec = row
    # Convert bytea or array to numpy
    # PostgreSQL float4send stores in big-endian (network byte order)
    if isinstance(raw_vec, (bytes, memoryview)):
        query_vec = np.frombuffer(raw_vec, dtype='>f4').astype(np.float32)
    elif isinstance(raw_vec, list):
        query_vec = np.array(raw_vec, dtype=np.float32)
    else:
        query_vec = np.array(raw_vec, dtype=np.float32)

    logger.log(
        Logger.INFO,
        f"Query: word='{word}', sense_id={sense_id}, dim={len(query_vec)}",
    )

    # Load codebook
    codebook_file = index_config.get("export_codebook", "codebook.pkl")
    codebook = load_codebook(codebook_file, logger)

    results = search_pq_with_post_verification(
        query_vec, codebook, con, cur, index_config, logger,
        top_n=top_n, final_k=final_k, use_bytea=use_bytea,
    )

    print(f"\n{'='*60}")
    print(f"Top-{final_k} nearest senses for '{query_word}':")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(
            f"  {i+1:3d}. word={r['word']:<20s} sense_id={r['sense_id']:<5d} "
            f"dist={r['distance']:.6f}"
        )

    cur.close()
    con.close()


# ===================================================================
# CLI
# ===================================================================

def print_usage():
    print("Usage:")
    print("  python main.py ingest              # full ETL pipeline")
    print("  python main.py search <word>        # demo search by word")
    print()
    print("Config files expected in config/ sub-directory:")
    print(f"  {DB_CONFIG_PATH}")
    print(f"  {PQ_CONFIG_PATH}")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "ingest":
        run_ingest()
    elif command == "search":
        if len(sys.argv) < 3:
            print("Error: please provide a word to search for.")
            print_usage()
            sys.exit(1)
        run_search(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
