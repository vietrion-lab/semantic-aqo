#!/usr/bin/env python3
"""
PQ approximate search with Post-verification.

Two-phase search:
  Phase 1 – PQ approximate search using precomputed distance tables
            to retrieve top-N candidates from pq_quantization + pq_codebook.
  Phase 2 – Post-verification: compute exact distances on the raw vectors
            of the N candidates and return top-K.

Both phases are executed as SQL queries via psycopg2 so the heavy lifting
stays inside PostgreSQL (matching the FREDDY approach).
"""

import numpy as np
import psycopg2

from logger import Logger


# ---------------------------------------------------------------------------
# Helper: build PQ distance lookup table
# ---------------------------------------------------------------------------

def _build_distance_table(query_vec, codebook):
    """
    Precompute squared L2 distance from the query sub-vector to every centroid
    in every subspace.

    Parameters
    ----------
    query_vec : np.ndarray, shape (dim,)
    codebook : np.ndarray, shape (m, k, d_sub)

    Returns
    -------
    dist_table : np.ndarray, shape (m, k)
        dist_table[j][c] = ||query_sub_j - centroid_j_c||^2
    """
    m, k, d_sub = codebook.shape
    dim = m * d_sub
    assert len(query_vec) == dim, (
        f"Query dim {len(query_vec)} != codebook dim {dim}"
    )

    dist_table = np.zeros((m, k), dtype=np.float32)
    for j in range(m):
        sub_q = query_vec[j * d_sub : (j + 1) * d_sub]
        # broadcast: (k, d_sub) - (d_sub,) -> squared L2
        diff = codebook[j] - sub_q
        dist_table[j] = np.sum(diff ** 2, axis=1)

    return dist_table


# ---------------------------------------------------------------------------
# Phase 1: PQ approximate search  (in-Python, operates on DB data)
# ---------------------------------------------------------------------------

def pq_search(query_vec, codebook, con, cur, index_config, top_n, logger):
    """
    Approximate nearest-neighbour search using PQ codes stored in PostgreSQL.

    1. Build a precomputed distance lookup table from the query.
    2. Fetch all PQ codes from the DB.
    3. Sum up sub-distances via the lookup table.
    4. Return top_n candidate IDs sorted by approximate distance.

    Parameters
    ----------
    query_vec : np.ndarray, shape (dim,)
    codebook : np.ndarray, shape (m, k, d_sub)
    con, cur : psycopg2 connection / cursor
    index_config : Configuration
    top_n : int
        Number of candidates to return (e.g. 500).
    logger : Logger

    Returns
    -------
    candidate_ids : list[int]
        DB IDs of the top_n approximate nearest neighbours.
    approx_dists : list[float]
        Corresponding approximate squared L2 distances.
    """
    pq_table = index_config.get_value("quantization_table_name")

    # 1. Precompute distance table
    dist_table = _build_distance_table(query_vec, codebook)
    m = codebook.shape[0]

    # 2. Fetch all PQ codes
    cur.execute(f"SELECT id, code_vector FROM {pq_table}")
    rows = cur.fetchall()

    logger.log(Logger.INFO, f"PQ search: scoring {len(rows)} candidates ...")

    # 3. Score each candidate
    ids = np.empty(len(rows), dtype=np.int64)
    scores = np.empty(len(rows), dtype=np.float32)

    for idx, (db_id, code_vec) in enumerate(rows):
        total = 0.0
        for j in range(m):
            total += dist_table[j, code_vec[j]]
        ids[idx] = db_id
        scores[idx] = total

    # 4. Top-N (cap to available candidates)
    top_n = min(top_n, len(rows))
    if top_n >= len(rows):
        # All candidates are within top_n, just sort them all
        top_indices = np.argsort(scores)[:top_n]
    else:
        top_indices = np.argpartition(scores, top_n)[:top_n]
        top_indices = top_indices[np.argsort(scores[top_indices])]

    candidate_ids = ids[top_indices].tolist()
    approx_dists = scores[top_indices].tolist()

    logger.log(
        Logger.INFO,
        f"PQ search complete: top-{top_n} candidates selected",
    )
    return candidate_ids, approx_dists


# ---------------------------------------------------------------------------
# Phase 2: Post-verification (exact re-ranking)
# ---------------------------------------------------------------------------

def post_verify(query_vec, candidate_ids, con, cur, index_config, final_k,
                logger, use_bytea=True):
    """
    Re-rank candidate IDs by exact L2 distance using the raw vectors.

    Parameters
    ----------
    query_vec : np.ndarray, shape (dim,)
    candidate_ids : list[int]
    con, cur : psycopg2 connection / cursor
    index_config : Configuration
    final_k : int
        Number of final results to return.
    logger : Logger
    use_bytea : bool

    Returns
    -------
    results : list[dict]
        Each dict has keys: id, word, sense_id, distance.
        Sorted ascending by exact L2 distance.
    """
    raw_table = index_config.get_value("raw_table_name")

    if not candidate_ids:
        return []

    # Build an IN-clause
    id_placeholders = ",".join(["%s"] * len(candidate_ids))

    # Always fetch bytea directly and convert in Python
    sql = (
        f"SELECT id, word, sense_id, vector "
        f"FROM {raw_table} WHERE id IN ({id_placeholders})"
    )

    cur.execute(sql, tuple(candidate_ids))
    rows = cur.fetchall()

    logger.log(
        Logger.INFO,
        f"Post-verification: computing exact distances for {len(rows)} candidates",
    )

    results = []
    for db_id, word, sense_id, raw_vec in rows:
        # Convert bytea or array to numpy
        if isinstance(raw_vec, (bytes, memoryview)):
            # PostgreSQL float4send stores in big-endian (network byte order)
            raw_arr = np.frombuffer(raw_vec, dtype='>f4').astype(np.float32)
        elif isinstance(raw_vec, list):
            raw_arr = np.array(raw_vec, dtype=np.float32)
        else:
            raw_arr = np.array(raw_vec, dtype=np.float32)

        dist = float(np.sum((query_vec - raw_arr) ** 2))
        results.append({
            "id": db_id,
            "word": word,
            "sense_id": sense_id,
            "distance": dist,
        })

    # Sort by exact distance and return top-K
    results.sort(key=lambda r: r["distance"])
    results = results[:final_k]

    logger.log(
        Logger.INFO,
        f"Post-verification complete: returning top-{final_k} results",
    )
    return results


# ---------------------------------------------------------------------------
# Combined search (convenience wrapper)
# ---------------------------------------------------------------------------

def search_pq_with_post_verification(
    query_vec, codebook, con, cur, index_config, logger,
    top_n=500, final_k=10, use_bytea=True,
):
    """
    Full two-phase search: PQ approximate → post-verification exact.

    Parameters
    ----------
    query_vec : np.ndarray, shape (dim,)
    codebook : np.ndarray, shape (m, k, d_sub)
    con, cur : psycopg2 connection / cursor
    index_config : Configuration
    logger : Logger
    top_n : int
        Candidates from PQ phase (default 500).
    final_k : int
        Final results after exact re-rank (default 10).
    use_bytea : bool

    Returns
    -------
    results : list[dict]
        Final top-K results with exact distances.
    """
    logger.log(Logger.INFO, "=== Starting PQ + Post-verification search ===")

    candidate_ids, _ = pq_search(
        query_vec, codebook, con, cur, index_config, top_n, logger,
    )

    results = post_verify(
        query_vec, candidate_ids, con, cur, index_config, final_k, logger,
        use_bytea=use_bytea,
    )

    logger.log(Logger.INFO, "=== Search complete ===")
    return results
