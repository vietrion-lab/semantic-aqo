#!/usr/bin/env python3
"""
Database importer – pushes raw vectors, PQ codebook, and PQ codes
into PostgreSQL tables.

Follows the FREDDY pattern of batch-inserting with optional bytea storage.
"""

import numpy as np
import psycopg2
import psycopg2.extras

from logger import Logger


# ---------------------------------------------------------------------------
# Serialization helpers (mirrors index_creation/index_utils.py)
# ---------------------------------------------------------------------------

def _serialize_vector(vec):
    """Convert a numeric list/array to a PostgreSQL array literal '{1.0,2.0,...}'."""
    return "{" + ",".join(str(float(x)) for x in vec) + "}"


def _serialize_int_vector(vec):
    """Convert an integer list/array to a PostgreSQL array literal '{0,1,...}'."""
    return "{" + ",".join(str(int(x)) for x in vec) + "}"


# ---------------------------------------------------------------------------
# Raw vector import
# ---------------------------------------------------------------------------

def import_raw_vectors(vectors, metadata, con, cur, index_config, batch_size,
                       logger, use_bytea=True):
    """
    Insert full-precision vectors into sense_vectors_raw.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, dim)
    metadata : list[tuple[str, int]]
        Each element is (word, sense_id).
    con, cur : psycopg2 connection and cursor.
    index_config : Configuration
    batch_size : int
    logger : Logger
    use_bytea : bool
        If True, store as bytea via vec_to_bytea UDF.

    Returns
    -------
    db_ids : list[int]
        Ordered list of the SERIAL ids assigned by PostgreSQL.
    """
    table = index_config.get_value("raw_table_name")
    n = len(metadata)
    db_ids = []

    logger.log(Logger.INFO, f"Importing {n} raw vectors into {table} ...")

    values_batch = []
    for i, (word, sense_id) in enumerate(metadata):
        vec_str = _serialize_vector(vectors[i])
        values_batch.append((word, sense_id, vec_str))

        if len(values_batch) >= batch_size or i == n - 1:
            if use_bytea:
                sql = (
                    f"INSERT INTO {table} (word, sense_id, vector) "
                    f"VALUES (%s, %s, vec_to_bytea(%s::float4[])) RETURNING id"
                )
            else:
                sql = (
                    f"INSERT INTO {table} (word, sense_id, vector) "
                    f"VALUES (%s, %s, %s) RETURNING id"
                )

            for row in values_batch:
                cur.execute(sql, row)
                db_ids.append(cur.fetchone()[0])
            con.commit()

            logger.log(Logger.INFO, f"  raw vectors inserted: {i + 1}/{n}")
            values_batch = []

    logger.log(Logger.INFO, f"Raw vector import complete: {len(db_ids)} rows")
    return db_ids


# ---------------------------------------------------------------------------
# Codebook import
# ---------------------------------------------------------------------------

def import_codebook(codebook, con, cur, index_config, logger, use_bytea=True):
    """
    Insert PQ codebook centroids into pq_codebook.

    Parameters
    ----------
    codebook : np.ndarray, shape (m, k, d_sub)
    """
    table = index_config.get_value("codebook_table_name")
    m, k, d_sub = codebook.shape

    logger.log(
        Logger.INFO,
        f"Importing codebook into {table}: m={m}, k={k}, d_sub={d_sub}",
    )

    for subspace_id in range(m):
        values = []
        for centroid_id in range(k):
            vec_str = _serialize_vector(codebook[subspace_id, centroid_id])
            values.append((subspace_id, centroid_id, vec_str))

        if use_bytea:
            sql = (
                f"INSERT INTO {table} (subspace_id, centroid_id, vector) "
                f"VALUES (%s, %s, vec_to_bytea(%s::float4[]))"
            )
        else:
            sql = (
                f"INSERT INTO {table} (subspace_id, centroid_id, vector) "
                f"VALUES (%s, %s, %s)"
            )

        cur.executemany(sql, values)
        con.commit()
        logger.log(Logger.INFO, f"  codebook subspace {subspace_id}/{m} imported")

    logger.log(Logger.INFO, "Codebook import complete")


# ---------------------------------------------------------------------------
# PQ codes import
# ---------------------------------------------------------------------------

def import_pq_codes(codes, db_ids, con, cur, index_config, batch_size, logger):
    """
    Insert quantized codes into pq_quantization.

    Parameters
    ----------
    codes : np.ndarray, shape (N, m), dtype uint8
    db_ids : list[int]
        Matching raw table IDs (same order as codes).
    """
    table = index_config.get_value("quantization_table_name")
    n = len(db_ids)

    logger.log(Logger.INFO, f"Importing {n} PQ codes into {table} ...")

    sql = f"INSERT INTO {table} (id, code_vector) VALUES (%s, %s)"
    values_batch = []

    for i in range(n):
        code_str = _serialize_int_vector(codes[i])
        values_batch.append((db_ids[i], code_str))

        if len(values_batch) >= batch_size or i == n - 1:
            cur.executemany(sql, values_batch)
            con.commit()
            logger.log(Logger.INFO, f"  PQ codes inserted: {i + 1}/{n}")
            values_batch = []

    logger.log(Logger.INFO, "PQ code import complete")
