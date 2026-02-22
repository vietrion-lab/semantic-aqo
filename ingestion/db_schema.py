#!/usr/bin/env python3
"""
Database schema management for the PQ + Post-verification pipeline.

Tables:
  1. sense_vectors_raw   – Full-precision vectors for post-verification
  2. pq_codebook         – PQ centroids (per subspace)
  3. pq_quantization     – Quantized codes referencing raw table IDs
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from logger import Logger


# ---------------------------------------------------------------------------
# Database creation helper
# ---------------------------------------------------------------------------

def ensure_database_exists(db_config, logger):
    """Create the target database if it does not already exist."""
    db_name = db_config.get_value('db_name')
    dsn = (
        f"dbname='postgres' "
        f"user='{db_config.get_value('username')}' "
        f"host='{db_config.get_value('host')}' "
        f"port='{db_config.get('port', 5432)}' "
        f"password='{db_config.get_value('password')}'"
    )
    try:
        con = psycopg2.connect(dsn)
        con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = con.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (db_name,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{db_name}";')
            logger.log(Logger.INFO, f"Database '{db_name}' created successfully")
        else:
            logger.log(Logger.INFO, f"Database '{db_name}' already exists")
        cur.close()
        con.close()
    except Exception as e:
        logger.log(Logger.ERROR, f"Failed to ensure database exists: {e}")
        raise


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def create_connection(db_config, logger):
    """Open a psycopg2 connection using a Configuration object."""
    # Auto-create the database if it doesn't exist
    ensure_database_exists(db_config, logger)

    dsn = (
        f"dbname='{db_config.get_value('db_name')}' "
        f"user='{db_config.get_value('username')}' "
        f"host='{db_config.get_value('host')}' "
        f"port='{db_config.get('port', 5432)}' "
        f"password='{db_config.get_value('password')}'"
    )
    try:
        con = psycopg2.connect(dsn)
        cur = con.cursor()
        logger.log(Logger.INFO, f"Connected to database: {db_config.get_value('db_name')}")
        return con, cur
    except Exception as e:
        logger.log(Logger.ERROR, f"Cannot connect to database: {e}")
        raise


# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

def get_table_definitions(index_config, use_bytea=True):
    """
    Return a list of (table_name, create_schema) tuples in creation order.

    Parameters
    ----------
    index_config : Configuration
        Must contain raw_table_name, codebook_table_name, quantization_table_name.
    use_bytea : bool
        If True, store raw vectors as bytea (matches FREDDY convention).
        If False, use float4[].
    """
    raw_table = index_config.get_value("raw_table_name")
    cb_table = index_config.get_value("codebook_table_name")
    pq_table = index_config.get_value("quantization_table_name")

    if use_bytea:
        raw_schema = (
            "(id SERIAL PRIMARY KEY, "
            "word VARCHAR(200), "
            "sense_id INT, "
            "vector bytea)"
        )
        cb_schema = (
            "(id SERIAL PRIMARY KEY, "
            "subspace_id INT, "
            "centroid_id INT, "
            "vector bytea)"
        )
    else:
        raw_schema = (
            "(id SERIAL PRIMARY KEY, "
            "word VARCHAR(200), "
            "sense_id INT, "
            "vector FLOAT4[])"
        )
        cb_schema = (
            "(id SERIAL PRIMARY KEY, "
            "subspace_id INT, "
            "centroid_id INT, "
            "vector FLOAT4[])"
        )

    pq_schema = (
        "(id INT REFERENCES {raw_table}(id), "
        "code_vector INT[])"
    ).format(raw_table=raw_table)

    return [
        (raw_table, raw_schema),
        (cb_table, cb_schema),
        (pq_table, pq_schema),
    ]


# ---------------------------------------------------------------------------
# Init / Drop helpers
# ---------------------------------------------------------------------------

def _create_vec_to_bytea(con, cur, logger):
    """Create the vec_to_bytea UDF if it does not already exist."""
    sql = """
    CREATE OR REPLACE FUNCTION vec_to_bytea(vec float4[])
    RETURNS bytea AS $$
    DECLARE
        result bytea := '';
        i int;
    BEGIN
        FOR i IN 1 .. array_length(vec, 1) LOOP
            result := result || substring(int4send(0) FROM 1 FOR 0)
                              || float4send(vec[i]);
        END LOOP;
        RETURN result;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE STRICT;
    """
    cur.execute(sql)
    con.commit()
    logger.log(Logger.INFO, "Created UDF: vec_to_bytea")


def _create_bytea_to_vec(con, cur, logger):
    """Create the bytea_to_vec UDF for reading back vectors from bytea."""
    sql = """
    CREATE OR REPLACE FUNCTION bytea_to_vec(data bytea)
    RETURNS float4[] AS $$
    DECLARE
        result float4[] := '{}';
        byte_length int := octet_length(data);
        i int := 1;
        bytes bytea;
    BEGIN
        -- Each float4 is 4 bytes
        WHILE i <= byte_length LOOP
            bytes := substring(data FROM i FOR 4);
            result := array_append(result, float4recv(bytes));
            i := i + 4;
        END LOOP;
        RETURN result;
    END;
    $$ LANGUAGE plpgsql IMMUTABLE STRICT;
    """
    cur.execute(sql)
    con.commit()
    logger.log(Logger.INFO, "Created UDF: bytea_to_vec")


def init_tables(con, cur, index_config, logger, use_bytea=True):
    """Drop existing tables and re-create them."""
    # Ensure required UDFs exist
    if use_bytea:
        _create_vec_to_bytea(con, cur, logger)
        _create_bytea_to_vec(con, cur, logger)

    table_defs = get_table_definitions(index_config, use_bytea)

    # Drop in reverse order (pq_quantization has FK -> raw)
    drop_names = ", ".join(name for name, _ in reversed(table_defs))
    cur.execute(f"DROP TABLE IF EXISTS {drop_names} CASCADE;")
    con.commit()
    logger.log(Logger.INFO, f"Dropped tables: {drop_names}")

    for name, schema in table_defs:
        cur.execute(f"CREATE TABLE {name} {schema};")
        con.commit()
        logger.log(Logger.INFO, f"Created table: {name}")


def create_index_on(table_name, index_name, column_name, con, cur, logger):
    """Create a B-tree index on a single column."""
    cur.execute(f"DROP INDEX IF EXISTS {index_name};")
    con.commit()
    cur.execute(f"CREATE INDEX {index_name} ON {table_name} ({column_name});")
    con.commit()
    logger.log(
        Logger.INFO,
        f"Created index {index_name} on {table_name}({column_name})",
    )


def create_all_indexes(index_config, con, cur, logger):
    """Create recommended indexes after data import."""
    raw_table = index_config.get_value("raw_table_name")
    pq_table = index_config.get_value("quantization_table_name")

    create_index_on(
        raw_table,
        index_config.get_value("raw_word_index_name"),
        "word",
        con, cur, logger,
    )
    create_index_on(
        pq_table,
        index_config.get_value("quantization_id_index_name"),
        "id",
        con, cur, logger,
    )


def disable_triggers(table_name, con, cur):
    cur.execute(f"ALTER TABLE {table_name} DISABLE TRIGGER ALL;")
    con.commit()


def enable_triggers(table_name, con, cur):
    cur.execute(f"ALTER TABLE {table_name} ENABLE TRIGGER ALL;")
    con.commit()
