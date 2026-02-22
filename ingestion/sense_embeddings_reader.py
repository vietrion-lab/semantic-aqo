#!/usr/bin/env python3
"""
Binary reader for sense_embeddings.bin and vocab.bin files.

File formats (all little-endian):

vocab.bin:
  Header: num_records (int32)
  Body per record:
    word_length (int32) -> word (UTF-8 bytes) -> word_id (int32)

sense_embeddings.bin:
  Header: num_records (int32), embedding_dim (int32)
  Body per record:
    word_length (int32) -> word (UTF-8 bytes) -> sense_id (int32)
    -> embedding (dim * float32)
"""

import struct
import numpy as np

from logger import Logger


def read_vocab(file_path, logger=None):
    """
    Read vocab.bin and return a list of (word, word_id) tuples.

    Parameters
    ----------
    file_path : str
        Path to vocab.bin.
    logger : Logger, optional
        Logger instance for progress reporting.

    Returns
    -------
    vocab : list[tuple[str, int]]
        List of (word, word_id) pairs.
    """
    vocab = []
    with open(file_path, "rb") as f:
        (num_records,) = struct.unpack("<i", f.read(4))
        if logger:
            logger.log(Logger.INFO, f"Reading vocab: {num_records} records")

        for idx in range(num_records):
            (word_len,) = struct.unpack("<i", f.read(4))
            word = f.read(word_len).decode("utf-8")
            (word_id,) = struct.unpack("<i", f.read(4))
            vocab.append((word, word_id))

            if logger and (idx + 1) % 50000 == 0:
                logger.log(Logger.INFO, f"  vocab read: {idx + 1}/{num_records}")

    if logger:
        logger.log(Logger.INFO, f"Vocab loaded: {len(vocab)} words")
    return vocab


def read_sense_embeddings(file_path, logger=None, normalization=False):
    """
    Read sense_embeddings.bin and return vectors + metadata.

    Parameters
    ----------
    file_path : str
        Path to sense_embeddings.bin.
    logger : Logger, optional
        Logger instance for progress reporting.
    normalization : bool
        If True, L2-normalize each vector after reading.

    Returns
    -------
    vectors : np.ndarray of shape (num_records, dim), dtype float32
    metadata : list[tuple[str, int]]
        Each element is (word, sense_id).
    dim : int
        Embedding dimensionality.
    """
    metadata = []

    with open(file_path, "rb") as f:
        # --- Header ---
        (num_records,) = struct.unpack("<i", f.read(4))
        (dim,) = struct.unpack("<i", f.read(4))

        if logger:
            logger.log(
                Logger.INFO,
                f"Reading sense embeddings: {num_records} records, dim={dim}",
            )

        vectors = np.zeros((num_records, dim), dtype=np.float32)

        for idx in range(num_records):
            # word
            (word_len,) = struct.unpack("<i", f.read(4))
            word = f.read(word_len).decode("utf-8")
            # sense_id
            (sense_id,) = struct.unpack("<i", f.read(4))
            # embedding
            embedding = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()

            if normalization:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

            vectors[idx] = embedding
            metadata.append((word, sense_id))

            if logger and (idx + 1) % 10000 == 0:
                logger.log(
                    Logger.INFO,
                    f"  embeddings read: {idx + 1}/{num_records}",
                )

    if logger:
        logger.log(
            Logger.INFO,
            f"Embeddings loaded: {len(metadata)} records, dim={dim}",
        )

    return vectors, metadata, dim
