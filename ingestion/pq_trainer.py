#!/usr/bin/env python3
"""
Product Quantization trainer using FAISS.

Responsibilities:
  - Train a FAISS ProductQuantizer on the embedding matrix.
  - Extract the codebook (centroids per subspace).
  - Encode all vectors to PQ codes.
  - Save / load the trained codebook to disk via pickle.
"""

import os
import pickle

import faiss
import numpy as np

from logger import Logger


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pq(vectors, dim, m_subspaces, nbits, logger, train_size=None):
    """
    Train a FAISS ProductQuantizer.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, dim)
    dim : int
    m_subspaces : int
        Number of subspaces (dim must be divisible by m).
    nbits : int
        Bits per sub-quantizer (8 → 256 centroids).
    logger : Logger
    train_size : int or None
        If set, only train on the first `train_size` vectors.

    Returns
    -------
    pq : faiss.ProductQuantizer
    """
    assert dim % m_subspaces == 0, (
        f"Dimension {dim} must be divisible by m_subspaces {m_subspaces}"
    )

    train_vecs = vectors[:train_size] if train_size else vectors
    train_vecs = np.ascontiguousarray(train_vecs, dtype=np.float32)

    # FAISS requires n_training >= 2^nbits centroids per subspace.
    # If dataset is too small, augment with noisy duplicates.
    k = 2 ** nbits
    if len(train_vecs) < k:
        logger.log(
            Logger.WARNING,
            f"Training set ({len(train_vecs)}) < centroids ({k}). "
            f"Augmenting with noisy duplicates.",
        )
        reps = int(np.ceil(k / len(train_vecs)))
        augmented = np.tile(train_vecs, (reps, 1))[:k]
        # Add small Gaussian noise to avoid degenerate centroids
        noise_scale = np.std(train_vecs) * 0.001
        augmented += np.random.randn(*augmented.shape).astype(np.float32) * noise_scale
        train_vecs = np.ascontiguousarray(augmented, dtype=np.float32)

    logger.log(
        Logger.INFO,
        f"Training PQ: dim={dim}, m={m_subspaces}, nbits={nbits}, "
        f"train_vectors={len(train_vecs)}",
    )

    pq = faiss.ProductQuantizer(dim, m_subspaces, nbits)
    pq.train(train_vecs)

    logger.log(Logger.INFO, "PQ training complete")
    return pq


# ---------------------------------------------------------------------------
# Codebook extraction
# ---------------------------------------------------------------------------

def extract_codebook(pq, m_subspaces, nbits):
    """
    Extract centroids from a trained FAISS ProductQuantizer.

    Returns
    -------
    codebook : np.ndarray, shape (m, k, d_sub)
        Where k = 2**nbits and d_sub = dim / m.
    """
    n_centroids = 2 ** nbits
    d_sub = pq.dsub
    centroids = faiss.vector_to_array(pq.centroids).reshape(
        m_subspaces, n_centroids, d_sub
    )
    return centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _decode_pq_codes(pq, codes_packed):
    """
    Decode bit-packed PQ codes into centroid IDs.
    
    For nbits < 8, FAISS packs multiple codes into bytes.
    This function extracts the actual centroid ID for each subspace.
    
    Parameters
    ----------
    pq : faiss.ProductQuantizer
    codes_packed : np.ndarray, shape (N, code_size), dtype uint8
        Bit-packed codes from pq.compute_codes()
        
    Returns
    -------
    codes_decoded : np.ndarray, shape (N, m), dtype uint8
        Centroid IDs for each subspace
    """
    n = codes_packed.shape[0]
    m = pq.M
    nbits = pq.nbits
    
    codes_decoded = np.zeros((n, m), dtype=np.uint8)
    
    if nbits == 8:
        # No bit-packing, direct copy
        return codes_packed
    
    # For nbits < 8, manually decode bit-packed format
    for i in range(n):
        bit_offset = 0
        for j in range(m):
            # Extract nbits starting at bit_offset
            byte_idx = bit_offset // 8
            bit_in_byte = bit_offset % 8
            
            # Read enough bytes to cover nbits
            if bit_in_byte + nbits <= 8:
                # Fits in one byte
                mask = (1 << nbits) - 1
                code = (codes_packed[i, byte_idx] >> bit_in_byte) & mask
            else:
                # Spans two bytes
                bits_from_first = 8 - bit_in_byte
                bits_from_second = nbits - bits_from_first
                
                code_low = (codes_packed[i, byte_idx] >> bit_in_byte)
                code_high = (codes_packed[i, byte_idx + 1] & ((1 << bits_from_second) - 1))
                code = code_low | (code_high << bits_from_first)
            
            codes_decoded[i, j] = code
            bit_offset += nbits
    
    return codes_decoded


def encode_vectors(pq, vectors):
    """
    Encode vectors to PQ codes using a trained ProductQuantizer.

    Parameters
    ----------
    pq : faiss.ProductQuantizer
    vectors : np.ndarray, shape (N, dim)

    Returns
    -------
    codes : np.ndarray, shape (N, m), dtype uint8
        Decoded centroid IDs for each subspace
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    codes_packed = pq.compute_codes(vectors)
    # Decode bit-packed format to centroid IDs
    codes = _decode_pq_codes(pq, codes_packed)
    return codes


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_codebook(codebook, filepath, logger):
    """Pickle the codebook array to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(codebook, f)
    logger.log(Logger.INFO, f"Codebook saved to {filepath}")


def load_codebook(filepath, logger):
    """Load a previously saved codebook from disk."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Codebook file not found: {filepath}")
    with open(filepath, "rb") as f:
        codebook = pickle.load(f)
    logger.log(Logger.INFO, f"Codebook loaded from {filepath}")
    return codebook


def save_pq(pq_obj, filepath, logger):
    """Save the full FAISS ProductQuantizer to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(pq_obj, f)
    logger.log(Logger.INFO, f"PQ object saved to {filepath}")


def load_pq(filepath, logger):
    """Load a previously saved FAISS ProductQuantizer from disk."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PQ file not found: {filepath}")
    with open(filepath, "rb") as f:
        pq_obj = pickle.load(f)
    logger.log(Logger.INFO, f"PQ object loaded from {filepath}")
    return pq_obj
