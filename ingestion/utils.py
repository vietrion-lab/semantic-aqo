#!/usr/bin/env python3
"""
Utility functions shared across the ingestion pipeline.
"""

import numpy as np


def serialize_vector(vec):
    """Convert a numeric iterable to a PostgreSQL array literal '{1.0,2.0,...}'."""
    return "{" + ",".join(str(x) for x in vec) + "}"


def cosine_distance(a, b):
    """Cosine distance between two vectors: 1 - cosine_similarity."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def euclidean_distance(a, b):
    """Squared Euclidean distance between two vectors."""
    diff = a - b
    return float(np.dot(diff, diff))


def normalize_vector(vec):
    """L2-normalize a vector in-place, return it."""
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec
