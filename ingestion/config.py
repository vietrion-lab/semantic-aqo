#!/usr/bin/env python3
"""
Configuration loader for the ingestion pipeline.
Mirrors the pattern from index_creation/config.py with additional helpers.
"""

import json
import os


class Configuration:
    """JSON-based configuration reader."""

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")
        with open(filename, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get_value(self, key):
        return self.data[key]

    def has_key(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __repr__(self):
        return f"Configuration({json.dumps(self.data, indent=2)})"
