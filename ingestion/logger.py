#!/usr/bin/env python3
"""
Logger utility for the ingestion pipeline.
Mirrors the pattern from index_creation/logger.py with fixes.
"""

import sys
import time
import datetime


class Logger:

    LEVELS = INFO, WARNING, ERROR = range(3)

    LEVEL_NAMES = {
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
    }

    def __init__(self, filename=""):
        self.filename = filename
        if not filename:
            self.stdout = True
            self.fileout = False
            self.dest_file = None
        else:
            self.dest_file = open(filename, "w", encoding="utf-8")
            self.stdout = False
            self.fileout = True

    def log(self, level, message):
        ts = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        output = f"{Logger.LEVEL_NAMES[level]} [{ts}]: {message}"
        if self.stdout:
            print(output)
        if self.fileout and self.dest_file:
            self.dest_file.write(output + "\n")
            self.dest_file.flush()

    def close(self):
        if self.dest_file:
            self.dest_file.close()
