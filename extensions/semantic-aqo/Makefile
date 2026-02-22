# src/semantic-aqo/Makefile

# Debug Flags - IMPORTANT FOR DEBUGGING
CFLAGS += -g -O0 -Wall -Wextra -Werror=format-security
CXXFLAGS += -g -O0 -Wall -Wextra

# Name of the extension
EXTENSION = semantic_aqo

# Name of the shared library file
MODULE_big = semantic_aqo

# SQL file containing function definitions
DATA = semantic_aqo--1.0.sql

# OBJS - List all object files from subdirectories
OBJS = \
    semantic_aqo.o \
    utils/calc.o \
    model/model_loader.o \
    storage/storage.o \

# Regression tests
REGRESS = semantic_aqo
REGRESS_OPTS = --inputdir=test --outputdir=test/results --port=5433 --use-existing --user=postgres

# ----- PGXS magic lines (PG build system) -----
PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Debug target - verify debug symbols
.PHONY: debug-check
debug-check: $(MODULE_big).so
	@objdump -t $(MODULE_big).so | grep -i debug && echo "✅ Debug symbols present" || echo "❌ No debug symbols"
	@file $(MODULE_big).so

