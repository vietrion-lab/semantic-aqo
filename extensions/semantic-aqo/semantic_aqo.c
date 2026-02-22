/* src/semantic-aqo/semantic_aqo.c */

#include "postgres.h"       // Main Postgres header
#include "fmgr.h"           // Header for function manager
#include "utils/builtins.h" // Needed for cstring_to_text

// Include sub-module headers
#include "utils/calc.h"
#include "model/model_loader.h"
#include "storage/storage.h"

// Magic block required for all extensions
PG_MODULE_MAGIC;

// Version function
PG_FUNCTION_INFO_V1(aqo_version);
Datum aqo_version(PG_FUNCTION_ARGS)
{
  PG_RETURN_TEXT_P(cstring_to_text("semantic-aqo version 1.0"));
}