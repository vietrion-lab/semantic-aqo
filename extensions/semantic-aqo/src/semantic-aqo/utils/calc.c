// src/semantic-aqo/utils/calc.c

#include "postgres.h"       // PostgreSQL server header
#include "fmgr.h"           // Header for function manager
#include "utils/builtins.h" // Needed for cstring_to_text
#include <math.h>           // For pow function

#include "utils/calc.h"

PG_FUNCTION_INFO_V1(aqo_add);
Datum aqo_add(PG_FUNCTION_ARGS)
{
  int32 arg1 = PG_GETARG_INT32(0);
  int32 arg2 = PG_GETARG_INT32(1);

  int32 result = arg1 + arg2;

  PG_RETURN_INT32(result);
}

PG_FUNCTION_INFO_V1(aqo_subtract);
Datum aqo_subtract(PG_FUNCTION_ARGS)
{
  int32 arg1 = PG_GETARG_INT32(0);
  int32 arg2 = PG_GETARG_INT32(1);

  int32 result = arg1 - arg2;

  PG_RETURN_INT32(result);
}

PG_FUNCTION_INFO_V1(aqo_multiply);
Datum aqo_multiply(PG_FUNCTION_ARGS)
{
  int32 arg1 = PG_GETARG_INT32(0);
  int32 arg2 = PG_GETARG_INT32(1);

  int32 result = arg1 * arg2;

  PG_RETURN_INT32(result);
}

PG_FUNCTION_INFO_V1(aqo_divide);
Datum aqo_divide(PG_FUNCTION_ARGS)
{
  int32 arg1 = PG_GETARG_INT32(0);
  int32 arg2 = PG_GETARG_INT32(1);

  int32 result = arg1 / arg2;

  PG_RETURN_INT32(result);
}

PG_FUNCTION_INFO_V1(aqo_modulus);
Datum aqo_modulus(PG_FUNCTION_ARGS)
{
  int32 arg1 = PG_GETARG_INT32(0);
  int32 arg2 = PG_GETARG_INT32(1);

  int32 result = arg1 % arg2;

  PG_RETURN_INT32(result);
}

