#ifndef CALC_H
#define CALC_H

#include "fmgr.h"

Datum aqo_add(PG_FUNCTION_ARGS);
Datum aqo_subtract(PG_FUNCTION_ARGS);
Datum aqo_multiply(PG_FUNCTION_ARGS);
Datum aqo_divide(PG_FUNCTION_ARGS);
Datum aqo_modulus(PG_FUNCTION_ARGS);

#endif /* CALC_H */