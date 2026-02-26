#ifndef PG_COMPAT_H
#define PG_COMPAT_H

#ifdef LOCAL_TEST
    #include <stdlib.h>
    #include <string.h>
    #include <stdio.h>
    #define palloc malloc
    #define palloc0(sz) calloc(1, (sz))
    #define pfree free
    #define repalloc realloc
    #define pstrdup strdup
    #define WARNING 1
    #define LOG 3
    #define elog(level, fmt, ...) printf("[LOG]: " fmt "\n", ##__VA_ARGS__)
#else
    #include "postgres.h"
    #include "utils/palloc.h"
#endif

#endif