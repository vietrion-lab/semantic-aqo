/*
 * compat_pg.h
 *
 * Minimal shim that maps PostgreSQL allocator / log macros → stdlib,
 * allowing utils/sql_preprocessor.c (and similar) to compile outside PG.
 *
 * Include this before any PostgreSQL header in non-PG translation units.
 */
#ifndef COMPAT_PG_H
#define COMPAT_PG_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>

/* ---- allocators ---- */
#define palloc(sz)        malloc(sz)
#define palloc0(sz)       calloc(1, (sz))
#define pfree(p)          free(p)
#define repalloc(p, sz)   realloc((p), (sz))

static inline char *pstrdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *d = malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

/* ---- logging ---- */
#define DEBUG1  10
#define LOG     20
#define INFO    30
#define WARNING 40
#define ERROR   50

#define elog(level, ...) \
    do { \
        if ((level) >= WARNING) \
            fprintf(stderr, "[compat_pg] " __VA_ARGS__), fputc('\n', stderr); \
    } while (0)

/* ---- misc types that postgres.h normally provides ---- */
typedef unsigned int  Oid;
typedef signed char   int8;
typedef short         int16;
typedef int           int32;

#endif /* COMPAT_PG_H */
