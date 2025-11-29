/* src/semantic-aqo/semantic-aqo--1.0.sql */

-- Block direct execution of this file with \i
\echo Use "CREATE EXTENSION semantic_aqo" to load this file. \quit


CREATE FUNCTION aqo_version()
RETURNS text
AS 'MODULE_PATHNAME', 'aqo_version'
LANGUAGE C IMMUTABLE STRICT;


CREATE FUNCTION aqo_add(integer, integer)
RETURNS integer
AS 'MODULE_PATHNAME', 'aqo_add'
LANGUAGE C IMMUTABLE STRICT;


CREATE FUNCTION aqo_subtract(integer, integer)
RETURNS integer
AS 'MODULE_PATHNAME', 'aqo_subtract'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION aqo_multiply(integer, integer)
RETURNS integer
AS 'MODULE_PATHNAME', 'aqo_multiply'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION aqo_divide(integer, integer)
RETURNS integer
AS 'MODULE_PATHNAME', 'aqo_divide'
LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION aqo_modulus(integer, integer)
RETURNS integer
AS 'MODULE_PATHNAME', 'aqo_modulus'
LANGUAGE C IMMUTABLE STRICT;


-- Model module functions
CREATE FUNCTION aqo_model_info()
RETURNS text
AS 'MODULE_PATHNAME', 'aqo_model_info'
LANGUAGE C IMMUTABLE STRICT;


-- Storage module functions
CREATE FUNCTION aqo_storage_info()
RETURNS text
AS 'MODULE_PATHNAME', 'aqo_storage_info'
LANGUAGE C IMMUTABLE STRICT;


