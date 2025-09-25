import re
from typing import Dict, List


SQL_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "JOIN",
    "ON",
    "AND",
    "OR",
    "GROUP",
    "BY",
    "ORDER",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "OUTER",
    "CROSS",
    "UNION",
    "DISTINCT",
    "LIMIT",
    "OFFSET",
    "AS",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "ASC",
    "DESC",
    "HAVING",
    "IN",
    "IS",
    "NOT",
    "NULL",
    "TRUE",
    "FALSE",
    "WITH",
    "INSERT",
    "UPDATE",
    "DELETE",
    "VALUES",
    "SET",
    "INTO",
    "TOP",
    "ALL",
    "ANY",
    "EXISTS",
    "BETWEEN",
    "LIKE",
    "USING",
}

SQL_FUNCTIONS = {
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "COALESCE",
    "NVL",
    "UPPER",
    "LOWER",
    "ROUND",
    "ABS",
    "LENGTH",
    "SUBSTRING",
    "TRIM",
    "RTRIM",
    "LTRIM",
    "DATE_TRUNC",
    "EXTRACT",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
}


IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_][\w$]*)\b")


FROM_SECTION_PATTERN = re.compile(
    r"\bFROM\b\s+(?P<section>.*?)(?=(\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bUNION\b|\bINTERSECT\b|\bEXCEPT\b|\bJOIN\b|$))",
    re.IGNORECASE | re.DOTALL,
)


JOIN_PATTERN = re.compile(
    r"\b(?:(?P<prefix>(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?|INNER|OUTER|CROSS|NATURAL)\s+)?JOIN\s+(?P<table>(?:\w+\.)*\w+)(?:\s+(?:AS\s+)?(?P<alias>\w+))?",
    re.IGNORECASE,
)


UPDATE_PATTERN = re.compile(
    r"\bUPDATE\b\s+(?P<table>(?:\w+\.)*\w+)(?:\s+(?:AS\s+)?(?P<alias>\w+))?",
    re.IGNORECASE,
)


INSERT_PATTERN = re.compile(
    r"\bINSERT\s+INTO\b\s+(?P<table>(?:\w+\.)*\w+)",
    re.IGNORECASE,
)


OUTPUT_ALIAS_PATTERN = re.compile(
    r"\bAS\s+([A-Za-z_][\w$]*)\b",
    re.IGNORECASE,
)


class PreprocessingPipeline:
    def __init__(self):
        self.alias_counter = 1
        self.alias_tokens: Dict[str, str] = {}
        self.alias_names: Dict[str, str] = {}
        self.alias_to_table: Dict[str, str] = {}
        self.output_aliases: set[str] = set()
        
    def _reset_aliases(self):
        """Reset state for each query."""
        self.alias_counter = 1
        self.alias_tokens = {}
        self.alias_names = {}
        self.alias_to_table = {}
        self.output_aliases = set()

    def _get_alias_token(self, alias_key: str) -> str:
        """Return the token assigned to the alias key, creating it if needed."""
        key = alias_key.lower()
        if key not in self.alias_tokens:
            self.alias_tokens[key] = f"<ALIAS_T{self.alias_counter}>"
            self.alias_counter += 1
        return self.alias_tokens[key]
    
    def _replace_literals(self, query: str) -> str:
        """Replace literal values (strings, numbers, booleans, etc.) with special tokens."""
        # Replace timestamp literals before more general numeric replacements
        query = re.sub(
            r"(?<!\w)(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?!\w)",
            "<TIMESTAMP>",
            query,
        )
        query = re.sub(
            r"(?<!\w)(\d{4}-\d{2}-\d{2})(?!\w)",
            "<DATE>",
            query,
        )

        # Replace string literals (handle escaped quotes)
        query = re.sub(r"'(?:\\.|[^'])*'", "<STR>", query)
        query = re.sub(r'"(?:\\.|[^"])*"', "<STR>", query)

        # Replace numeric literals (floats first, then integers)
        query = re.sub(r"\b\d+\.\d+\b", "<NUM>", query)
        query = re.sub(r"\b\d+\b", "<NUM>", query)
        
        # Replace boolean literals
        query = re.sub(r"\bTRUE\b", "<BOOL_TRUE>", query, flags=re.IGNORECASE)
        query = re.sub(r"\bFALSE\b", "<BOOL_FALSE>", query, flags=re.IGNORECASE)

        # Replace NULL literals
        query = re.sub(r"\bNULL\b", "<NULL>", query, flags=re.IGNORECASE)
        
        return query
    
    @staticmethod
    def _split_table_segment(segment: str) -> List[str]:
        """Split a FROM clause segment by commas, respecting parentheses."""
        entries: List[str] = []
        current: List[str] = []
        depth = 0

        for char in segment:
            if char == '(':  # track subquery depth
                depth += 1
            elif char == ')':
                depth = max(depth - 1, 0)

            if char == ',' and depth == 0:
                entry = ''.join(current).strip()
                if entry:
                    entries.append(entry)
                current = []
                continue

            current.append(char)

        final_entry = ''.join(current).strip()
        if final_entry:
            entries.append(final_entry)

        return entries

    def _register_table_entry(self, entry: str) -> str:
        """Register a table expression and return its processed representation."""
        stripped = entry.strip()
        if not stripped:
            return stripped

        if stripped.startswith('('):  # Subquery; keep as-is
            return stripped

        match = re.match(
            r"((?:\w+\.)*\w+)(?:\s+(?:AS\s+)?(\w+))?",
            stripped,
            flags=re.IGNORECASE,
        )
        if not match:
            return stripped

        table_name = match.group(1)
        alias = match.group(2)

        alias_candidate = alias if alias and alias.upper() not in SQL_KEYWORDS else None
        alias_candidate = alias_candidate or table_name.split('.')[-1]

        alias_key = alias_candidate.lower()
        alias_token = self._get_alias_token(alias_key)
        self.alias_names[alias_key] = alias_candidate
        self.alias_to_table[alias_key] = table_name

        return alias_token

    def _process_from_clauses(self, query: str) -> str:
        """Process FROM clauses, registering tables and aliases."""

        def replacer(match: re.Match[str]) -> str:
            section = match.group('section')
            parts = self._split_table_segment(section)
            tokens = ["FROM"]
            for index, part in enumerate(parts, start=1):
                alias_token = self._register_table_entry(part)
                tokens.extend(["<TAB>", alias_token])
                if index < len(parts):
                    tokens.append(",")
            return " ".join(tokens) + " "

        return FROM_SECTION_PATTERN.sub(replacer, query)

    def _process_join_clauses(self, query: str) -> str:
        """Process JOIN clauses, registering each table alias."""

        def replacer(match: re.Match[str]) -> str:
            table = match.group('table')
            alias = match.group('alias')
            prefix = match.group('prefix')

            alias_candidate = alias if alias and alias.upper() not in SQL_KEYWORDS else None
            if alias_candidate is None:
                alias_candidate = self.alias_to_table.get(table.lower())
            alias_candidate = alias_candidate or table.split('.')[-1]

            alias_key = alias_candidate.lower()
            alias_token = self._get_alias_token(alias_key)
            self.alias_names[alias_key] = alias_candidate
            self.alias_to_table[alias_key] = table

            prefix_part = f"{prefix.upper()} " if prefix else ''
            return f" {prefix_part}JOIN <TAB> {alias_token}"

        return JOIN_PATTERN.sub(replacer, query)

    def _process_update_clauses(self, query: str) -> str:
        """Handle UPDATE statements by normalizing the target table."""

        def replacer(match: re.Match[str]) -> str:
            table = match.group('table')
            alias = match.group('alias')

            alias_candidate = alias if alias and alias.upper() not in SQL_KEYWORDS else None
            alias_candidate = alias_candidate or table.split('.')[-1]

            alias_key = alias_candidate.lower()
            alias_token = self._get_alias_token(alias_key)
            self.alias_names[alias_key] = alias_candidate
            self.alias_to_table[alias_key] = table

            return f"UPDATE <TAB> {alias_token}"

        return UPDATE_PATTERN.sub(replacer, query)

    def _process_insert_clauses(self, query: str) -> str:
        """Handle INSERT INTO statements by normalizing the target table."""

        def replacer(match: re.Match[str]) -> str:
            table = match.group('table')
            alias_candidate = table.split('.')[-1]
            alias_key = alias_candidate.lower()
            alias_token = self._get_alias_token(alias_key)
            self.alias_names[alias_key] = alias_candidate
            self.alias_to_table[alias_key] = table
            return f"INSERT INTO <TAB> {alias_token}"

        return INSERT_PATTERN.sub(replacer, query)

    def _replace_alias_column_references(self, query: str) -> str:
        """Replace alias.column and standalone alias usages with tokens."""
        for alias_key, alias_name in self.alias_names.items():
            alias_token = self.alias_tokens[alias_key]

            column_pattern = re.compile(
                rf"\b{re.escape(alias_name)}\.(\w+)\b",
                re.IGNORECASE,
            )

            def column_replacer(match: re.Match[str]) -> str:
                return f"{alias_token}.<COL>"

            query = column_pattern.sub(column_replacer, query)

            standalone_pattern = re.compile(
                rf"\b{re.escape(alias_name)}\b(?!\s*\.)",
                re.IGNORECASE,
            )
            query = standalone_pattern.sub(alias_token, query)

        return query

    @staticmethod
    def _postprocess_tokens(tokens: List[str]) -> List[str]:
        result: List[str] = []
        for token in tokens:
            stripped = token.strip()
            if not stripped:
                continue

            if stripped.startswith('<') and stripped.endswith('>'):
                result.append(stripped)
                continue

            if stripped in {',', '.', '(', ')', ';'}:
                result.append(stripped)
                continue

            upper_value = stripped.upper()
            if upper_value in SQL_KEYWORDS or upper_value in SQL_FUNCTIONS:
                result.append(upper_value)
                continue

            if IDENTIFIER_PATTERN.fullmatch(stripped):
                result.append('<COL>')
                continue

            result.append(stripped)

        return result
    
    def _replace_output_aliases(self, query: str) -> str:
        """Replace output aliases declared with AS and track them."""

        def replacer(match: re.Match[str]) -> str:
            alias = match.group(1)
            if alias.upper() in SQL_KEYWORDS:
                return match.group(0)
            self.output_aliases.add(alias.lower())
            return 'AS <COL_OUT>'

        return OUTPUT_ALIAS_PATTERN.sub(replacer, query)

    def _replace_output_alias_references(self, query: str) -> str:
        """Replace subsequent references to output aliases with <COL_OUT>."""
        for alias in sorted(self.output_aliases, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            query = pattern.sub('<COL_OUT>', query)
        return query
    
    @staticmethod
    def _normalize_spacing(query: str) -> str:
        """Normalize spacing around punctuation while preserving placeholder tokens."""

        placeholders: Dict[str, str] = {}

        def protect(match: re.Match[str]) -> str:
            key = f"__PLACEHOLDER_{len(placeholders)}__"
            placeholders[key] = match.group(0)
            return key

        protected = re.sub(r'<[A-Z0-9_]+>', protect, query)

        for operator in ('<>', '!=', '>=', '<='):
            protected = protected.replace(operator, f' {operator} ')

        protected = re.sub(r'([=+\-*/%<>])', r' \1 ', protected)
        protected = re.sub(r'([(),.;])', r' \1 ', protected)

        normalized = re.sub(r'\s+', ' ', protected).strip()

        for key, value in placeholders.items():
            normalized = normalized.replace(key, value)

        return normalized
    
    def tokenize(self, input: str) -> List[str]:
        """Convert a raw SQL query into a list of normalized tokens."""
        self._reset_aliases()
        
        sql = input.strip()
        if not sql:
            return []

        processed = self._replace_literals(sql)
        processed = self._process_update_clauses(processed)
        processed = self._process_insert_clauses(processed)
        processed = self._process_from_clauses(processed)
        processed = self._process_join_clauses(processed)
        processed = self._replace_alias_column_references(processed)
        processed = self._replace_output_aliases(processed)
        processed = self._replace_output_alias_references(processed)
        processed = self._normalize_spacing(processed)

        tokens = [token for token in processed.split(' ') if token]
        return self._postprocess_tokens(tokens)

    def __call__(self, batch: List[str]) -> List[List[str]]:
        for index, text in enumerate(batch):
            batch[index] = self.tokenize(text)
        return batch
    
# # Usage example:
# if __name__ == "__main__":
#     pipeline = PreprocessingPipeline()
    
#     # Test with example SQL query from the image
#     test_query = """DELETE FROM Employees
# WHERE EmployeeID = 2;"""
    
#     print("Original query:")
#     print(test_query)
#     print("\nTokenized result:")
#     tokens = pipeline.tokenize(test_query)
#     print(tokens)
#     print("\nTokens joined:")
#     print(" ".join(tokens))