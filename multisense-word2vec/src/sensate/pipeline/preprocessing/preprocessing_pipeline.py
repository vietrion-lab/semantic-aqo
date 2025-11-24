import re
from typing import List, Dict, Tuple, Optional


class PreprocessingPipeline:
    """
    SQL preprocessing pipeline that converts pure SQL to processed SQL with special tokens.
    
    Supports various RDBMS dialects and converts:
    - Table names -> <TAB>
    - Column names -> <COL>
    - Table aliases -> <ALIAS_T1>, <ALIAS_T2>, etc.
    - Output aliases -> <COL_OUT>
    - Literals -> <NUM>, <STR>, <DATE>, <TIMESTAMP>, <BOOL_TRUE>, <BOOL_FALSE>, <NULL>
    
    Example:
    "SELECT u.name, SUM(o.amount) AS total FROM users u JOIN orders o ON o.user_id = u.id"
    -> "SELECT <ALIAS_T1>.<COL>, SUM(<ALIAS_T2>.<COL>) AS <COL_OUT> FROM <TAB> <ALIAS_T1> JOIN <TAB> <ALIAS_T2> ON <ALIAS_T2>.<COL> = <ALIAS_T1>.<COL>"
    """
    
    # SQL keywords that should remain uppercase
    KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'GROUP', 'BY', 'ORDER',
        'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS', 'UNION', 'DISTINCT',
        'LIMIT', 'OFFSET', 'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ASC', 'DESC',
        'HAVING', 'IN', 'IS', 'NOT', 'NULL', 'INSERT', 'UPDATE', 'DELETE', 'VALUES',
        'SET', 'INTO', 'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'TRUE', 'FALSE', 'BETWEEN',
        'LIKE', 'EXISTS', 'ALL', 'ANY', 'TOP', 'WITH', 'USING', 'CREATE', 'TABLE',
        'DROP', 'ALTER', 'INDEX', 'VIEW', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
        'CONSTRAINT', 'UNIQUE', 'CHECK', 'DEFAULT', 'AUTO_INCREMENT', 'CASCADE',
        'DBO'  # Common schema prefix
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for processing a new query."""
        self.table_aliases: Dict[str, int] = {}  # Maps alias name -> table number
        self.alias_counter = 0
        self.table_names: List[str] = []  # Track table names in order
        self.output_aliases: List[str] = []  # Track output column aliases from SELECT ... AS
    
    def tokenize(self, sql: str) -> str:
        """
        Main method that converts pure SQL to processed SQL string.
        Returns a processed SQL string with special tokens.
        """
        if not sql or not sql.strip():
            return ""
        
        self.reset()
        query = sql.strip()
        
        # Step 1: Extract table aliases from the query FIRST (before any modifications)
        self._extract_table_aliases(query)
        
        # Step 2: Extract output aliases from SELECT ... AS ...
        self._extract_output_aliases(query)
        
        # Step 3: Tokenize the query - split into words, operators, etc.
        tokens = self._tokenize_query(query)
        
        # Step 4: Process tokens and classify them
        processed_tokens = self._process_tokens(tokens)
        
        # Step 5: Join back into a string
        return ' '.join(processed_tokens)
    
    def _extract_table_aliases(self, query: str):
        """
        Extract all table aliases from the query and assign them numbers.
        Handles: FROM table alias, JOIN table alias, etc.
        """
        # Match FROM clause with aliases: FROM table_name alias or FROM table_name AS alias
        from_pattern = r'\bFROM\s+([#\w]+)\s+(?:AS\s+)?([a-zA-Z_]\w*)\b'
        for match in re.finditer(from_pattern, query, re.IGNORECASE):
            table_name = match.group(1)
            alias = match.group(2)
            if alias.upper() not in self.KEYWORDS:
                self._register_alias(alias)
                self.table_names.append(table_name)
        
        # Match JOIN clauses with aliases: JOIN table_name alias or JOIN table_name AS alias
        join_pattern = r'\bJOIN\s+([#\w]+)\s+(?:AS\s+)?([a-zA-Z_]\w*)\b'
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            table_name = match.group(1)
            alias = match.group(2)
            if alias.upper() not in self.KEYWORDS:
                self._register_alias(alias)
                self.table_names.append(table_name)
    
    def _register_alias(self, alias: str):
        """Register an alias and assign it a number if not already registered."""
        alias_lower = alias.lower()
        if alias_lower not in self.table_aliases:
            self.alias_counter += 1
            self.table_aliases[alias_lower] = self.alias_counter
    
    def _extract_output_aliases(self, query: str):
        """
        Extract output aliases from SELECT clause (AS alias_name).
        These appear after AS in the SELECT portion of the query.
        """
        # Match AS alias_name pattern, but only in SELECT clause (before FROM)
        # Simple approach: find everything between SELECT and FROM
        select_match = re.search(r'\bSELECT\b(.*?)\bFROM\b', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Find all AS alias patterns, excluding bracketed ones for now
            # Pattern: AS followed by an identifier (not a keyword)
            as_pattern = r'\bAS\s+([a-zA-Z_]\w*)\b'
            for match in re.finditer(as_pattern, select_clause, re.IGNORECASE):
                alias = match.group(1)
                if alias.upper() not in self.KEYWORDS:
                    self.output_aliases.append(alias.lower())
            
            # Also handle bracketed aliases [name]
            bracket_pattern = r'\bAS\s+\[([^\]]+)\]'
            for match in re.finditer(bracket_pattern, select_clause, re.IGNORECASE):
                alias = match.group(1)
                self.output_aliases.append(alias.lower())
    
    def _tokenize_query(self, query: str) -> List[str]:
        """
        Tokenize SQL query into a list of tokens.
        Preserves strings, numbers, operators, identifiers, etc.
        """
        # Regex pattern to match:
        # - Strings in quotes: '...' or "..."
        # - Numbers: integers and decimals
        # - Operators: =, !=, <=, >=, <>, etc.
        # - Identifiers: table names, column names, keywords
        # - Punctuation: (, ), ,, ;, .
        # - Bracketed identifiers: [name]
        pattern = r"""
            '(?:[^']|'')*'|           # Single-quoted strings
            "(?:[^"]|"")*"|           # Double-quoted strings
            \d+\.?\d*|                # Numbers (int and decimal)
            <>|!=|<=|>=|              # Multi-char operators
            [a-zA-Z_#][\w]*|          # Identifiers (including # for temp tables)
            \[[^\]]+\]|               # Bracketed identifiers [name]
            [(),;.=<>+\-*/]           # Single-char operators and punctuation
        """
        tokens = re.findall(pattern, query, re.VERBOSE | re.IGNORECASE)
        return tokens
    
    def _process_tokens(self, tokens: List[str]) -> List[str]:
        """
        Process tokens and classify them as keywords, tables, columns, literals, etc.
        """
        result = []
        i = 0
        in_select_clause = False
        after_as_in_select = False
        after_as_in_from = False
        expect_table = False
        expect_table_alias = False
        
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.upper()
            
            # Skip whitespace
            if not token or token.isspace():
                i += 1
                continue
            
            # Handle literals first
            if self._is_string_literal(token):
                result.append('<STR>')
                i += 1
                continue
            
            if self._is_number(token):
                result.append('<NUM>')
                i += 1
                continue
            
            # Handle keywords
            if token_upper == 'SELECT':
                result.append('SELECT')
                in_select_clause = True
                i += 1
                continue
            
            if token_upper == 'FROM':
                result.append('FROM')
                in_select_clause = False
                expect_table = True
                i += 1
                continue
            
            if token_upper == 'JOIN' or token_upper in ('INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS'):
                result.append(token_upper)
                if token_upper == 'JOIN':
                    expect_table = True
                i += 1
                continue
            
            if token_upper == 'AS':
                result.append('AS')
                if in_select_clause:
                    after_as_in_select = True
                else:
                    after_as_in_from = True
                i += 1
                continue
            
            if token_upper in self.KEYWORDS:
                result.append(token_upper)
                after_as_in_select = False
                after_as_in_from = False
                if token_upper in ('WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
                    in_select_clause = False
                i += 1
                continue
            
            # Handle qualified column references: alias.column
            if i + 2 < len(tokens) and tokens[i + 1] == '.':
                alias_or_schema = token.lower()
                column = tokens[i + 2]
                
                # Check if it's a known table alias
                if alias_or_schema in self.table_aliases:
                    alias_num = self.table_aliases[alias_or_schema]
                    result.append(f'<ALIAS_T{alias_num}>')
                    result.append('.')
                    result.append('<COL>')
                    i += 3
                    after_as_in_select = False
                    after_as_in_from = False
                    continue
                else:
                    # Might be schema.table or schema.function - treat as column for now
                    result.append('<COL>')
                    result.append('.')
                    result.append('<COL>')
                    i += 3
                    after_as_in_select = False
                    after_as_in_from = False
                    continue
            
            # Handle table names (after FROM or JOIN)
            if expect_table:
                # This is a table name
                result.append('<TAB>')
                expect_table = False
                expect_table_alias = True
                i += 1
                continue
            
            # Handle table aliases (after table name in FROM/JOIN clause)
            if expect_table_alias:
                # This could be a table alias or the next keyword
                if token_upper in self.KEYWORDS and token_upper != 'AS':
                    # Not an alias, it's a keyword - no alias provided
                    result.append(token_upper)
                    expect_table_alias = False
                    if token_upper == 'JOIN':
                        expect_table = True
                    elif token_upper in ('WHERE', 'GROUP', 'ORDER', 'HAVING', 'LIMIT'):
                        in_select_clause = False
                elif token.lower() in self.table_aliases:
                    # This is an alias
                    alias_num = self.table_aliases[token.lower()]
                    result.append(f'<ALIAS_T{alias_num}>')
                    expect_table_alias = False
                    after_as_in_from = False
                    i += 1
                    continue
                else:
                    # Unknown identifier after table - shouldn't happen if aliases are extracted correctly
                    # Treat as column
                    result.append('<COL>')
                    expect_table_alias = False
                i += 1
                continue
            
            # Handle output aliases (after AS in SELECT clause)
            if after_as_in_select:
                if token.startswith('[') and token.endswith(']'):
                    result.append('<COL_OUT>')
                elif token_upper not in self.KEYWORDS:
                    result.append('<COL_OUT>')
                else:
                    # It's a keyword, not an alias
                    result.append(token_upper)
                after_as_in_select = False
                i += 1
                continue
            
            # Handle table aliases after AS in FROM/JOIN clause
            if after_as_in_from:
                if token.lower() in self.table_aliases:
                    alias_num = self.table_aliases[token.lower()]
                    result.append(f'<ALIAS_T{alias_num}>')
                else:
                    result.append('<COL>')
                after_as_in_from = False
                i += 1
                continue
            
            # Handle bracketed identifiers
            if token.startswith('[') and token.endswith(']'):
                if after_as_in_select:
                    result.append('<COL_OUT>')
                    after_as_in_select = False
                else:
                    result.append('<COL>')
                i += 1
                continue
            
            # Handle function calls
            if i + 1 < len(tokens) and tokens[i + 1] == '(':
                # This is a function name
                result.append(token_upper)
                i += 1
                continue
            
            # Handle punctuation
            if token in '(),;.=<>!+-*/':
                result.append(token)
                i += 1
                continue
            
            # Handle standalone aliases (when they appear without dot notation)
            if token.lower() in self.table_aliases:
                alias_num = self.table_aliases[token.lower()]
                result.append(f'<ALIAS_T{alias_num}>')
                i += 1
                continue
            
            # Check if it's an output alias (from SELECT ... AS ...)
            # These can appear in ORDER BY, GROUP BY, HAVING clauses
            if token.lower() in self.output_aliases:
                result.append('<COL_OUT>')
                i += 1
                continue
            
            # Everything else is a column
            result.append('<COL>')
            i += 1
        
        return result
    
    def _is_string_literal(self, token: str) -> bool:
        """Check if token is a string literal."""
        return (token.startswith("'") and token.endswith("'")) or \
               (token.startswith('"') and token.endswith('"'))
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def __call__(self, query_or_batch):
        """
        Process a single query or a batch of queries.
        
        Args:
            query_or_batch: Either a single SQL query string or a list of SQL query strings
            
        Returns:
            List of tokens for a single query, or list of token lists for a batch
        """
        if isinstance(query_or_batch, str):
            # Return list of tokens
            processed_string = self.tokenize(query_or_batch)
            return processed_string.split()
        elif isinstance(query_or_batch, (list, tuple)):
            # Return list of token lists
            return [self.tokenize(query).split() for query in query_or_batch]
        else:
            # Handle other iterables (like HuggingFace dataset columns)
            try:
                return [self.tokenize(str(query)).split() for query in query_or_batch]
            except TypeError:
                raise TypeError(f"Input must be a string or iterable of strings, got {type(query_or_batch)}")
