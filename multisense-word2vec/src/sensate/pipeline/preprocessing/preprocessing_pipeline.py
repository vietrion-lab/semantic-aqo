import re
from typing import List


class PreprocessingPipeline:
    """
    Simplified SQL preprocessing pipeline.
    Converts SQL queries to normalized token sequences following the diagram:
    
    Pure SQL -> Processed SQL -> Tokenized SQL
    
    Example:
    "DELETE FROM Employees WHERE EmployeeID = 2;"
    -> ['DELETE', 'FROM', '<TAB>', '<ALIAS_T1>', 'WHERE', '<COL>', '=', '<NUM>', ';']
    """
    
    # SQL keywords that should remain uppercase
    KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'ON', 'AND', 'OR', 'GROUP', 'BY', 'ORDER',
        'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'CROSS', 'UNION', 'DISTINCT',
        'LIMIT', 'OFFSET', 'AS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ASC', 'DESC',
        'HAVING', 'IN', 'IS', 'NOT', 'NULL', 'INSERT', 'UPDATE', 'DELETE', 'VALUES',
        'SET', 'INTO', 'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'TRUE', 'FALSE', 'BETWEEN',
        'LIKE', 'EXISTS', 'ALL', 'ANY', 'TOP', 'WITH', 'USING'
    }
    
    def __init__(self):
        self.table_num = 1
    
    def tokenize(self, sql: str) -> List[str]:
        """Main tokenization method."""
        if not sql or not sql.strip():
            return []
        
        self.table_num = 1  # Reset for each query
        original_query = sql.strip()
        
        # Step 1: Replace literals
        query = self._replace_literals(original_query)
        
        # Step 2: Extract alias information BEFORE modifying the query
        aliases_info = self._extract_alias_info(original_query)
        
        # Step 3: Handle table references
        query = self._handle_table_references(query)
        
        # Step 4: Handle remaining identifiers and keywords
        tokens = self._tokenize_and_classify(query, aliases_info)
        
        return tokens
    
    def _extract_alias_info(self, query: str) -> list:
        """Extract alias information from the original query."""
        aliases = []
        
        # Find aliases from FROM clause
        from_match = re.search(r'\bFROM\s+\w+\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        if from_match:
            aliases.append(from_match.group(1).lower())
        
        # Find aliases from JOIN clauses
        join_matches = re.finditer(r'\bJOIN\s+\w+\s+([a-zA-Z_]\w*)', query, re.IGNORECASE)
        for match in join_matches:
            aliases.append(match.group(1).lower())
        
        return aliases
    
    def _replace_literals(self, query: str) -> str:
        """Replace literal values with tokens."""
        # Strings
        query = re.sub(r"'[^']*'", '<STR>', query)
        query = re.sub(r'"[^"]*"', '<STR>', query)
        
        # Timestamps and dates
        query = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', query)
        query = re.sub(r'\d{4}-\d{2}-\d{2}', '<DATE>', query)
        
        # Numbers (decimals first)
        query = re.sub(r'\b\d+\.\d+\b', '<NUM>', query)
        query = re.sub(r'\b\d+\b', '<NUM>', query)
        
        # Booleans and NULL
        query = re.sub(r'\bTRUE\b', '<BOOL_TRUE>', query, flags=re.IGNORECASE)
        query = re.sub(r'\bFALSE\b', '<BOOL_FALSE>', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNULL\b', '<NULL>', query, flags=re.IGNORECASE)
        
        return query
    
    def _handle_table_references(self, query: str) -> str:
        """Replace table references. Only add <ALIAS_Tn> when there's actually an alias."""
        
        # DELETE FROM table (no alias expected)
        query = re.sub(
            r'\bDELETE\s+FROM\s+(\w+)',
            r'DELETE FROM <TAB>',
            query, flags=re.IGNORECASE
        )
        
        # UPDATE table (no alias expected)  
        query = re.sub(
            r'\bUPDATE\s+(\w+)',
            r'UPDATE <TAB>',
            query, flags=re.IGNORECASE
        )
        
        # INSERT INTO table (no alias expected)
        query = re.sub(
            r'\bINSERT\s+INTO\s+(\w+)',
            r'INSERT INTO <TAB>',
            query, flags=re.IGNORECASE
        )
        
        # FROM table_name alias (with alias) - e.g., "FROM users u"
        def replace_from_with_alias(match):
            full_match = match.group(0)
            # Check if there are actual aliases (table followed by identifier that's not a keyword)
            if re.search(r'\b\w+\s+[a-zA-Z_]\w*\b', full_match):
                # Count number of table alias pairs
                aliases = re.findall(r'\b\w+\s+([a-zA-Z_]\w*)\b', full_match)
                num_tables = len(aliases)
                if num_tables == 0:
                    # Fallback: count commas + 1
                    num_tables = full_match.count(',') + 1
                
                result = 'FROM'
                for i in range(num_tables):
                    if i > 0:
                        result += ' ,'
                    result += f' <TAB> <ALIAS_T{self._next_table()}>'
                return result
            else:
                # No aliases, just table names
                return 'FROM <TAB>'
        
        query = re.sub(
            r'\bFROM\s+\w+(?:\s+[a-zA-Z_]\w*)?(?:\s*,\s*\w+(?:\s+[a-zA-Z_]\w*)?)*',
            replace_from_with_alias, query, flags=re.IGNORECASE
        )
        
        # JOIN table alias (with alias) - e.g., "JOIN orders o"
        def replace_join_with_alias(match):
            join_type = match.group(1)
            full_match = match.group(0)
            prefix = f'{join_type} ' if join_type else ''
            
            # Check if there's an alias after the table name
            if re.search(r'\bJOIN\s+\w+\s+[a-zA-Z_]\w*', full_match, re.IGNORECASE):
                return f'{prefix}JOIN <TAB> <ALIAS_T{self._next_table()}>'
            else:
                return f'{prefix}JOIN <TAB>'
        
        query = re.sub(
            r'\b(?:(INNER|LEFT|RIGHT|FULL|OUTER|CROSS)\s+)?JOIN\s+\w+(?:\s+[a-zA-Z_]\w*)?',
            replace_join_with_alias, query, flags=re.IGNORECASE
        )
        
        return query
    
    def _tokenize_and_classify(self, query: str, aliases_info: list) -> List[str]:
        """Split query and classify tokens."""
        
        # Handle alias.column references BEFORE general processing
        for i, alias in enumerate(aliases_info, 1):
            # Replace alias.column with <ALIAS_Tn>.<COL>
            pattern = rf'\b{re.escape(alias)}\.(\w+)\b'
            query = re.sub(pattern, f'<ALIAS_T{i}>.<COL>', query, flags=re.IGNORECASE)
        
        # Handle AS output_alias
        query = re.sub(
            r'\bAS\s+(\w+)',
            lambda m: 'AS <COL_OUT>' if m.group(1).upper() not in self.KEYWORDS else m.group(0),
            query, flags=re.IGNORECASE
        )
        
        # Split into tokens first
        tokens = re.findall(r'<[^>]+>|[a-zA-Z_]\w*|[=<>!]+|[(),;.]|\S', query)
        
        # Classify each token
        result = []
        for token in tokens:
            if not token or token.isspace():
                continue
            
            # Already processed tokens
            if token.startswith('<') and token.endswith('>'):
                result.append(token)
            # Keywords
            elif token.upper() in self.KEYWORDS:
                result.append(token.upper())
            # Operators and punctuation
            elif token in '=<>!(),.;':
                result.append(token)
            elif re.match(r'[=<>!]+', token):
                result.append(token)
            # Check if it's one of our known aliases - if so, replace with appropriate <ALIAS_Tn>
            elif token.lower() in aliases_info:
                alias_index = aliases_info.index(token.lower()) + 1
                result.append(f'<ALIAS_T{alias_index}>')
            # Everything else becomes <COL>
            elif re.match(r'[a-zA-Z_]\w*', token):
                result.append('<COL>')
            else:
                result.append(token)
        
        return result
    
    def _next_table(self) -> int:
        """Get next table alias number."""
        num = self.table_num
        self.table_num += 1
        return num
    
    def __call__(self, batch: List[str]) -> List[List[str]]:
        """Process a batch of queries."""
        return [self.tokenize(query) for query in batch]

# if __name__ == "__main__":
#     pipeline = PreprocessingPipeline()
    
#     # Test cases matching the diagram
#     test_queries = [
#         "DELETE FROM Employees WHERE EmployeeID = 2;",
#         "SELECT u.name, SUM(o.amount) AS total FROM users u JOIN orders o ON u.user_id = o.id WHERE u.age >= 21",
#         "UPDATE products SET price = 99.99 WHERE category = 'electronics'",
#         "INSERT INTO customers VALUES (1, 'John Doe', '2023-01-15')",
#         """SELECT u.name, SUM(o.amount) AS total
# FROM users u
# JOIN orders o ON o.user_id = u.id
# WHERE u.age >= 21 AND o.status IN ('paid','shipped')
# GROUP BY u.name
# ORDER BY total DESC"""
#     ]
    
#     print("=== Clean SQL Preprocessing Pipeline ===\n")
    
#     for i, query in enumerate(test_queries, 1):
#         print(f"Test {i}: {query}")
#         tokens = pipeline.tokenize(query)
#         print(f"Result:  {' '.join(tokens)}")
#         print(f"Tokens:  {tokens}")
#         print("-" * 70)