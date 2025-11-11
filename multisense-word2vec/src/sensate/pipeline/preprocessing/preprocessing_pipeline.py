import re
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import sqlglot
from sqlglot import exp
from sqlglot.expressions import Expression


class PreprocessingPipeline:
    """
    Robust SQL preprocessing pipeline using sqlglot.
    Converts SQL queries to normalized token sequences following strict rules:
    
    Pure SQL -> Processed SQL -> Tokenized SQL
    
    Rules:
    - Tables: <TAB> with <ALIAS_Ti> (i = 1-based encounter order)
    - Columns: <COL> (qualified: <ALIAS_Ti>.<COL>)
    - Output aliases: <COL_OUT> (only when AS exists)
    - Literals: <NUM>, <STR>, <DATE>, <TIMESTAMP>, <BOOL_TRUE>, <BOOL_FALSE>, <NULL>
    - Functions/operators/keywords: preserved
    """
    
    def __init__(self):
        self.table_counter = 0
        self.alias_map: Dict[str, str] = {}  # real_alias -> <ALIAS_Ti>
        self.table_map: Dict[str, Tuple[str, Optional[str], bool]] = {}  # <ALIAS_Ti> -> (real_table, real_alias, is_temp)
        self.dialect = None
    
    def _detect_dialect(self, sql: str) -> str:
        """Detect SQL dialect based on syntax hints."""
        # Quick heuristics for common dialects
        sql_lower = sql.lower()
        
        if '#' in sql or '##' in sql:
            return 'tsql'
        if 'isnull(' in sql_lower:
            return 'tsql'
        if 'ifnull(' in sql_lower:
            return 'mysql'
        if 'nvl(' in sql_lower:
            return 'oracle'
            
        # Try parsing with common dialects (suppress warnings)
        for dialect in ['postgres', 'mysql', 'tsql', 'sqlite']:
            try:
                sqlglot.parse_one(sql, read=dialect, error_level=None)
                return dialect
            except:
                continue
        
        # Default to postgres as it's the most permissive
        return 'postgres'
    
    def _next_alias(self) -> str:
        """Generate next table alias token."""
        self.table_counter += 1
        return f'<ALIAS_T{self.table_counter}>'
    
    def _register_table(self, table_name: str, alias: Optional[str] = None) -> str:
        """Register a table and return its alias token."""
        # Check if it's a temp table
        is_temp = table_name.startswith('#') or table_name.startswith('##')
        
        # If alias exists, check if already registered
        if alias and alias in self.alias_map:
            return self.alias_map[alias]
        
        # Generate new alias token
        alias_token = self._next_alias()
        
        if alias:
            self.alias_map[alias] = alias_token
        else:
            # No explicit alias - use table name as key
            self.alias_map[table_name] = alias_token
        
        # Store mapping
        self.table_map[alias_token] = (table_name, alias, is_temp)
        return alias_token
    
    def _extract_alias_string(self, alias) -> Optional[str]:
        """Extract alias string from alias node."""
        if not alias:
            return None
        if isinstance(alias, exp.TableAlias):
            if hasattr(alias, 'this'):
                if isinstance(alias.this, exp.Identifier):
                    return alias.this.this if hasattr(alias.this, 'this') else str(alias.this)
                return str(alias.this)
            return str(alias)
        elif isinstance(alias, exp.Identifier):
            return alias.this if hasattr(alias, 'this') else str(alias)
        return str(alias)
    
    def _collect_table_aliases(self, node: Expression):
        """First pass: collect all table aliases from FROM/JOIN clauses."""
        if isinstance(node, exp.Table):
            table_name = node.name if isinstance(node.name, str) else (node.name.this if hasattr(node.name, 'this') else str(node.name))
            alias = node.alias if hasattr(node, 'alias') and node.alias else None
            
            # Extract alias string
            alias_str = self._extract_alias_string(alias)
            
            # Register table and alias
            if alias_str and alias_str not in self.alias_map:
                self._register_table(table_name, alias_str)
            elif not alias_str and table_name not in self.alias_map:
                self._register_table(table_name, None)
        
        # Recursively collect from children
        for child in node.iter_expressions():
            self._collect_table_aliases(child)
    
    def _transform_node(self, node: Expression) -> Expression:
        """Transform AST node according to tokenization rules."""
        
        # Handle Table references in FROM/JOIN
        if isinstance(node, exp.Table):
            table_name = node.name if isinstance(node.name, str) else (node.name.this if hasattr(node.name, 'this') else str(node.name))
            alias = node.alias if hasattr(node, 'alias') and node.alias else None
            
            # Extract alias string
            alias_str = self._extract_alias_string(alias)
            
            # Get the alias token (should already be registered)
            if alias_str and alias_str in self.alias_map:
                alias_token = self.alias_map[alias_str]
            elif table_name in self.alias_map:
                alias_token = self.alias_map[table_name]
            else:
                # Fallback: register now
                alias_token = self._register_table(table_name, alias_str)
            
            # Replace table name with <TAB>
            node.set('this', exp.Identifier(this='<TAB>'))
            
            # Set alias to our token
            node.set('alias', exp.TableAlias(this=exp.Identifier(this=alias_token)))
        
        # Handle Column references
        elif isinstance(node, exp.Column):
            table_ref = node.table if hasattr(node, 'table') and node.table else None
            
            if table_ref:
                # Qualified column: table.column -> <ALIAS_Ti>.<COL>
                if isinstance(table_ref, exp.Identifier):
                    table_str = table_ref.this if hasattr(table_ref, 'this') else str(table_ref)
                else:
                    table_str = str(table_ref)
                    
                if table_str in self.alias_map:
                    alias_token = self.alias_map[table_str]
                    node.set('table', exp.Identifier(this=alias_token))
                node.set('this', exp.Identifier(this='<COL>'))
            else:
                # Unqualified column: column -> <COL>
                node.set('this', exp.Identifier(this='<COL>'))
        
        # Handle Alias in SELECT (output columns)
        elif isinstance(node, exp.Alias):
            # Only replace if it's in SELECT clause (column output alias)
            parent = node.parent
            if parent and isinstance(parent, exp.Select):
                # Replace alias with <COL_OUT>
                node.set('alias', exp.Identifier(this='<COL_OUT>'))
        
        # Handle Literals
        elif isinstance(node, exp.Literal):
            if node.is_string:
                node.set('this', '<STR>')
            elif node.is_int or node.is_number:
                node.set('this', '<NUM>')
        
        elif isinstance(node, (exp.Boolean)):
            if str(node).upper() in ('TRUE', '1'):
                return exp.Identifier(this='<BOOL_TRUE>')
            else:
                return exp.Identifier(this='<BOOL_FALSE>')
        
        elif isinstance(node, exp.Null):
            return exp.Identifier(this='<NULL>')
        
        elif isinstance(node, exp.DataType):
            # Handle DATE/TIMESTAMP types
            type_name = str(node.this).upper() if hasattr(node, 'this') else str(node).upper()
            if 'TIMESTAMP' in type_name:
                return exp.Identifier(this='<TIMESTAMP>')
            elif 'DATE' in type_name:
                return exp.Identifier(this='<DATE>')
        
        # Recursively transform children
        for child in node.iter_expressions():
            self._transform_node(child)
        
        return node
    
    def _extract_mapping(self) -> List[str]:
        """Extract table and alias mapping for pretty printing."""
        lines = []
        for alias_token in sorted(self.table_map.keys(), key=lambda x: int(x.split('T')[1].rstrip('>'))):
            table_name, real_alias, is_temp = self.table_map[alias_token]
            temp_marker = " (temp)" if is_temp else ""
            if real_alias:
                lines.append(f"<TAB> {alias_token} -> {table_name} {real_alias}{temp_marker}")
            else:
                lines.append(f"<TAB> {alias_token} -> {table_name}{temp_marker}")
        return lines
    
    def tokenize(self, sql: str) -> Tuple[str, List[str], List[str]]:
        """
        Main tokenization method.
        
        Returns:
            - processed_sql: The tokenized SQL string
            - tokens: List of tokens
            - mapping: List of table/alias mapping strings
        """
        if not sql or not sql.strip():
            return "", [], []
        
        # Reset state
        self.table_counter = 0
        self.alias_map = {}
        self.table_map = {}
        
        # Detect dialect
        self.dialect = self._detect_dialect(sql)
        
        try:
            # Parse SQL to AST (suppress warnings)
            ast = sqlglot.parse_one(sql, read=self.dialect, error_level=None)
            
            if ast is None:
                # Parsing failed, use fallback
                return self._fallback_tokenize(sql)
            
            # First pass: collect all table aliases
            self._collect_table_aliases(ast)
            
            # Second pass: transform AST
            self._transform_node(ast)
            
            # Generate processed SQL
            processed_sql = ast.sql(dialect=self.dialect, pretty=True)
            
            # Extract mapping
            mapping = self._extract_mapping()
            
            # Tokenize the processed SQL
            tokens = self._sql_to_tokens(processed_sql)
            
            return processed_sql, tokens, mapping
        
        except Exception:
            # Fallback to simple tokenization if parsing fails (silently)
            return self._fallback_tokenize(sql)
    
    def _fallback_tokenize(self, sql: str) -> Tuple[str, List[str], List[str]]:
        """Simple fallback tokenization when sqlglot fails."""
        # Just split on whitespace and basic punctuation
        tokens = self._sql_to_tokens(sql)
        return sql, tokens, []
    
    def _sql_to_tokens(self, sql: str) -> List[str]:
        """Convert SQL string to list of tokens."""
        # Tokenization that preserves special tokens like <ALIAS_T1>
        tokens = []
        current_token = ""
        in_special_token = False
        
        for char in sql:
            if char == '<':
                # Start of a special token
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                in_special_token = True
                current_token = char
            elif char == '>' and in_special_token:
                # End of a special token
                current_token += char
                tokens.append(current_token)
                current_token = ""
                in_special_token = False
            elif in_special_token:
                # Inside a special token, keep accumulating
                current_token += char
            elif char in ' \t\n\r':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '(),;.=!':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return [t for t in tokens if t.strip()]
    
    def __call__(self, batch: List[str]) -> List[List[str]]:
        """
        Process a batch of queries.
        
        Returns list of token lists (matching old output format).
        """
        results = []
        for query in tqdm(batch, desc="Preprocessing queries", unit="query"):
            processed_sql, tokens, mapping = self.tokenize(query)
            results.append(tokens)
        return results


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    
    # Test cases covering various SQL patterns
    test_queries = [
        "DELETE FROM Employees WHERE EmployeeID = 2;",
        "SELECT u.name, SUM(o.amount) AS total FROM users u JOIN orders o ON u.user_id = o.id WHERE u.age >= 21",
        "UPDATE products SET price = 99.99 WHERE category = 'electronics'",
        "INSERT INTO customers VALUES (1, 'John Doe', '2023-01-15')",
        """SELECT u.name, SUM(o.amount) AS total
FROM users u
JOIN orders o ON o.user_id = u.id
WHERE u.age >= 21 AND o.status IN ('paid','shipped')
GROUP BY u.name
ORDER BY total DESC""",
        "SELECT * FROM #temp_table WHERE id = 1",  # T-SQL temp table
        "SELECT a.id, b.name FROM table1 a, table2 b WHERE a.id = b.id",  # Comma join
        "WITH cte AS (SELECT id FROM users) SELECT * FROM cte",  # CTE
    ]
    
    print("=== Robust SQL Preprocessing Pipeline ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}:")
        print(f"Original: {query[:80]}...")
        processed_sql, tokens, mapping = pipeline.tokenize(query)
        print(f"\nProcessed SQL:")
        print(processed_sql)
        print(f"\nMapping:")
        for line in mapping:
            print(f"  {line}")
        print(f"\nTokens: {tokens[:20]}...")  # Show first 20 tokens
        print("-" * 80)
