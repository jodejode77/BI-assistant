"""
SQL Generator with dialect-specific features and retry logic
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from core.database_adapter import DatabaseAdapter, DatabaseDialect
from core.dynamic_semantic_layer import DynamicSemanticLayer
from core.enhanced_rag_system import EnhancedRAGSystem
from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class SQLGenerationStrategy(Enum):
    """SQL generation strategies"""
    DIRECT = "direct"  # Direct LLM generation
    RAG_ENHANCED = "rag_enhanced"  # With RAG context
    SEMANTIC_GUIDED = "semantic_guided"  # With semantic layer guidance
    FULL_CONTEXT = "full_context"  # With all available context


@dataclass
class SQLGenerationContext:
    """Context for SQL generation"""
    question: str
    schema_context: str
    semantic_context: Dict[str, Any]
    similar_queries: List[Dict[str, str]]
    dialect_features: Dict[str, Any]
    strategy: SQLGenerationStrategy
    attempt_number: int = 1
    previous_errors: List[str] = None
    
    def __post_init__(self):
        if self.previous_errors is None:
            self.previous_errors = []


@dataclass
class GeneratedSQL:
    """Generated SQL with metadata"""
    query: str
    confidence: float
    strategy_used: SQLGenerationStrategy
    tables_used: List[str]
    columns_used: List[str]
    aggregations: List[str]
    filters: List[str]
    dialect_specific: bool
    generation_time: float
    retry_count: int = 0


class DialectSpecificSQLGenerator:
    """
    SQL Generator with dialect-specific optimizations and retry logic
    """
    
    def __init__(
        self,
        database_adapter: DatabaseAdapter,
        semantic_layer: DynamicSemanticLayer,
        rag_system: EnhancedRAGSystem,
        llm_manager: LLMManager,
        max_retries: int = 3
    ):
        """
        Initialize SQL Generator
        
        Args:
            database_adapter: Database adapter
            semantic_layer: Dynamic semantic layer
            rag_system: Enhanced RAG system
            llm_manager: LLM manager
            max_retries: Maximum number of retry attempts
        """
        self.db_adapter = database_adapter
        self.semantic_layer = semantic_layer
        self.rag_system = rag_system
        self.llm = llm_manager
        self.max_retries = max_retries
        
        # Get dialect-specific features
        self.dialect_features = database_adapter.get_sql_dialect_features()
        self.dialect = database_adapter.dialect
        
        logger.info(f"Initialized SQL Generator for {self.dialect.value}")
    
    def generate_sql(
        self,
        question: str,
        strategy: SQLGenerationStrategy = SQLGenerationStrategy.FULL_CONTEXT
    ) -> GeneratedSQL:
        """
        Generate SQL with retry logic
        
        Args:
            question: Natural language question
            strategy: Generation strategy to use
            
        Returns:
            GeneratedSQL object
        """
        import time
        start_time = time.time()
        
        logger.info(f"[SQL_GEN] Generating SQL for: '{question[:100]}...' with strategy: {strategy.value}")
        
        # Build generation context
        context = self._build_generation_context(question, strategy)
        
        # Try generation with retries
        retry_count = 0
        last_error = None
        generated_sql = None
        
        while retry_count <= self.max_retries:
            try:
                logger.info(f"[SQL_GEN] Attempt {retry_count + 1}/{self.max_retries + 1}")
                
                # Generate SQL
                sql = self._generate_with_strategy(context)
                
                # Post-process SQL
                sql = self._post_process_sql(sql)
                
                # Validate SQL
                is_valid, error = self.db_adapter.validate_query(sql)
                
                # If validation failed due to type error, try to fix it
                if not is_valid and error and ('function' in error.lower() and 'text' in error.lower() or 'does not exist' in error.lower()):
                    logger.warning(f"[SQL_GEN] Type error detected, attempting automatic fix: {error}")
                    # Try to fix numeric aggregations
                    sql_fixed = self._fix_numeric_aggregations(sql)
                    if sql_fixed != sql:
                        logger.info(f"[SQL_GEN] Applied automatic CAST fix")
                        sql = sql_fixed
                        # Re-validate
                        is_valid, error = self.db_adapter.validate_query(sql)
                
                if is_valid:
                    # Extract metadata
                    metadata = self._extract_query_metadata(sql)
                    
                    generation_time = time.time() - start_time
                    
                    generated_sql = GeneratedSQL(
                        query=sql,
                        confidence=self._calculate_confidence(context, metadata),
                        strategy_used=strategy,
                        tables_used=metadata['tables'],
                        columns_used=metadata['columns'],
                        aggregations=metadata['aggregations'],
                        filters=metadata['filters'],
                        dialect_specific=True,
                        generation_time=generation_time,
                        retry_count=retry_count
                    )
                    
                    logger.info(f"[SQL_GEN] Successfully generated SQL in {generation_time:.2f}s after {retry_count} retries")
                    break
                else:
                    last_error = error
                    context.previous_errors.append(error)
                    context.attempt_number += 1
                    retry_count += 1
                    
                    logger.warning(f"[SQL_GEN] Validation failed: {error}")
                    
                    if retry_count <= self.max_retries:
                        # Adjust strategy for retry
                        context = self._adjust_context_for_retry(context, error)
                    
            except Exception as e:
                last_error = str(e)
                context.previous_errors.append(str(e))
                retry_count += 1
                
                logger.error(f"[SQL_GEN] Generation error: {e}")
                
                if retry_count > self.max_retries:
                    break
        
        if generated_sql is None:
            # Failed after all retries
            logger.error(f"[SQL_GEN] Failed to generate valid SQL after {self.max_retries} attempts")
            raise Exception(f"SQL generation failed after {self.max_retries} attempts. Last error: {last_error}")
        
        return generated_sql
    
    def _build_generation_context(
        self,
        question: str,
        strategy: SQLGenerationStrategy
    ) -> SQLGenerationContext:
        """Build context for SQL generation"""
        logger.debug(f"[SQL_GEN] Building context with strategy: {strategy.value}")
        
        # Get semantic context
        semantic_context = self.semantic_layer.enhance_query(question)
        
        # Get schema context from RAG
        schema_context = ""
        similar_queries = []
        
        if strategy in [SQLGenerationStrategy.RAG_ENHANCED, SQLGenerationStrategy.FULL_CONTEXT]:
            # Get relevant schema with structured format for schema-guided reasoning
            suggested_tables = semantic_context.get('suggested_tables', [])
            schema_context = self.rag_system.get_schema_context(
                question, 
                suggested_tables,
                structured=(strategy == SQLGenerationStrategy.FULL_CONTEXT)
            )
            
            # Get similar queries
            similar_queries = self.rag_system.get_similar_queries(question)
        
        return SQLGenerationContext(
            question=question,
            schema_context=schema_context,
            semantic_context=semantic_context,
            similar_queries=similar_queries,
            dialect_features=self.dialect_features,
            strategy=strategy
        )
    
    def _generate_with_strategy(self, context: SQLGenerationContext) -> str:
        """Generate SQL based on strategy"""
        logger.debug(f"[SQL_GEN] Generating with strategy: {context.strategy.value}")
        
        # Build prompt based on strategy
        prompt = self._build_generation_prompt(context)
        
        # Call LLM
        if context.strategy == SQLGenerationStrategy.DIRECT:
            # Simple generation, but still provide table list
            tables = self.db_adapter.get_table_list()
            minimal_schema = f"Available tables: {', '.join(tables)}\n\nUse only these table names in your SQL query."
            sql = self.llm.generate_sql(
                question=context.question,
                schema_context=minimal_schema,
                examples=None
            )
        else:
            # Enhanced generation with RAG context and schema-guided reasoning
            sql = self.llm.generate_sql(
                question=context.question,
                schema_context=context.schema_context,
                examples=context.similar_queries[:3] if context.similar_queries else None,
                use_schema_reasoning=(context.strategy in [
                    SQLGenerationStrategy.SEMANTIC_GUIDED,
                    SQLGenerationStrategy.FULL_CONTEXT
                ])
            )
        
        return sql
    
    def _build_generation_prompt(self, context: SQLGenerationContext) -> str:
        """Build detailed prompt for SQL generation"""
        prompt_parts = []
        
        # Base question
        prompt_parts.append(f"Generate SQL query for: {context.question}")
        
        # Add dialect information
        prompt_parts.append(f"\nDatabase: {self.dialect.value.upper()}")
        
        # Add dialect-specific features
        if self.dialect_features:
            features = []
            if self.dialect_features.get('limit_syntax'):
                features.append(f"Use {self.dialect_features['limit_syntax']} for limiting results")
            if self.dialect_features.get('null_handling'):
                features.append(f"Use {self.dialect_features['null_handling']} for null handling")
            if features:
                prompt_parts.append("Dialect features: " + ", ".join(features))
        
        # Add semantic context
        if context.semantic_context:
            if context.semantic_context.get('detected_entities'):
                prompt_parts.append(f"Detected entities: {', '.join(context.semantic_context['detected_entities'])}")
            
            if context.semantic_context.get('suggested_tables'):
                prompt_parts.append(f"Relevant tables: {', '.join(context.semantic_context['suggested_tables'])}")
            
            if context.semantic_context.get('aggregations'):
                prompt_parts.append(f"Suggested aggregations: {', '.join(context.semantic_context['aggregations'])}")
        
        # Add schema context
        if context.schema_context:
            prompt_parts.append(f"\nSchema Context:\n{context.schema_context}")
        
        # Add examples
        if context.similar_queries:
            examples = []
            for ex in context.similar_queries[:2]:
                examples.append(f"Q: {ex['question']}\nSQL: {ex['sql']}")
            prompt_parts.append(f"\nSimilar examples:\n" + "\n\n".join(examples))
        
        # Add retry context
        if context.previous_errors:
            prompt_parts.append(f"\nPrevious attempt failed with: {context.previous_errors[-1]}")
            prompt_parts.append("Please fix the error and generate a valid query.")
        
        return "\n".join(prompt_parts)
    
    def _post_process_sql(self, sql: str) -> str:
        """Post-process SQL for dialect-specific optimizations"""
        # Clean SQL
        sql = sql.replace("```sql", "").replace("```", "").strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.replace(";", "").strip()
        
        # Fix numeric aggregations on text columns
        sql = self._fix_numeric_aggregations(sql)
        
        # Add AS alias for calculated columns without alias
        # Pattern: expressions like SUM(x)/COUNT(y) or AVG(x) without AS
        sql = self._add_missing_aliases(sql)
        
        # Apply dialect-specific processing
        sql = self._apply_dialect_specific_rules(sql)
        
        # Add LIMIT if not present (for SELECT queries)
        if 'select' in sql.lower() and 'limit' not in sql.lower():
            sql = self._add_limit_clause(sql)
        
        return sql
    
    def _add_missing_aliases(self, sql: str) -> str:
        """Add AS alias for calculated columns that don't have one"""
        # Find SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql
        
        select_clause = select_match.group(1)
        # Split by comma, but be careful with nested parentheses
        columns = []
        current_col = ""
        paren_depth = 0
        
        for char in select_clause:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                columns.append(current_col.strip())
                current_col = ""
                continue
            current_col += char
        
        if current_col.strip():
            columns.append(current_col.strip())
        
        # Process each column
        processed_columns = []
        for col in columns:
            col = col.strip()
            # Check if it's a calculated expression without AS
            if any(func in col.upper() for func in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN(', '/', '*', '+', '-']) and ' AS ' not in col.upper():
                # Generate a simple alias
                if 'SUM(' in col.upper() and '/' in col:
                    alias = 'rate'
                elif 'AVG(' in col.upper():
                    alias = 'avg_value'
                elif 'SUM(' in col.upper():
                    alias = 'total'
                elif 'COUNT(' in col.upper():
                    alias = 'count_value'
                else:
                    alias = 'calculated_value'
                
                # Check if alias already exists at the end
                if not col.upper().endswith(f' AS {alias.upper()}'):
                    col = f"{col} AS {alias}"
            
            processed_columns.append(col)
        
        # Reconstruct SELECT clause
        new_select_clause = ', '.join(processed_columns)
        sql = sql[:select_match.start(1)] + new_select_clause + sql[select_match.end(1):]
        
        return sql
    
    def _fix_numeric_aggregations(self, sql: str) -> str:
        """Fix numeric aggregations on text columns by adding CAST"""
        # Pattern: SUM(column), AVG(column), MAX(column), MIN(column) where column might be text
        # We'll add CAST for common text column patterns in ORDER BY and aggregations
        
        # Fix in ORDER BY clause
        order_by_pattern = r'ORDER\s+BY\s+(SUM|AVG|MAX|MIN)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
        def add_cast_order_by(match):
            func = match.group(1)
            col = match.group(2)
            # Check if already has CAST
            if 'CAST' not in sql.upper() or f'CAST({col}' not in sql.upper():
                return f'ORDER BY {func}(CAST({col} AS numeric))'
            return match.group(0)
        
        sql = re.sub(order_by_pattern, add_cast_order_by, sql, flags=re.IGNORECASE)
        
        # Fix in SELECT clause for aggregations
        # Pattern: SUM(column) or AVG(column) without CAST
        agg_pattern = r'\b(SUM|AVG|MAX|MIN)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\)'
        def add_cast_agg(match):
            func = match.group(1)
            col = match.group(2)
            # Skip if already has CAST or is COUNT
            if func.upper() == 'COUNT':
                return match.group(0)
            # Check if column name suggests it might be numeric (contains amt, amount, credit, etc)
            numeric_indicators = ['amt', 'amount', 'credit', 'debt', 'balance', 'sum', 'total', 'value', 'price', 'cost']
            if any(ind in col.lower() for ind in numeric_indicators):
                # Add CAST for safety
                return f'{func}(CAST({col} AS numeric))'
            return match.group(0)
        
        # Apply to SELECT clause only
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Fix aggregations in SELECT
            fixed_select = re.sub(agg_pattern, add_cast_agg, select_clause, flags=re.IGNORECASE)
            sql = sql[:select_match.start(1)] + fixed_select + sql[select_match.end(1):]
        
        return sql
    
    def _apply_dialect_specific_rules(self, sql: str) -> str:
        """Apply dialect-specific SQL rules"""
        if self.dialect == DatabaseDialect.POSTGRESQL:
            # PostgreSQL specific
            # Convert MySQL GROUP_CONCAT to STRING_AGG
            sql = re.sub(
                r"GROUP_CONCAT\((.*?)\)",
                r"STRING_AGG(\1, ',')",
                sql,
                flags=re.IGNORECASE
            )
            # Use COALESCE instead of IFNULL
            sql = re.sub(
                r"IFNULL\((.*?),(.*?)\)",
                r"COALESCE(\1,\2)",
                sql,
                flags=re.IGNORECASE
            )
            
        elif self.dialect == DatabaseDialect.MYSQL:
            # MySQL specific
            # Convert PostgreSQL STRING_AGG to GROUP_CONCAT
            sql = re.sub(
                r"STRING_AGG\((.*?),\s*'.*?'\)",
                r"GROUP_CONCAT(\1)",
                sql,
                flags=re.IGNORECASE
            )
            # Use IFNULL instead of COALESCE (both work but IFNULL is more common)
            sql = re.sub(
                r"COALESCE\((.*?),(.*?)\)",
                r"IFNULL(\1,\2)",
                sql,
                flags=re.IGNORECASE
            )
            
        elif self.dialect == DatabaseDialect.SQLITE:
            # SQLite specific
            # Ensure proper date functions
            sql = re.sub(
                r"NOW\(\)",
                r"datetime('now')",
                sql,
                flags=re.IGNORECASE
            )
            sql = re.sub(
                r"CURRENT_DATE",
                r"date('now')",
                sql,
                flags=re.IGNORECASE
            )
        
        return sql
    
    def _add_limit_clause(self, sql: str) -> str:
        """Add appropriate LIMIT clause based on dialect"""
        # Check if LIMIT already exists
        if 'limit' in sql.lower():
            return sql
        
        # Check if there's ORDER BY - LIMIT should come after ORDER BY
        limit = 100
        
        if self.dialect in [DatabaseDialect.POSTGRESQL, DatabaseDialect.MYSQL, DatabaseDialect.SQLITE]:
            # Find ORDER BY position
            order_by_match = re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE)
            if order_by_match:
                # LIMIT should be after ORDER BY
                sql = f"{sql} LIMIT {limit}"
            else:
                # No ORDER BY, add LIMIT at the end
                sql = f"{sql} LIMIT {limit}"
        elif self.dialect == DatabaseDialect.MSSQL:
            # For SQL Server, use TOP
            sql = re.sub(
                r"SELECT\s+",
                f"SELECT TOP {limit} ",
                sql,
                flags=re.IGNORECASE,
                count=1
            )
        elif self.dialect == DatabaseDialect.ORACLE:
            # For Oracle, use FETCH FIRST
            sql = f"{sql} FETCH FIRST {limit} ROWS ONLY"
        
        return sql
    
    def _extract_query_metadata(self, sql: str) -> Dict[str, Any]:
        """Extract metadata from generated SQL"""
        metadata = {
            'tables': [],
            'columns': [],
            'aggregations': [],
            'filters': []
        }
        
        # Extract tables
        table_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'INTO\s+(\w+)'
        ]
        for pattern in table_patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            metadata['tables'].extend(matches)
        
        metadata['tables'] = list(set(metadata['tables']))
        
        # Extract aggregations
        agg_patterns = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP_CONCAT', 'STRING_AGG']
        for agg in agg_patterns:
            if agg in sql.upper():
                metadata['aggregations'].append(agg)
        
        # Extract WHERE conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            metadata['filters'].append(where_match.group(1).strip())
        
        # Extract columns (basic extraction)
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            # Simple column extraction (doesn't handle all cases)
            columns = re.findall(r'\b(\w+\.\w+|\w+)\b', columns_str)
            metadata['columns'] = list(set(columns))
        
        return metadata
    
    def _calculate_confidence(
        self,
        context: SQLGenerationContext,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for generated SQL"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on context availability
        if context.schema_context:
            confidence += 0.15
        
        if context.similar_queries:
            confidence += 0.1 * min(len(context.similar_queries), 3)
        
        if context.semantic_context.get('suggested_tables'):
            # Check if we're using suggested tables
            suggested = set(context.semantic_context['suggested_tables'])
            used = set(metadata['tables'])
            if suggested & used:
                confidence += 0.15
        
        # Decrease confidence for retries
        confidence -= 0.1 * context.attempt_number
        
        # Cap confidence
        return min(max(confidence, 0.1), 0.95)
    
    def _adjust_context_for_retry(
        self,
        context: SQLGenerationContext,
        error: str
    ) -> SQLGenerationContext:
        """Adjust context for retry attempt"""
        logger.debug(f"[SQL_GEN] Adjusting context for retry due to: {error}")
        
        # If it's a type error (function does not exist for text), add CAST instructions
        if 'function' in error.lower() and ('text' in error.lower() or 'does not exist' in error.lower()):
            context.schema_context += "\n\nIMPORTANT: Some columns might be TEXT type. For numeric aggregations (SUM, AVG, MAX, MIN) on text columns, use CAST(column AS numeric) or CAST(column AS double precision)."
        
        # If it's a schema error, try to get more schema context
        if 'column' in error.lower() or 'table' in error.lower():
            # Try to extract table/column names from error
            table_match = re.search(r'table ["\']?(\w+)["\']?', error, re.IGNORECASE)
            if table_match:
                table = table_match.group(1)
                # Get more detailed schema for this table
                additional_context = self.rag_system.get_schema_context(
                    context.question,
                    [table]
                )
                context.schema_context += f"\n\nAdditional context for {table}:\n{additional_context}"
        
        # If it's a syntax error, switch to more conservative strategy
        if 'syntax' in error.lower():
            if context.strategy == SQLGenerationStrategy.DIRECT:
                context.strategy = SQLGenerationStrategy.RAG_ENHANCED
            elif context.strategy == SQLGenerationStrategy.RAG_ENHANCED:
                context.strategy = SQLGenerationStrategy.FULL_CONTEXT
        
        return context
    
    def validate_and_optimize(self, sql: str) -> Tuple[bool, str, List[str]]:
        """
        Validate and suggest optimizations for SQL
        
        Args:
            sql: SQL query to validate
            
        Returns:
            Tuple of (is_valid, processed_sql, optimization_suggestions)
        """
        # Validate
        is_valid, error = self.db_adapter.validate_query(sql)
        
        if not is_valid:
            return False, sql, [f"Validation error: {error}"]
        
        # Process for dialect
        processed_sql = self._post_process_sql(sql)
        
        # Generate optimization suggestions
        suggestions = []
        
        # Check for SELECT *
        if 'select *' in processed_sql.lower():
            suggestions.append("Consider specifying only required columns instead of SELECT *")
        
        # Check for missing indexes (basic check)
        if 'where' in processed_sql.lower():
            suggestions.append("Ensure columns in WHERE clause have appropriate indexes")
        
        # Check for DISTINCT
        if 'distinct' in processed_sql.lower():
            suggestions.append("DISTINCT can be expensive; ensure it's necessary")
        
        # Check for subqueries
        if processed_sql.lower().count('select') > 1:
            suggestions.append("Consider using JOINs instead of subqueries for better performance")
        
        # Check for proper JOIN conditions
        if 'join' in processed_sql.lower() and 'on' not in processed_sql.lower():
            suggestions.append("WARNING: JOIN without ON condition detected")
        
        return True, processed_sql, suggestions
    
    def explain_query(self, sql: str) -> str:
        """
        Generate explanation for SQL query
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Human-readable explanation
        """
        metadata = self._extract_query_metadata(sql)
        
        explanation_parts = []
        
        # Explain what's being selected
        explanation_parts.append("This query:")
        
        # Tables
        if metadata['tables']:
            explanation_parts.append(f"- Retrieves data from: {', '.join(metadata['tables'])}")
        
        # Aggregations
        if metadata['aggregations']:
            explanation_parts.append(f"- Performs calculations: {', '.join(metadata['aggregations'])}")
        
        # Filters
        if metadata['filters']:
            explanation_parts.append(f"- Applies filters: {metadata['filters'][0][:100]}...")
        
        # Dialect
        explanation_parts.append(f"- Optimized for: {self.dialect.value}")
        
        return "\n".join(explanation_parts)
