import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import time
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, connection_url: str, max_query_time: int = 30, max_rows: int = 100):
        self.connection_url = connection_url
        self.max_query_time = max_query_time
        self.max_rows = max_rows
        
        self.engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_pre_ping=True
        )
        
        self._table_schema_cache = {}
        self._initialize_schema_cache()
    
    def _initialize_schema_cache(self):
        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                inspector = inspect(self.engine)
                for table_name in inspector.get_table_names():
                    columns = []
                    for col in inspector.get_columns(table_name):
                        columns.append({
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'default': col.get('default'),
                            'primary_key': col.get('primary_key', False)
                        })
                    self._table_schema_cache[table_name] = columns
                logger.info(f"Cached schema for {len(self._table_schema_cache)} tables")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to cache schema (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to cache schema after {max_retries} attempts: {e}")
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self.connection_url)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        start_time = time.time()
        
        try:
            if 'limit' not in query.lower():
                query = f"{query} LIMIT {self.max_rows}"
            
            with self.engine.connect() as conn:
                result = pd.read_sql_query(query, conn, params=params)
            
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f}s, returned {len(result)} rows")
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_list(self) -> List[str]:
        return list(self._table_schema_cache.keys())
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        return self._table_schema_cache.get(table_name, [])
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        stats = {}
        
        count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
        result = self.execute_query(count_query)
        stats['row_count'] = int(result['row_count'].iloc[0])
        
        stats['column_count'] = len(self.get_table_schema(table_name))
        
        schema = self.get_table_schema(table_name)
        type_dist = {}
        for col in schema:
            col_type = col['type'].split('(')[0].upper()
            type_dist[col_type] = type_dist.get(col_type, 0) + 1
        stats['column_types'] = type_dist
        
        return stats
    
    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous operation: {keyword}"
        
        try:
            explain_query = f"EXPLAIN {query}"
            with self.engine.connect() as conn:
                conn.execute(text(explain_query))
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_database_info(self) -> Dict[str, Any]:
        info = {
            'tables': self.get_table_list(),
            'total_tables': len(self._table_schema_cache),
            'table_details': {}
        }
        
        for table in info['tables']:
            info['table_details'][table] = {
                'columns': len(self.get_table_schema(table)),
                'sample_columns': [col['name'] for col in self.get_table_schema(table)[:5]]
            }
        
        return info
