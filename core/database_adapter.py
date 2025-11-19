"""
Universal Database Adapter for working with different database systems
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import pandas as pd
from sqlalchemy import create_engine, inspect, text, MetaData
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
import time

logger = logging.getLogger(__name__)


class DatabaseDialect(Enum):
    """Supported database dialects"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    ORACLE = "oracle"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters"""
    
    def __init__(self, connection_url: str, max_query_time: int = 30, max_rows: int = 100):
        self.connection_url = connection_url
        self.max_query_time = max_query_time
        self.max_rows = max_rows
        self.dialect = self._get_dialect()
        self._table_schema_cache = {}
        self._foreign_keys_cache = {}
        self._indexes_cache = {}
        
    @abstractmethod
    def _get_dialect(self) -> DatabaseDialect:
        """Get the database dialect"""
        pass
    
    @abstractmethod
    def get_connection_engine(self) -> Engine:
        """Get SQLAlchemy engine for the database"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        pass
    
    @abstractmethod
    def get_table_list(self) -> List[str]:
        """Get list of all tables in the database"""
        pass
    
    @abstractmethod
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a table"""
        pass
    
    @abstractmethod
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table"""
        pass
    
    @abstractmethod
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get indexes for a table"""
        pass
    
    @abstractmethod
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a query for safety and syntax"""
        pass
    
    @abstractmethod
    def get_sql_dialect_features(self) -> Dict[str, Any]:
        """Get dialect-specific SQL features and syntax"""
        pass
    
    def get_database_metadata(self) -> Dict[str, Any]:
        """Get comprehensive database metadata"""
        return {
            "dialect": self.dialect.value,
            "tables": self.get_table_list(),
            "total_tables": len(self.get_table_list()),
            "features": self.get_sql_dialect_features(),
            "relationships": self._get_all_relationships(),
            "indexes": self._get_all_indexes()
        }
    
    def _get_all_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all foreign key relationships in the database"""
        relationships = {}
        for table in self.get_table_list():
            rels = self.get_table_relationships(table)
            if rels:
                relationships[table] = rels
        return relationships
    
    def _get_all_indexes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all indexes in the database"""
        indexes = {}
        for table in self.get_table_list():
            idx = self.get_table_indexes(table)
            if idx:
                indexes[table] = idx
        return indexes
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table"""
        dialect_limit = self._get_limit_clause(limit)
        query = f"SELECT * FROM {table_name} {dialect_limit}"
        return self.execute_query(query)
    
    def _get_limit_clause(self, limit: int) -> str:
        """Get dialect-specific LIMIT clause"""
        if self.dialect in [DatabaseDialect.POSTGRESQL, DatabaseDialect.MYSQL, DatabaseDialect.SQLITE]:
            return f"LIMIT {limit}"
        elif self.dialect == DatabaseDialect.MSSQL:
            return f"TOP {limit}"
        elif self.dialect == DatabaseDialect.ORACLE:
            return f"FETCH FIRST {limit} ROWS ONLY"
        else:
            return f"LIMIT {limit}"
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics for a table"""
        stats = {}
        
        # Row count
        count_query = f"SELECT COUNT(*) as cnt FROM {table_name}"
        result = self.execute_query(count_query)
        stats['row_count'] = int(result['cnt'].iloc[0])
        
        # Column info
        schema = self.get_table_schema(table_name)
        stats['column_count'] = len(schema)
        
        # Data types distribution
        type_dist = {}
        for col in schema:
            col_type = str(col.get('type', '')).split('(')[0].upper()
            type_dist[col_type] = type_dist.get(col_type, 0) + 1
        stats['column_types'] = type_dist
        
        # Nullable columns
        nullable_count = sum(1 for col in schema if col.get('nullable', False))
        stats['nullable_columns'] = nullable_count
        
        # Primary keys
        pk_count = sum(1 for col in schema if col.get('primary_key', False))
        stats['primary_keys'] = pk_count
        
        return stats


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter"""
    
    def __init__(self, connection_url: str, **kwargs):
        super().__init__(connection_url, **kwargs)
        self.engine = self.get_connection_engine()
        self._initialize_cache()
    
    def _get_dialect(self) -> DatabaseDialect:
        return DatabaseDialect.POSTGRESQL
    
    def get_connection_engine(self) -> Engine:
        """Create PostgreSQL connection engine"""
        return create_engine(
            self.connection_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            pool_pre_ping=True
        )
    
    def _initialize_cache(self):
        """Initialize schema cache"""
        try:
            inspector = inspect(self.engine)
            
            # Cache table schemas
            for table_name in inspector.get_table_names():
                columns = []
                for col in inspector.get_columns(table_name):
                    columns.append({
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'default': col.get('default'),
                        'primary_key': col.get('primary_key', False),
                        'comment': col.get('comment')
                    })
                self._table_schema_cache[table_name] = columns
                
                # Cache foreign keys
                fks = inspector.get_foreign_keys(table_name)
                self._foreign_keys_cache[table_name] = fks
                
                # Cache indexes
                indexes = inspector.get_indexes(table_name)
                self._indexes_cache[table_name] = indexes
            
            logger.info(f"Cached schema for {len(self._table_schema_cache)} tables")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute PostgreSQL query"""
        start_time = time.time()
        
        try:
            # Add limit if not present
            if 'limit' not in query.lower() and 'select' in query.lower():
                query = f"{query} LIMIT {self.max_rows}"
            
            # Convert params to dict if needed (handle immutabledict)
            # pandas.read_sql_query doesn't accept immutabledict, so convert to regular dict
            if params is not None:
                try:
                    # Always convert dict-like objects to regular dict
                    if isinstance(params, dict):
                        # Already a dict, but might be immutabledict subclass
                        if 'immutabledict' in str(type(params)):
                            params = dict(params)
                    elif hasattr(params, 'keys'):
                        # Dict-like object (including immutabledict)
                        params = dict(params)
                    elif isinstance(params, (list, tuple)):
                        # List/tuple of parameters - keep as is
                        pass
                    else:
                        params = None
                except (TypeError, ValueError, AttributeError):
                    params = None
            
            with self.engine.connect() as conn:
                # Don't pass params if None to avoid issues
                if params is not None:
                    result = pd.read_sql_query(query, conn, params=params)
                else:
                    result = pd.read_sql_query(query, conn)
            
            execution_time = time.time() - start_time
            logger.info(f"Query executed in {execution_time:.2f}s, returned {len(result)} rows")
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_list(self) -> List[str]:
        """Get list of PostgreSQL tables"""
        return list(self._table_schema_cache.keys())
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get PostgreSQL table schema"""
        return self._table_schema_cache.get(table_name, [])
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get PostgreSQL foreign key relationships"""
        return self._foreign_keys_cache.get(table_name, [])
    
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get PostgreSQL table indexes"""
        return self._indexes_cache.get(table_name, [])
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate PostgreSQL query"""
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous operation: {keyword}"
        
        try:
            # Use EXPLAIN to validate query
            explain_query = f"EXPLAIN {query}"
            with self.engine.connect() as conn:
                conn.execute(text(explain_query))
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_features(self) -> Dict[str, Any]:
        """Get PostgreSQL-specific features"""
        return {
            "supports_schemas": True,
            "supports_arrays": True,
            "supports_json": True,
            "supports_window_functions": True,
            "supports_cte": True,
            "supports_full_text_search": True,
            "date_functions": ["NOW()", "CURRENT_DATE", "DATE_TRUNC", "EXTRACT"],
            "string_functions": ["CONCAT", "LENGTH", "LOWER", "UPPER", "SUBSTRING"],
            "aggregate_functions": ["AVG", "SUM", "COUNT", "MIN", "MAX", "STRING_AGG"],
            "limit_syntax": "LIMIT",
            "null_handling": "COALESCE",
            "case_sensitivity": False
        }


class MySQLAdapter(DatabaseAdapter):
    """MySQL database adapter"""
    
    def __init__(self, connection_url: str, **kwargs):
        super().__init__(connection_url, **kwargs)
        self.engine = self.get_connection_engine()
        self._initialize_cache()
    
    def _get_dialect(self) -> DatabaseDialect:
        return DatabaseDialect.MYSQL
    
    def get_connection_engine(self) -> Engine:
        """Create MySQL connection engine"""
        return create_engine(
            self.connection_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600
        )
    
    def _initialize_cache(self):
        """Initialize MySQL schema cache"""
        try:
            inspector = inspect(self.engine)
            
            for table_name in inspector.get_table_names():
                columns = []
                for col in inspector.get_columns(table_name):
                    columns.append({
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'default': col.get('default'),
                        'primary_key': col.get('primary_key', False),
                        'comment': col.get('comment'),
                        'autoincrement': col.get('autoincrement', False)
                    })
                self._table_schema_cache[table_name] = columns
                
                # Cache foreign keys
                fks = inspector.get_foreign_keys(table_name)
                self._foreign_keys_cache[table_name] = fks
                
                # Cache indexes
                indexes = inspector.get_indexes(table_name)
                self._indexes_cache[table_name] = indexes
            
            logger.info(f"MySQL: Cached schema for {len(self._table_schema_cache)} tables")
        except Exception as e:
            logger.error(f"MySQL: Failed to initialize cache: {e}")
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute MySQL query"""
        start_time = time.time()
        
        try:
            # Add limit if not present
            if 'limit' not in query.lower() and 'select' in query.lower():
                query = f"{query} LIMIT {self.max_rows}"
            
            # Convert params to dict if needed (handle immutabledict)
            # pandas.read_sql_query doesn't accept immutabledict, so convert to regular dict
            if params is not None:
                try:
                    # Always convert dict-like objects to regular dict
                    if isinstance(params, dict):
                        # Already a dict, but might be immutabledict subclass
                        if 'immutabledict' in str(type(params)):
                            params = dict(params)
                    elif hasattr(params, 'keys'):
                        # Dict-like object (including immutabledict)
                        params = dict(params)
                    elif isinstance(params, (list, tuple)):
                        # List/tuple of parameters - keep as is
                        pass
                    else:
                        params = None
                except (TypeError, ValueError, AttributeError):
                    params = None
            
            with self.engine.connect() as conn:
                # Don't pass params if None to avoid issues
                if params is not None:
                    result = pd.read_sql_query(query, conn, params=params)
                else:
                    result = pd.read_sql_query(query, conn)
            
            execution_time = time.time() - start_time
            logger.info(f"MySQL: Query executed in {execution_time:.2f}s, returned {len(result)} rows")
            
            return result
            
        except Exception as e:
            logger.error(f"MySQL: Query execution failed: {e}")
            raise
    
    def get_table_list(self) -> List[str]:
        """Get list of MySQL tables"""
        return list(self._table_schema_cache.keys())
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get MySQL table schema"""
        return self._table_schema_cache.get(table_name, [])
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get MySQL foreign key relationships"""
        return self._foreign_keys_cache.get(table_name, [])
    
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get MySQL table indexes"""
        return self._indexes_cache.get(table_name, [])
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate MySQL query"""
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous operation: {keyword}"
        
        try:
            # Use EXPLAIN to validate query
            explain_query = f"EXPLAIN {query}"
            with self.engine.connect() as conn:
                conn.execute(text(explain_query))
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_features(self) -> Dict[str, Any]:
        """Get MySQL-specific features"""
        return {
            "supports_schemas": False,
            "supports_arrays": False,
            "supports_json": True,
            "supports_window_functions": True,  # MySQL 8.0+
            "supports_cte": True,  # MySQL 8.0+
            "supports_full_text_search": True,
            "date_functions": ["NOW()", "CURDATE()", "DATE_FORMAT", "DATEDIFF"],
            "string_functions": ["CONCAT", "LENGTH", "LOWER", "UPPER", "SUBSTRING"],
            "aggregate_functions": ["AVG", "SUM", "COUNT", "MIN", "MAX", "GROUP_CONCAT"],
            "limit_syntax": "LIMIT",
            "null_handling": "IFNULL",
            "case_sensitivity": True  # Depends on collation
        }


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter"""
    
    def __init__(self, connection_url: str, **kwargs):
        super().__init__(connection_url, **kwargs)
        self.engine = self.get_connection_engine()
        self._initialize_cache()
    
    def _get_dialect(self) -> DatabaseDialect:
        return DatabaseDialect.SQLITE
    
    def get_connection_engine(self) -> Engine:
        """Create SQLite connection engine"""
        return create_engine(self.connection_url)
    
    def _initialize_cache(self):
        """Initialize SQLite schema cache"""
        try:
            inspector = inspect(self.engine)
            
            for table_name in inspector.get_table_names():
                columns = []
                for col in inspector.get_columns(table_name):
                    columns.append({
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col.get('nullable', True),
                        'default': col.get('default'),
                        'primary_key': col.get('primary_key', False)
                    })
                self._table_schema_cache[table_name] = columns
                
                # Cache foreign keys
                fks = inspector.get_foreign_keys(table_name)
                self._foreign_keys_cache[table_name] = fks
                
                # Cache indexes
                indexes = inspector.get_indexes(table_name)
                self._indexes_cache[table_name] = indexes
            
            logger.info(f"SQLite: Cached schema for {len(self._table_schema_cache)} tables")
        except Exception as e:
            logger.error(f"SQLite: Failed to initialize cache: {e}")
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute SQLite query"""
        start_time = time.time()
        
        try:
            # Add limit if not present
            if 'limit' not in query.lower() and 'select' in query.lower():
                query = f"{query} LIMIT {self.max_rows}"
            
            # Convert params to dict if needed (handle immutabledict)
            # pandas.read_sql_query doesn't accept immutabledict, so convert to regular dict
            if params is not None:
                try:
                    # Always convert dict-like objects to regular dict
                    if isinstance(params, dict):
                        # Already a dict, but might be immutabledict subclass
                        if 'immutabledict' in str(type(params)):
                            params = dict(params)
                    elif hasattr(params, 'keys'):
                        # Dict-like object (including immutabledict)
                        params = dict(params)
                    elif isinstance(params, (list, tuple)):
                        # List/tuple of parameters - keep as is
                        pass
                    else:
                        params = None
                except (TypeError, ValueError, AttributeError):
                    params = None
            
            with self.engine.connect() as conn:
                # Don't pass params if None to avoid issues
                if params is not None:
                    result = pd.read_sql_query(query, conn, params=params)
                else:
                    result = pd.read_sql_query(query, conn)
            
            execution_time = time.time() - start_time
            logger.info(f"SQLite: Query executed in {execution_time:.2f}s, returned {len(result)} rows")
            
            return result
            
        except Exception as e:
            logger.error(f"SQLite: Query execution failed: {e}")
            raise
    
    def get_table_list(self) -> List[str]:
        """Get list of SQLite tables"""
        return list(self._table_schema_cache.keys())
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get SQLite table schema"""
        return self._table_schema_cache.get(table_name, [])
    
    def get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """Get SQLite foreign key relationships"""
        return self._foreign_keys_cache.get(table_name, [])
    
    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get SQLite table indexes"""
        return self._indexes_cache.get(table_name, [])
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate SQLite query"""
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous operation: {keyword}"
        
        try:
            # Use EXPLAIN QUERY PLAN to validate query
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            with self.engine.connect() as conn:
                conn.execute(text(explain_query))
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_sql_dialect_features(self) -> Dict[str, Any]:
        """Get SQLite-specific features"""
        return {
            "supports_schemas": False,
            "supports_arrays": False,
            "supports_json": True,  # SQLite 3.38.0+
            "supports_window_functions": True,  # SQLite 3.25.0+
            "supports_cte": True,
            "supports_full_text_search": True,  # FTS5
            "date_functions": ["date('now')", "datetime('now')", "strftime"],
            "string_functions": ["length", "lower", "upper", "substr"],
            "aggregate_functions": ["avg", "sum", "count", "min", "max", "group_concat"],
            "limit_syntax": "LIMIT",
            "null_handling": "COALESCE",
            "case_sensitivity": False  # By default
        }


def create_database_adapter(connection_url: str, **kwargs) -> DatabaseAdapter:
    """
    Factory function to create appropriate database adapter based on connection URL
    
    Args:
        connection_url: Database connection URL
        **kwargs: Additional parameters for the adapter
    
    Returns:
        Appropriate DatabaseAdapter instance
    """
    if connection_url.startswith('postgresql://') or connection_url.startswith('postgres://'):
        return PostgreSQLAdapter(connection_url, **kwargs)
    elif connection_url.startswith('mysql://') or connection_url.startswith('mysql+pymysql://'):
        return MySQLAdapter(connection_url, **kwargs)
    elif connection_url.startswith('sqlite://'):
        return SQLiteAdapter(connection_url, **kwargs)
    else:
        raise ValueError(f"Unsupported database type in connection URL: {connection_url}")
