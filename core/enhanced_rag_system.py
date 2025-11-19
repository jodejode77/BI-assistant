"""
Enhanced RAG System with automatic database adaptation
"""

import logging
from typing import List, Dict, Any, Optional, Set
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import pandas as pd
import hashlib
from datetime import datetime
from core.database_adapter import DatabaseAdapter
from core.dynamic_semantic_layer import DynamicSemanticLayer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    document: Document
    score: float
    relevance_type: str  # 'schema', 'example', 'semantic', 'learned'


class EnhancedRAGSystem:
    """
    Enhanced RAG system that automatically adapts to any database
    """
    
    def __init__(
        self,
        database_adapter: DatabaseAdapter,
        semantic_layer: DynamicSemanticLayer,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        collection_name: Optional[str] = None
    ):
        """
        Initialize Enhanced RAG System
        
        Args:
            database_adapter: Database adapter instance
            semantic_layer: Dynamic semantic layer instance
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection (auto-generated if None)
        """
        self.db_adapter = database_adapter
        self.semantic_layer = semantic_layer
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        
        # Generate collection name based on database
        if collection_name is None:
            db_hash = hashlib.md5(database_adapter.connection_url.encode()).hexdigest()[:8]
            collection_name = f"rag_{database_adapter.dialect.value}_{db_hash}"
        
        self.collection_name = collection_name
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized Enhanced RAG system for {database_adapter.dialect.value}")
        
        self._document_cache = {}
        self._index_initialized = False
        self._column_descriptions = {}  # Cache for column descriptions from CSV
        
        # Load column descriptions from CSV if available
        self._load_column_descriptions()
        
        # Auto-index database if not already done
        if not self._is_indexed():
            self.auto_index_database()
    
    def _is_indexed(self) -> bool:
        """Check if database is already indexed"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False
    
    def auto_index_database(self):
        """Automatically index the entire database schema and metadata"""
        logger.info("Auto-indexing database schema...")
        
        documents = []
        
        # 1. Index table schemas
        tables = self.db_adapter.get_table_list()
        for table in tables:
            # Index table overview
            table_doc = self._create_table_document(table)
            documents.append(table_doc)
            
            # Index columns
            schema = self.db_adapter.get_table_schema(table)
            for column in schema:
                col_doc = self._create_column_document(table, column)
                documents.append(col_doc)
            
            # Index relationships
            relationships = self.db_adapter.get_table_relationships(table)
            if relationships:
                rel_doc = self._create_relationship_document(table, relationships)
                documents.append(rel_doc)
            
            # Index table statistics
            try:
                stats = self.db_adapter.get_table_statistics(table)
                stats_doc = self._create_statistics_document(table, stats)
                documents.append(stats_doc)
            except Exception as e:
                logger.debug(f"Could not get stats for {table}: {e}")
        
        # 2. Index semantic mappings
        semantic_docs = self._create_semantic_documents()
        documents.extend(semantic_docs)
        
        # 3. Index column descriptions from CSV
        description_docs = self._create_column_description_documents()
        documents.extend(description_docs)
        
        # 4. Add documents to collection
        if documents:
            self.add_documents(documents)
            logger.info(f"Indexed {len(documents)} documents for database")
        
        self._index_initialized = True
    
    def _create_table_document(self, table_name: str) -> Document:
        """Create document for a table"""
        schema = self.db_adapter.get_table_schema(table_name)
        
        # Format columns with types
        col_info = []
        for col in schema[:20]:  # First 20 columns
            col_type = str(col.get('type', 'unknown'))
            col_info.append(f"{col['name']} ({col_type})")
        
        # Get semantic info
        semantic_info = self.semantic_layer.translate_term(table_name)
        
        content = f"""
Table: {table_name}
Type: Database Table
Columns with Types: {', '.join(col_info)}
Total Columns: {len(schema)}
Primary Keys: {', '.join([col['name'] for col in schema if col.get('primary_key')])}
Semantic Mapping: {', '.join(semantic_info.get('related_terms', []))}
Note: Check column types before using aggregations - TEXT columns need CAST(column AS numeric) for SUM/AVG/MAX/MIN
        """.strip()
        
        return Document(
            id=f"table_{table_name}",
            content=content,
            metadata={
                'type': 'table',
                'table_name': table_name,
                'column_count': len(schema),
                'database_dialect': self.db_adapter.dialect.value
            }
        )
    
    def _create_column_document(self, table_name: str, column_info: Dict[str, Any]) -> Document:
        """Create document for a column"""
        col_name = column_info['name']
        
        # Get semantic info from semantic layer
        semantic_key = f"{table_name}.{col_name.lower()}"
        semantic_info = self.semantic_layer.column_semantics.get(semantic_key, {})
        
        col_type = str(column_info.get('type', 'unknown'))
        col_type_normalized = col_type.upper().split('(')[0]
        
        # Add type category for better SQL generation
        type_category = "UNKNOWN"
        if col_type_normalized in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']:
            type_category = "TEXT"
        elif col_type_normalized in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT']:
            type_category = "INTEGER"
        elif col_type_normalized in ['NUMERIC', 'DECIMAL', 'DOUBLE', 'FLOAT', 'REAL']:
            type_category = "NUMERIC"
        elif col_type_normalized in ['DATE', 'TIMESTAMP', 'TIME', 'DATETIME']:
            type_category = "DATE"
        elif col_type_normalized == 'BOOLEAN':
            type_category = "BOOLEAN"
        
        # Get column description from CSV if available
        col_description = self._get_column_description(table_name, col_name)
        
        # Add special notes for important columns
        special_notes = []
        if col_name.upper() == 'TARGET':
            special_notes.append("CRITICAL: TARGET=0 means approved loans, TARGET=1 means defaulted loans. Use WHERE TARGET=0 for approved loans, WHERE TARGET=1 for defaults")
        elif col_name.upper() in ['CODE_GENDER', 'GENDER']:
            special_notes.append("Gender column: typically 'M' for male, 'F' for female")
        elif 'AMT_CREDIT' in col_name.upper() or 'AMT_LOAN' in col_name.upper():
            special_notes.append("Loan amount - use SUM() for total amounts, AVG() for average amounts")
        elif 'AMT_INCOME' in col_name.upper():
            special_notes.append("Income amount - use AVG() for average income")
        
        special_notes_str = "\n".join([f"Special Note: {note}" for note in special_notes]) if special_notes else ""
        description_str = f"Description: {col_description}\n" if col_description else ""
        
        content = f"""
Column: {col_name}
Table: {table_name}
Data Type: {col_type} ({type_category})
Nullable: {column_info.get('nullable', True)}
Primary Key: {column_info.get('primary_key', False)}
Semantic Type: {semantic_info.get('semantic_type', 'unknown')}
Business Terms: {', '.join(semantic_info.get('business_terms', []))}
Data Category: {semantic_info.get('data_category', 'unknown')}
{description_str}Note: For aggregations (SUM, AVG, MAX, MIN) on {type_category} columns, use CAST({col_name} AS numeric) if type is TEXT
{special_notes_str}
        """.strip()
        
        return Document(
            id=f"column_{table_name}_{col_name}",
            content=content,
            metadata={
                'type': 'column',
                'table_name': table_name,
                'column_name': col_name,
                'data_type': str(column_info.get('type', '')),
                'semantic_type': semantic_info.get('semantic_type', 'unknown'),
                'database_dialect': self.db_adapter.dialect.value
            }
        )
    
    def _create_relationship_document(self, table_name: str, relationships: List[Dict[str, Any]]) -> Document:
        """Create document for table relationships"""
        rel_descriptions = []
        for rel in relationships:
            rel_descriptions.append(
                f"- Foreign key to {rel.get('referred_table', 'unknown')}: "
                f"{rel.get('constrained_columns', [])} -> {rel.get('referred_columns', [])}"
            )
        
        content = f"""
Table Relationships: {table_name}
Number of Relationships: {len(relationships)}
Relationships:
{chr(10).join(rel_descriptions)}
        """.strip()
        
        return Document(
            id=f"relationships_{table_name}",
            content=content,
            metadata={
                'type': 'relationship',
                'table_name': table_name,
                'relationship_count': len(relationships),
                'database_dialect': self.db_adapter.dialect.value
            }
        )
    
    def _create_statistics_document(self, table_name: str, stats: Dict[str, Any]) -> Document:
        """Create document for table statistics"""
        content = f"""
Table Statistics: {table_name}
Row Count: {stats.get('row_count', 0):,}
Column Count: {stats.get('column_count', 0)}
Primary Keys: {stats.get('primary_keys', 0)}
Nullable Columns: {stats.get('nullable_columns', 0)}
Data Types: {', '.join([f"{k}: {v}" for k, v in stats.get('column_types', {}).items()])}
        """.strip()
        
        return Document(
            id=f"stats_{table_name}",
            content=content,
            metadata={
                'type': 'statistics',
                'table_name': table_name,
                'row_count': stats.get('row_count', 0),
                'database_dialect': self.db_adapter.dialect.value
            }
        )
    
    def _load_column_descriptions(self):
        """Load column descriptions from HomeCredit_columns_description.csv"""
        csv_paths = [
            Path("data/HomeCredit_columns_description.csv"),
            Path("./data/HomeCredit_columns_description.csv"),
            Path("../data/HomeCredit_columns_description.csv"),
        ]
        
        csv_path = None
        for path in csv_paths:
            if path.exists():
                csv_path = path
                break
        
        if not csv_path:
            logger.warning("HomeCredit_columns_description.csv not found, skipping column descriptions")
            return
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding, on_bad_lines='skip')
                    logger.info(f"Loading column descriptions from {csv_path} (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"Failed to read CSV with any encoding")
                return
            
            for _, row in df.iterrows():
                table_name = str(row.get('Table', '')).strip()
                column_name = str(row.get('Row', '')).strip()
                description = str(row.get('Description', '')).strip()
                special = str(row.get('Special', '')).strip()
                
                if not table_name or not column_name or table_name == 'nan' or column_name == 'nan':
                    continue
                
                # Normalize table name (remove {train|test} pattern)
                table_name = table_name.replace('{train|test}', '').replace('application_', 'application_train').strip()
                if table_name.endswith('.csv'):
                    table_name = table_name[:-4]
                
                # Create key for lookup
                key = f"{table_name.lower()}.{column_name.lower()}"
                
                # Build full description
                full_desc = description
                if special and special.lower() not in ['nan', '']:
                    full_desc += f" (Special: {special})"
                
                self._column_descriptions[key] = full_desc
            
            logger.info(f"Loaded {len(self._column_descriptions)} column descriptions")
        except Exception as e:
            logger.error(f"Failed to load column descriptions: {e}")
    
    def _get_column_description(self, table_name: str, column_name: str) -> Optional[str]:
        """Get column description from loaded CSV data"""
        key = f"{table_name.lower()}.{column_name.lower()}"
        return self._column_descriptions.get(key)
    
    def _create_column_description_documents(self) -> List[Document]:
        """Create documents from column descriptions for better RAG retrieval"""
        documents = []
        
        for key, description in self._column_descriptions.items():
            table_name, column_name = key.split('.', 1)
            
            content = f"""
Column Description: {column_name}
Table: {table_name}
Description: {description}
This is official documentation for the {column_name} column in the {table_name} table.
Use this information to understand what the column represents and how to use it in SQL queries.
            """.strip()
            
            documents.append(Document(
                id=f"col_desc_{table_name}_{column_name}",
                content=content,
                metadata={
                    'type': 'column_description',
                    'table_name': table_name,
                    'column_name': column_name,
                    'database_dialect': self.db_adapter.dialect.value
                }
            ))
        
        return documents
    
    def _create_semantic_documents(self) -> List[Document]:
        """Create documents from semantic layer"""
        documents = []
        
        # Create documents for entity mappings
        for entity, table in self.semantic_layer.entity_mappings.items():
            doc = Document(
                id=f"semantic_entity_{entity}",
                content=f"Business Term: '{entity}' refers to table: {table}",
                metadata={
                    'type': 'semantic_mapping',
                    'entity': entity,
                    'table': table,
                    'database_dialect': self.db_adapter.dialect.value
                }
            )
            documents.append(doc)
        
        # Create documents for metrics
        for metric_name, metric_def in self.semantic_layer.metric_definitions.items():
            doc = Document(
                id=f"semantic_metric_{metric_name}",
                content=f"""
Metric: {metric_name}
Description: {metric_def.get('description', '')}
SQL: {metric_def.get('sql', '')}
Table: {metric_def.get('table', '')}
                """.strip(),
                metadata={
                    'type': 'metric_definition',
                    'metric_name': metric_name,
                    'database_dialect': self.db_adapter.dialect.value
                }
            )
            documents.append(doc)
        
        return documents
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the RAG system"""
        if not documents:
            return
        
        # Check for existing documents
        ids = [doc.id for doc in documents]
        existing_ids = set()
        
        try:
            existing = self.collection.get(ids=ids)
            if existing and existing['ids']:
                existing_ids = set(existing['ids'])
        except Exception as e:
            logger.debug(f"Error checking existing documents: {e}")
        
        # Filter new documents
        new_documents = [doc for doc in documents if doc.id not in existing_ids]
        
        if not new_documents:
            logger.debug(f"All {len(documents)} documents already exist")
            return
        
        # Generate embeddings
        contents = [doc.content for doc in new_documents]
        embeddings = self.embedder.encode(contents).tolist()
        
        # Prepare data for ChromaDB
        new_ids = [doc.id for doc in new_documents]
        # Clean metadata: remove None values (ChromaDB doesn't accept None)
        metadatas = []
        for doc in new_documents:
            clean_metadata = {
                k: v for k, v in doc.metadata.items() 
                if v is not None
            }
            metadatas.append(clean_metadata)
        
        # Add to collection
        self.collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        # Update cache
        for doc, embedding in zip(new_documents, embeddings):
            doc.embedding = np.array(embedding)
            self._document_cache[doc.id] = doc
        
        logger.info(f"Added {len(new_documents)} documents to RAG system")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        search_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents with type filtering
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            search_types: Optional list of document types to search
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Build metadata filter (ChromaDB format)
        where_clause = None
        
        # Если filter_metadata уже в правильном формате ($and, $or), используем его
        if filter_metadata and ('$and' in filter_metadata or '$or' in filter_metadata):
            where_clause = filter_metadata
            # Добавляем search_types если нужно
            if search_types:
                if '$and' in where_clause:
                    where_clause['$and'].append({'type': {'$in': search_types}})
                else:
                    # Если был $or, создаем новый $and
                    where_clause = {'$and': [where_clause, {'type': {'$in': search_types}}]}
        else:
            # Собираем условия
            conditions = []
            
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if isinstance(value, dict):
                        # Уже в правильном формате ($eq, $in, etc.)
                        conditions.append({key: value})
                    else:
                        # Простое значение - конвертируем в $eq
                        conditions.append({key: {'$eq': value}})
            
            if search_types:
                conditions.append({'type': {'$in': search_types}})
            
            # Если есть условия, оборачиваем в $and
            if conditions:
                if len(conditions) == 1:
                    where_clause = conditions[0]
                else:
                    where_clause = {'$and': conditions}
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Parse results
        search_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                
                # Get document
                if doc_id in self._document_cache:
                    doc = self._document_cache[doc_id]
                else:
                    doc = Document(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    )
                
                # Determine relevance type
                doc_type = doc.metadata.get('type', 'unknown')
                if doc_type in ['table', 'column', 'relationship']:
                    relevance_type = 'schema'
                elif doc_type == 'sql_example':
                    relevance_type = 'example'
                elif doc_type in ['semantic_mapping', 'metric_definition']:
                    relevance_type = 'semantic'
                elif doc_type == 'learned_query':
                    relevance_type = 'learned'
                else:
                    relevance_type = 'unknown'
                
                score = 1 - results['distances'][0][i]
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    relevance_type=relevance_type
                ))
        
        return search_results
    
    def get_schema_context(self, query: str, tables: Optional[List[str]] = None, structured: bool = True) -> str:
        """
        Get relevant schema context for a query with schema-guided reasoning support
        
        Args:
            query: User query
            tables: Optional list of specific tables to focus on
            structured: If True, return structured format for schema-guided reasoning
            
        Returns:
            Schema context as formatted string
        """
        # Always include list of available tables first with their columns
        all_tables = self.db_adapter.get_table_list()
        
        if structured:
            # Structured format for schema-guided reasoning
            context_parts = ["=" * 80]
            context_parts.append("DATABASE SCHEMA FOR SQL GENERATION")
            context_parts.append("=" * 80)
            context_parts.append(f"\nAVAILABLE TABLES ({len(all_tables)} total):")
            context_parts.append(f"   {', '.join(all_tables)}")
            context_parts.append("\nCRITICAL: Use ONLY these exact table names. Do NOT invent or guess names.\n")
        else:
            context_parts = [f"AVAILABLE TABLES IN DATABASE: {', '.join(all_tables)}"]
            context_parts.append("IMPORTANT: Use ONLY these exact table names and column names. Do not invent or guess names.\n")
        
        # Add detailed column information for relevant tables
        tables_to_show = tables if tables else all_tables[:10]  # Show more tables for better context
        
        if structured:
            context_parts.append("=" * 80)
            context_parts.append("TABLE SCHEMAS (for mapping entities to columns)")
            context_parts.append("=" * 80)
        
        for table in tables_to_show:
            schema = self.db_adapter.get_table_schema(table)
            if schema:
                if structured:
                    # Structured format with column types and constraints
                    context_parts.append(f"\n📊 Table: {table}")
                    context_parts.append(f"   Columns ({len(schema)}):")
                    for col in schema:
                        col_type = str(col.get('type', 'unknown'))
                        col_name = col['name']
                        # Normalize type for better understanding
                        col_type_normalized = col_type.upper().split('(')[0]  # Remove size, e.g., VARCHAR(255) -> VARCHAR
                        nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                        pk = " [PRIMARY KEY]" if col.get('primary_key', False) else ""
                        # Add note about numeric vs text for aggregations
                        type_note = ""
                        if col_type_normalized in ['TEXT', 'VARCHAR', 'CHAR', 'STRING']:
                            type_note = " [TEXT - use CAST(column AS numeric) for SUM/AVG/MAX/MIN]"
                        elif col_type_normalized in ['INTEGER', 'INT', 'BIGINT', 'SMALLINT', 'NUMERIC', 'DECIMAL', 'DOUBLE', 'FLOAT', 'REAL']:
                            type_note = " [NUMERIC - safe for aggregations]"
                        
                        # Get column description from CSV
                        col_desc = self._get_column_description(table, col_name)
                        desc_note = f" - {col_desc}" if col_desc else ""
                        
                        # Add special notes for important columns
                        special_note = ""
                        if col_name.upper() == 'TARGET':
                            special_note = " [CRITICAL: TARGET=0 means approved loans, TARGET=1 means defaulted loans. Use WHERE TARGET=0 for approved, WHERE TARGET=1 for defaults]"
                        elif col_name.upper() in ['CODE_GENDER', 'GENDER']:
                            special_note = " [Gender column: typically 'M' for male, 'F' for female]"
                        elif 'AMT_CREDIT' in col_name.upper() or 'AMT_LOAN' in col_name.upper():
                            special_note = " [Loan amount - use SUM() for total amounts, AVG() for average amounts]"
                        elif 'AMT_INCOME' in col_name.upper():
                            special_note = " [Income amount - use AVG() for average income]"
                        
                        context_parts.append(f"     - {col_name}: {col_type} {nullable}{pk}{desc_note}{type_note}{special_note}")
                    
                    # Add foreign keys if available
                    relationships = self.db_adapter.get_table_relationships(table)
                    if relationships:
                        context_parts.append(f"   Foreign Keys:")
                        for rel in relationships[:3]:  # Limit to 3 most relevant
                            ref_table = rel.get('referred_table', 'unknown')
                            constrained_cols = ', '.join(rel.get('constrained_columns', []))
                            referred_cols = ', '.join(rel.get('referred_columns', []))
                            context_parts.append(f"     - {constrained_cols} -> {ref_table}.{referred_cols}")
                else:
                    columns = [col['name'] for col in schema]
                    context_parts.append(f"Table '{table}' columns: {', '.join(columns)}")
        
        # Search for relevant schema using RAG
        search_types = ['table', 'column', 'relationship', 'statistics']
        filter_metadata = {}
        
        if tables:
            filter_metadata['table_name'] = {'$in': tables}
        
        results = self.search(
            query,
            top_k=15,  # Get more results for better context
            filter_metadata=filter_metadata if filter_metadata else None,
            search_types=search_types
        )
        
        if not results:
            if structured:
                context_parts.append("\n" + "=" * 80)
                context_parts.append("No additional schema details found via RAG search.")
                context_parts.append("=" * 80)
            else:
                return "\n".join(context_parts) + "\n\nNo additional schema details found."
            return "\n".join(context_parts)
        
        # Group by type for structured output
        schema_by_type = {
            'tables': [],
            'columns': [],
            'relationships': [],
            'statistics': []
        }
        
        for result in results:
            doc_type = result.document.metadata.get('type')
            if doc_type == 'table':
                schema_by_type['tables'].append((result.document.content, result.score))
            elif doc_type == 'column':
                schema_by_type['columns'].append((result.document.content, result.score))
            elif doc_type == 'relationship':
                schema_by_type['relationships'].append((result.document.content, result.score))
            elif doc_type == 'statistics':
                schema_by_type['statistics'].append((result.document.content, result.score))
        
        # Format structured context
        if structured:
            context_parts.append("\n" + "=" * 80)
            context_parts.append("RAG-ENHANCED SCHEMA CONTEXT (from semantic search)")
            context_parts.append("=" * 80)
        
        if schema_by_type['relationships']:
            if structured:
                context_parts.append("\n🔗 TABLE RELATIONSHIPS (for JOIN planning):")
                for content, score in sorted(schema_by_type['relationships'], key=lambda x: x[1], reverse=True)[:3]:
                    context_parts.append(f"   {content}")
            else:
                context_parts.append("TABLE RELATIONSHIPS:\n" + "\n".join([c[0] for c in schema_by_type['relationships'][:2]]))
        
        if schema_by_type['tables']:
            if structured:
                context_parts.append("\n📋 RELEVANT TABLE DETAILS:")
                for content, score in sorted(schema_by_type['tables'], key=lambda x: x[1], reverse=True)[:3]:
                    context_parts.append(f"   {content}")
            else:
                context_parts.append("RELEVANT TABLES:\n" + "\n".join([c[0] for c in schema_by_type['tables'][:3]]))
        
        if schema_by_type['columns']:
            if structured:
                context_parts.append("\n📊 RELEVANT COLUMNS (for SELECT and WHERE clauses):")
                for content, score in sorted(schema_by_type['columns'], key=lambda x: x[1], reverse=True)[:5]:
                    context_parts.append(f"   {content}")
            else:
                context_parts.append("RELEVANT COLUMNS:\n" + "\n".join([c[0] for c in schema_by_type['columns'][:5]]))
        
        if schema_by_type['statistics']:
            if structured:
                context_parts.append("\n📈 TABLE STATISTICS (for query optimization):")
                for content, score in sorted(schema_by_type['statistics'], key=lambda x: x[1], reverse=True)[:2]:
                    context_parts.append(f"   {content}")
            else:
                context_parts.append("TABLE STATISTICS:\n" + "\n".join([c[0] for c in schema_by_type['statistics'][:2]]))
        
        if structured:
            context_parts.append("\n" + "=" * 80)
        
        return "\n".join(context_parts)
    
    def index_sql_examples(self, examples: List[Dict[str, str]]) -> None:
        """
        Index SQL query examples for few-shot learning
        
        Args:
            examples: List of dictionaries with 'question' and 'sql' keys
        """
        documents = []
        
        for i, example in enumerate(examples):
            doc = Document(
                id=f"sql_example_{self.db_adapter.dialect.value}_{i}",
                content=f"""
Question: {example['question']}
SQL Query: {example['sql']}
Database: {self.db_adapter.dialect.value}
                """.strip(),
                metadata={
                    'type': 'sql_example',
                    'question': example['question'],
                    'sql': example['sql'],
                    'database_dialect': self.db_adapter.dialect.value,
                    'tables': example.get('tables', [])
                }
            )
            documents.append(doc)
        
        if documents:
            self.add_documents(documents)
            logger.info(f"Indexed {len(documents)} SQL examples")
    
    def get_similar_queries(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get similar SQL queries for few-shot learning
        
        Args:
            question: User question
            top_k: Number of examples to retrieve
            min_score: Minimum similarity score
            
        Returns:
            List of similar query examples
        """
        # ChromaDB требует правильный формат фильтра с $and для множественных условий
        where_filter = {
            '$and': [
                {'type': {'$eq': 'sql_example'}},
                {'database_dialect': {'$eq': self.db_adapter.dialect.value}}
            ]
        }
        
        results = self.search(
            question,
            top_k=top_k,
            filter_metadata=where_filter
        )
        
        examples = []
        for result in results:
            if result.score > min_score:
                examples.append({
                    'question': result.document.metadata.get('question', ''),
                    'sql': result.document.metadata.get('sql', ''),
                    'score': result.score,
                    'relevance_type': result.relevance_type
                })
        
        return examples
    
    def learn_from_query(
        self,
        question: str,
        sql: str,
        execution_time: float,
        rows_returned: int,
        user_feedback: Optional[str] = None
    ):
        """
        Learn from a successful query execution
        
        Args:
            question: Natural language question
            sql: Generated SQL query
            execution_time: Query execution time
            rows_returned: Number of rows returned
            user_feedback: Optional user feedback
        """
        # Create learned query document
        doc = Document(
            id=f"learned_{hashlib.md5((question + sql).encode()).hexdigest()[:10]}",
            content=f"""
Learned Query:
Question: {question}
SQL: {sql}
Execution Time: {execution_time:.2f}s
Rows Returned: {rows_returned}
Feedback: {user_feedback or 'N/A'}
Timestamp: {datetime.now().isoformat()}
Database: {self.db_adapter.dialect.value}
            """.strip(),
            metadata={
                'type': 'learned_query',
                'question': question,
                'sql': sql,
                'execution_time': execution_time,
                'rows_returned': rows_returned,
                'feedback': user_feedback,
                'database_dialect': self.db_adapter.dialect.value,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self.add_documents([doc])
        
        # Also update semantic layer
        self.semantic_layer.learn_from_query(question, sql, user_feedback)
        
        logger.info(f"Learned from query execution: {rows_returned} rows in {execution_time:.2f}s")
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get a summary of the indexed database"""
        # Get document counts by type
        all_docs = self.collection.get()
        doc_counts = {}
        
        if all_docs and all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                doc_type = metadata.get('type', 'unknown')
                doc_counts[doc_type] = doc_counts.get(doc_type, 0) + 1
        
        return {
            'database_dialect': self.db_adapter.dialect.value,
            'total_documents': self.collection.count(),
            'document_types': doc_counts,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
            'tables_indexed': len(self.db_adapter.get_table_list()),
            'semantic_mappings': len(self.semantic_layer.entity_mappings),
            'learned_queries': doc_counts.get('learned_query', 0)
        }
    
    def export_knowledge(self, output_file: str):
        """Export all knowledge to a file"""
        knowledge = {
            'database_summary': self.get_database_summary(),
            'semantic_knowledge': self.semantic_layer.export_knowledge_base(),
            'indexed_examples': self.get_similar_queries("", top_k=100, min_score=0)  # Get all examples
        }
        
        with open(output_file, 'w') as f:
            json.dump(knowledge, f, indent=2)
        
        logger.info(f"Exported knowledge to {output_file}")
