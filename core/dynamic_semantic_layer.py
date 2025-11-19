"""
Dynamic Semantic Layer that automatically adapts to any database
"""

import logging
from typing import Dict, List, Any, Optional, Set
import json
from pathlib import Path
import re
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher
from core.database_adapter import DatabaseAdapter

logger = logging.getLogger(__name__)


class DynamicSemanticLayer:
    """
    Dynamic semantic layer that automatically learns and adapts to any database schema
    """
    
    def __init__(self, database_adapter: DatabaseAdapter, config_file: Optional[str] = None):
        """
        Initialize dynamic semantic layer
        
        Args:
            database_adapter: Database adapter instance
            config_file: Optional configuration file path
        """
        self.db_adapter = database_adapter
        self.entity_mappings = {}
        self.metric_definitions = {}
        self.relationship_mappings = {}
        self.synonyms = {}
        self.column_semantics = {}
        self.learned_patterns = {}
        
        # Initialize from database schema
        self._auto_discover_schema()
        
        # Load any existing configuration
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def _auto_discover_schema(self):
        """Automatically discover and analyze database schema"""
        logger.info("Auto-discovering database schema...")
        
        # Get all tables
        tables = self.db_adapter.get_table_list()
        
        for table in tables:
            # Analyze table name
            self._analyze_table_name(table)
            
            # Get table schema
            schema = self.db_adapter.get_table_schema(table)
            
            # Analyze columns
            for column in schema:
                self._analyze_column(table, column)
            
            # Get relationships
            relationships = self.db_adapter.get_table_relationships(table)
            self._analyze_relationships(table, relationships)
        
        # Detect common patterns
        self._detect_patterns()
        
        logger.info(f"Discovered {len(tables)} tables with {len(self.column_semantics)} semantic mappings")
    
    def _analyze_table_name(self, table_name: str):
        """Analyze table name to extract semantic meaning"""
        # Clean table name
        clean_name = table_name.lower().replace('_', ' ').replace('-', ' ')
        
        # Common table name patterns
        patterns = {
            r'.*user.*': 'users',
            r'.*customer.*': 'customers',
            r'.*client.*': 'clients',
            r'.*order.*': 'orders',
            r'.*product.*': 'products',
            r'.*transaction.*': 'transactions',
            r'.*payment.*': 'payments',
            r'.*invoice.*': 'invoices',
            r'.*account.*': 'accounts',
            r'.*employee.*': 'employees',
            r'.*department.*': 'departments',
            r'.*category.*': 'categories',
            r'.*log.*': 'logs',
            r'.*audit.*': 'audit',
            r'.*config.*': 'configuration',
            r'.*setting.*': 'settings'
        }
        
        # Match patterns
        for pattern, entity_type in patterns.items():
            if re.match(pattern, clean_name):
                self.entity_mappings[entity_type] = table_name
                self.entity_mappings[clean_name] = table_name
                break
        else:
            # Use the table name itself as entity
            self.entity_mappings[clean_name] = table_name
            
            # Also map singular/plural forms
            if clean_name.endswith('s'):
                self.entity_mappings[clean_name[:-1]] = table_name
            else:
                self.entity_mappings[clean_name + 's'] = table_name
    
    def _analyze_column(self, table_name: str, column_info: Dict[str, Any]):
        """Analyze column to extract semantic meaning"""
        col_name = column_info['name'].lower()
        col_type = str(column_info.get('type', '')).lower()
        
        # Create semantic key
        semantic_key = f"{table_name}.{col_name}"
        
        # Initialize column semantics
        self.column_semantics[semantic_key] = {
            'table': table_name,
            'column': column_info['name'],
            'type': col_type,
            'nullable': column_info.get('nullable', True),
            'primary_key': column_info.get('primary_key', False),
            'semantic_type': self._infer_semantic_type(col_name, col_type),
            'business_terms': self._extract_business_terms(col_name),
            'data_category': self._infer_data_category(col_name, col_type)
        }
        
        # Update synonyms
        for term in self.column_semantics[semantic_key]['business_terms']:
            if term not in self.synonyms:
                self.synonyms[term] = []
            self.synonyms[term].append(column_info['name'])
    
    def _infer_semantic_type(self, col_name: str, col_type: str) -> str:
        """Infer semantic type of a column based on name and data type"""
        col_lower = col_name.lower()
        
        # Common semantic patterns
        if 'id' in col_lower or col_lower.endswith('_key'):
            return 'identifier'
        elif 'date' in col_lower or 'time' in col_lower or 'datetime' in col_type:
            return 'temporal'
        elif 'amount' in col_lower or 'price' in col_lower or 'cost' in col_lower:
            return 'monetary'
        elif 'count' in col_lower or 'number' in col_lower or 'qty' in col_lower:
            return 'quantity'
        elif 'percent' in col_lower or 'rate' in col_lower or 'ratio' in col_lower:
            return 'percentage'
        elif 'name' in col_lower or 'title' in col_lower:
            return 'name'
        elif 'email' in col_lower:
            return 'email'
        elif 'phone' in col_lower or 'mobile' in col_lower:
            return 'phone'
        elif 'address' in col_lower or 'street' in col_lower or 'city' in col_lower:
            return 'address'
        elif 'status' in col_lower or 'state' in col_lower:
            return 'status'
        elif 'flag' in col_lower or col_lower.startswith('is_') or col_lower.startswith('has_'):
            return 'boolean'
        elif 'description' in col_lower or 'comment' in col_lower or 'note' in col_lower:
            return 'text'
        elif 'url' in col_lower or 'link' in col_lower:
            return 'url'
        elif 'json' in col_type:
            return 'json'
        elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type:
            return 'numeric'
        elif 'varchar' in col_type or 'text' in col_type or 'char' in col_type:
            return 'text'
        else:
            return 'unknown'
    
    def _extract_business_terms(self, col_name: str) -> List[str]:
        """Extract business terms from column name"""
        terms = []
        
        # Split by underscore and camelCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', col_name)
        if not parts:
            parts = col_name.split('_')
        
        # Clean and add terms
        for part in parts:
            clean_part = part.lower().strip()
            if clean_part and len(clean_part) > 1:
                terms.append(clean_part)
        
        # Add full column name
        terms.append(col_name.lower())
        
        # Add common variations
        if col_name.lower().endswith('_id'):
            base = col_name[:-3].lower()
            terms.append(base)
        
        return list(set(terms))
    
    def _infer_data_category(self, col_name: str, col_type: str) -> str:
        """Infer data category (dimension, measure, etc.)"""
        col_lower = col_name.lower()
        
        # Measures (aggregatable numeric values)
        if any(keyword in col_lower for keyword in ['amount', 'price', 'cost', 'revenue', 'profit', 'sum', 'total']):
            return 'measure'
        elif 'count' in col_lower or 'number' in col_lower:
            return 'measure'
        elif any(keyword in col_lower for keyword in ['rate', 'ratio', 'percent', 'score']):
            return 'measure'
        
        # Dimensions (categorical/descriptive)
        elif any(keyword in col_lower for keyword in ['name', 'type', 'category', 'group', 'class']):
            return 'dimension'
        elif any(keyword in col_lower for keyword in ['status', 'state', 'flag']):
            return 'dimension'
        elif 'id' in col_lower and not col_lower.endswith('_id'):
            return 'dimension'
        
        # Time dimensions
        elif 'date' in col_lower or 'time' in col_lower:
            return 'time_dimension'
        
        # Keys
        elif col_lower.endswith('_id') or col_lower.endswith('_key'):
            return 'key'
        
        # Default based on data type
        elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type:
            return 'measure'
        else:
            return 'dimension'
    
    def _analyze_relationships(self, table_name: str, relationships: List[Dict[str, Any]]):
        """Analyze table relationships"""
        if table_name not in self.relationship_mappings:
            self.relationship_mappings[table_name] = {}
        
        for rel in relationships:
            referred_table = rel.get('referred_table')
            if referred_table:
                # Store the relationship
                self.relationship_mappings[table_name][referred_table] = {
                    'local_columns': rel.get('constrained_columns', []),
                    'foreign_columns': rel.get('referred_columns', []),
                    'relationship_type': 'foreign_key'
                }
    
    def _detect_patterns(self):
        """Detect common patterns in the database schema"""
        # Detect common ID patterns
        id_patterns = defaultdict(list)
        for semantic_key, info in self.column_semantics.items():
            if info['semantic_type'] == 'identifier':
                col_name = info['column'].lower()
                if '_id' in col_name:
                    base = col_name.replace('_id', '')
                    id_patterns[base].append(semantic_key)
        
        self.learned_patterns['id_patterns'] = dict(id_patterns)
        
        # Detect date patterns
        date_patterns = []
        for semantic_key, info in self.column_semantics.items():
            if info['semantic_type'] == 'temporal':
                date_patterns.append(info['column'])
        
        self.learned_patterns['date_columns'] = date_patterns
        
        # Detect measure columns
        measure_patterns = []
        for semantic_key, info in self.column_semantics.items():
            if info['data_category'] == 'measure':
                measure_patterns.append({
                    'table': info['table'],
                    'column': info['column'],
                    'type': info['semantic_type']
                })
        
        self.learned_patterns['measures'] = measure_patterns
    
    def translate_term(self, term: str) -> Dict[str, Any]:
        """
        Translate a business term to database schema
        
        Args:
            term: Business term to translate
            
        Returns:
            Dictionary with translation details
        """
        term_lower = term.lower()
        
        result = {
            'original_term': term,
            'tables': [],
            'columns': [],
            'metrics': [],
            'related_terms': [],
            'confidence': 0.0
        }
        
        # Check entity mappings (tables)
        for entity, table in self.entity_mappings.items():
            similarity = SequenceMatcher(None, term_lower, entity.lower()).ratio()
            if similarity > 0.7 or term_lower in entity or entity in term_lower:
                result['tables'].append({
                    'name': table,
                    'match_type': 'entity',
                    'confidence': similarity
                })
        
        # Check column semantics
        for semantic_key, info in self.column_semantics.items():
            for business_term in info['business_terms']:
                similarity = SequenceMatcher(None, term_lower, business_term).ratio()
                if similarity > 0.7 or term_lower in business_term or business_term in term_lower:
                    result['columns'].append({
                        'table': info['table'],
                        'column': info['column'],
                        'semantic_type': info['semantic_type'],
                        'match_type': 'semantic',
                        'confidence': similarity
                    })
        
        # Check synonyms
        for synonym, columns in self.synonyms.items():
            if term_lower == synonym or term_lower in synonym:
                result['related_terms'].extend(columns)
        
        # Check metric definitions
        for metric_name, metric_def in self.metric_definitions.items():
            if term_lower in metric_name or metric_name in term_lower:
                result['metrics'].append(metric_def)
        
        # Calculate overall confidence
        if result['tables'] or result['columns'] or result['metrics']:
            confidences = []
            if result['tables']:
                confidences.extend([t['confidence'] for t in result['tables']])
            if result['columns']:
                confidences.extend([c['confidence'] for c in result['columns']])
            result['confidence'] = max(confidences) if confidences else 0.0
        
        return result
    
    def suggest_metrics(self, context: str) -> List[Dict[str, Any]]:
        """
        Suggest relevant metrics based on context
        
        Args:
            context: Query context or topic
            
        Returns:
            List of suggested metrics
        """
        context_lower = context.lower()
        suggestions = []
        
        # Check learned measure patterns
        for measure in self.learned_patterns.get('measures', []):
            col_name = measure['column'].lower()
            if any(word in context_lower for word in col_name.split('_')):
                suggestions.append({
                    'name': f"{measure['table']}.{measure['column']}",
                    'type': measure['type'],
                    'aggregations': self._suggest_aggregations(measure['type']),
                    'table': measure['table'],
                    'column': measure['column']
                })
        
        # Add predefined metrics
        for metric_name, metric_def in self.metric_definitions.items():
            metric_words = metric_name.replace('_', ' ').split()
            if any(word in context_lower for word in metric_words):
                suggestions.append(metric_def)
        
        return suggestions
    
    def _suggest_aggregations(self, semantic_type: str) -> List[str]:
        """Suggest appropriate aggregations based on semantic type"""
        if semantic_type in ['monetary', 'quantity']:
            return ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT']
        elif semantic_type == 'percentage':
            return ['AVG', 'MIN', 'MAX']
        elif semantic_type == 'identifier':
            return ['COUNT', 'COUNT DISTINCT']
        else:
            return ['COUNT']
    
    def get_join_suggestions(self, tables: List[str]) -> List[Dict[str, Any]]:
        """
        Get join suggestions for given tables
        
        Args:
            tables: List of table names
            
        Returns:
            List of join suggestions
        """
        suggestions = []
        
        # Check direct relationships
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                # Check if there's a direct relationship
                if table1 in self.relationship_mappings:
                    if table2 in self.relationship_mappings[table1]:
                        rel = self.relationship_mappings[table1][table2]
                        suggestions.append({
                            'type': 'direct',
                            'table1': table1,
                            'table2': table2,
                            'join_columns': list(zip(rel['local_columns'], rel['foreign_columns'])),
                            'confidence': 1.0
                        })
                
                # Check reverse relationship
                if table2 in self.relationship_mappings:
                    if table1 in self.relationship_mappings[table2]:
                        rel = self.relationship_mappings[table2][table1]
                        suggestions.append({
                            'type': 'direct',
                            'table1': table2,
                            'table2': table1,
                            'join_columns': list(zip(rel['local_columns'], rel['foreign_columns'])),
                            'confidence': 1.0
                        })
        
        # If no direct relationships, suggest based on common column names
        if not suggestions:
            for i, table1 in enumerate(tables):
                schema1 = self.db_adapter.get_table_schema(table1)
                cols1 = [col['name'].lower() for col in schema1]
                
                for table2 in tables[i+1:]:
                    schema2 = self.db_adapter.get_table_schema(table2)
                    cols2 = [col['name'].lower() for col in schema2]
                    
                    # Find common columns
                    common_cols = set(cols1) & set(cols2)
                    if common_cols:
                        # Prioritize ID columns
                        id_cols = [col for col in common_cols if 'id' in col]
                        if id_cols:
                            suggestions.append({
                                'type': 'inferred',
                                'table1': table1,
                                'table2': table2,
                                'join_columns': [(col, col) for col in id_cols],
                                'confidence': 0.8
                            })
        
        return suggestions
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance a natural language query with semantic understanding
        
        Args:
            query: Natural language query
            
        Returns:
            Enhanced query context
        """
        query_lower = query.lower()
        
        context = {
            'original_query': query,
            'detected_entities': [],
            'suggested_tables': [],
            'suggested_columns': [],
            'suggested_metrics': self.suggest_metrics(query),
            'potential_joins': [],
            'filters': [],
            'aggregations': []
        }
        
        # Detect entities and tables
        for entity, table in self.entity_mappings.items():
            if entity in query_lower:
                context['detected_entities'].append(entity)
                if table not in context['suggested_tables']:
                    context['suggested_tables'].append(table)
        
        # Detect columns
        for semantic_key, info in self.column_semantics.items():
            for term in info['business_terms']:
                if term in query_lower:
                    context['suggested_columns'].append({
                        'table': info['table'],
                        'column': info['column'],
                        'semantic_type': info['semantic_type']
                    })
        
        # Detect aggregation keywords
        agg_keywords = {
            'average': 'AVG', 'mean': 'AVG',
            'sum': 'SUM', 'total': 'SUM',
            'count': 'COUNT', 'number of': 'COUNT',
            'maximum': 'MAX', 'highest': 'MAX',
            'minimum': 'MIN', 'lowest': 'MIN'
        }
        
        for keyword, agg in agg_keywords.items():
            if keyword in query_lower:
                context['aggregations'].append(agg)
        
        # Detect filter keywords
        filter_patterns = [
            (r'where\s+(\w+)\s*=\s*(.+)', 'equals'),
            (r'greater than\s+(\d+)', 'greater_than'),
            (r'less than\s+(\d+)', 'less_than'),
            (r'between\s+(\d+)\s+and\s+(\d+)', 'between'),
            (r'in\s+\((.*?)\)', 'in'),
            (r'like\s+[\'"](.+?)[\'"]', 'like')
        ]
        
        for pattern, filter_type in filter_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                context['filters'].append({
                    'type': filter_type,
                    'values': matches
                })
        
        # Get join suggestions if multiple tables detected
        if len(context['suggested_tables']) > 1:
            context['potential_joins'] = self.get_join_suggestions(context['suggested_tables'])
        
        return context
    
    def learn_from_query(self, question: str, sql: str, feedback: Optional[str] = None):
        """
        Learn from a successful query execution
        
        Args:
            question: Natural language question
            sql: Generated SQL query
            feedback: Optional user feedback
        """
        # Extract tables from SQL
        tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.IGNORECASE)
        tables = [t for pair in tables for t in pair if t]
        
        # Extract columns from SQL
        columns = re.findall(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        
        # Learn entity mappings
        question_lower = question.lower()
        for table in tables:
            # Find words in question that might refer to this table
            table_words = table.lower().replace('_', ' ').split()
            for word in table_words:
                if word in question_lower and word not in self.entity_mappings:
                    self.entity_mappings[word] = table
        
        # Store as a learned pattern
        if 'learned_queries' not in self.learned_patterns:
            self.learned_patterns['learned_queries'] = []
        
        self.learned_patterns['learned_queries'].append({
            'question': question,
            'sql': sql,
            'tables': list(set(tables)),
            'feedback': feedback
        })
        
        logger.info(f"Learned from query: {len(tables)} tables, feedback: {feedback}")
    
    def save_config(self, config_file: str):
        """Save semantic configuration to file"""
        config = {
            'entity_mappings': self.entity_mappings,
            'metric_definitions': self.metric_definitions,
            'relationship_mappings': self.relationship_mappings,
            'synonyms': self.synonyms,
            'column_semantics': self.column_semantics,
            'learned_patterns': self.learned_patterns
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved semantic configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save semantic configuration: {e}")
    
    def load_config(self, config_file: str):
        """Load semantic configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update mappings
            if 'entity_mappings' in config:
                self.entity_mappings.update(config['entity_mappings'])
            if 'metric_definitions' in config:
                self.metric_definitions.update(config['metric_definitions'])
            if 'relationship_mappings' in config:
                self.relationship_mappings.update(config['relationship_mappings'])
            if 'synonyms' in config:
                self.synonyms.update(config['synonyms'])
            if 'column_semantics' in config:
                self.column_semantics.update(config['column_semantics'])
            if 'learned_patterns' in config:
                self.learned_patterns.update(config['learned_patterns'])
            
            logger.info(f"Loaded semantic configuration from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load semantic configuration: {e}")
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export the entire knowledge base"""
        return {
            'database_dialect': self.db_adapter.dialect.value,
            'tables_count': len(self.db_adapter.get_table_list()),
            'entity_mappings': self.entity_mappings,
            'metric_definitions': self.metric_definitions,
            'relationship_mappings': self.relationship_mappings,
            'synonyms': self.synonyms,
            'column_semantics': self.column_semantics,
            'learned_patterns': self.learned_patterns,
            'statistics': {
                'total_entities': len(self.entity_mappings),
                'total_metrics': len(self.metric_definitions),
                'total_relationships': sum(len(v) for v in self.relationship_mappings.values()),
                'total_semantic_mappings': len(self.column_semantics),
                'learned_queries': len(self.learned_patterns.get('learned_queries', []))
            }
        }
