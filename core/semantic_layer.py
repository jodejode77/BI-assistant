import logging
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticLayer:
    def __init__(self, config_file: Optional[str] = None):
        self.entity_mappings = {}
        self.metric_definitions = {}
        self.relationship_mappings = {}
        self.synonyms = {}
        
        self._init_home_credit_mappings()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def _init_home_credit_mappings(self):
        self.entity_mappings = {
            "client": "application_train",
            "customer": "application_train",
            "applicant": "application_train",
            "loan": "application_train",
            "application": "application_train",
            "credit": "application_train",
            "bureau": "bureau",
            "credit history": "bureau",
            "credit bureau": "bureau",
            "previous loans": "previous_application",
            "previous applications": "previous_application",
            "credit card": "credit_card_balance",
            "card balance": "credit_card_balance",
            "installments": "installments_payments",
            "payments": "installments_payments",
            "cash loans": "pos_cash_balance",
            "pos loans": "pos_cash_balance"
        }
        
        # Metric definitions
        self.metric_definitions = {
            "default_rate": {
                "description": "Percentage of loans that defaulted",
                "sql": "AVG(CASE WHEN TARGET = 1 THEN 1.0 ELSE 0.0 END) * 100",
                "table": "application_train"
            },
            "average_income": {
                "description": "Average annual income of clients",
                "sql": "AVG(AMT_INCOME_TOTAL)",
                "table": "application_train"
            },
            "average_loan_amount": {
                "description": "Average loan amount",
                "sql": "AVG(AMT_CREDIT)",
                "table": "application_train"
            },
            "income_to_loan_ratio": {
                "description": "Ratio of income to loan amount",
                "sql": "AVG(AMT_INCOME_TOTAL / NULLIF(AMT_CREDIT, 0))",
                "table": "application_train"
            },
            "average_age": {
                "description": "Average age of clients in years",
                "sql": "AVG(-DAYS_BIRTH / 365.25)",
                "table": "application_train"
            },
            "employment_duration": {
                "description": "Average employment duration in years",
                "sql": "AVG(-DAYS_EMPLOYED / 365.25)",
                "table": "application_train"
            }
        }
        
        # Relationship mappings
        self.relationship_mappings = {
            "application_train": {
                "bureau": "SK_ID_CURR",
                "previous_application": "SK_ID_CURR",
                "credit_card_balance": "SK_ID_CURR",
                "installments_payments": "SK_ID_CURR",
                "pos_cash_balance": "SK_ID_CURR"
            },
            "bureau": {
                "bureau_balance": "SK_ID_BUREAU"
            }
        }
        
        # Synonyms for common terms
        self.synonyms = {
            "income": ["salary", "earnings", "revenue", "amt_income_total"],
            "age": ["years old", "birth", "days_birth"],
            "gender": ["sex", "code_gender"],
            "married": ["marriage", "marital", "name_family_status"],
            "education": ["education level", "degree", "name_education_type"],
            "employment": ["job", "work", "employed", "days_employed"],
            "loan": ["credit", "amt_credit", "loan amount"],
            "default": ["target", "bad loan", "failed payment"],
            "housing": ["house", "home", "property", "real estate", "name_housing_type"]
        }
    
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
            "original_term": term,
            "table": None,
            "column": None,
            "metric": None,
            "related_terms": []
        }
        
        # Check entity mappings
        if term_lower in self.entity_mappings:
            result["table"] = self.entity_mappings[term_lower]
        
        # Check metric definitions
        if term_lower in self.metric_definitions:
            result["metric"] = self.metric_definitions[term_lower]
        
        # Check synonyms
        for key, synonyms in self.synonyms.items():
            if term_lower in [s.lower() for s in synonyms]:
                result["related_terms"].append(key)
                if key in self.metric_definitions:
                    result["metric"] = self.metric_definitions[key]
        
        # Check if term matches a column name pattern
        if "_" in term_lower:
            # Likely a column name
            result["column"] = term_lower.upper()
        
        return result
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance a natural language query with semantic understanding
        
        Args:
            query: Natural language query
            
        Returns:
            Enhanced query with semantic context
        """
        query_lower = query.lower()
        enhanced_parts = []
        
        # Check for entity mentions
        for entity, table in self.entity_mappings.items():
            if entity in query_lower:
                enhanced_parts.append(f"(referring to table: {table})")
        
        # Check for metric mentions
        for metric, definition in self.metric_definitions.items():
            if metric.replace("_", " ") in query_lower:
                enhanced_parts.append(
                    f"(metric: {metric} = {definition['description']})"
                )
        
        # Build enhanced query
        if enhanced_parts:
            return f"{query} {' '.join(enhanced_parts)}"
        return query
    
    def get_join_path(self, table1: str, table2: str) -> Optional[str]:
        """
        Get the JOIN path between two tables
        
        Args:
            table1: First table name
            table2: Second table name
            
        Returns:
            JOIN clause or None if no relationship exists
        """
        if table1 in self.relationship_mappings:
            if table2 in self.relationship_mappings[table1]:
                join_key = self.relationship_mappings[table1][table2]
                return f"JOIN {table2} ON {table1}.{join_key} = {table2}.{join_key}"
        
        # Check reverse relationship
        if table2 in self.relationship_mappings:
            if table1 in self.relationship_mappings[table2]:
                join_key = self.relationship_mappings[table2][table1]
                return f"JOIN {table2} ON {table1}.{join_key} = {table2}.{join_key}"
        
        return None
    
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
        
        for metric_name, metric_def in self.metric_definitions.items():
            # Check if metric is relevant to context
            metric_words = metric_name.replace("_", " ").split()
            if any(word in context_lower for word in metric_words):
                suggestions.append({
                    "name": metric_name,
                    "description": metric_def["description"],
                    "sql": metric_def["sql"],
                    "table": metric_def["table"]
                })
        
        return suggestions
    
    def load_config(self, config_file: str):
        """
        Load semantic configuration from file
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if "entity_mappings" in config:
                self.entity_mappings.update(config["entity_mappings"])
            if "metric_definitions" in config:
                self.metric_definitions.update(config["metric_definitions"])
            if "relationship_mappings" in config:
                self.relationship_mappings.update(config["relationship_mappings"])
            if "synonyms" in config:
                self.synonyms.update(config["synonyms"])
            
            logger.info(f"Loaded semantic configuration from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load semantic configuration: {e}")
    
    def save_config(self, config_file: str):
        """
        Save semantic configuration to file
        
        Args:
            config_file: Path to save configuration
        """
        config = {
            "entity_mappings": self.entity_mappings,
            "metric_definitions": self.metric_definitions,
            "relationship_mappings": self.relationship_mappings,
            "synonyms": self.synonyms
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved semantic configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save semantic configuration: {e}")
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Get full semantic context for a query
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with semantic context
        """
        context = {
            "enhanced_query": self.enhance_query(query),
            "detected_entities": [],
            "suggested_metrics": self.suggest_metrics(query),
            "potential_tables": set(),
            "potential_joins": []
        }
        
        query_lower = query.lower()
        
        # Detect entities and tables
        for entity, table in self.entity_mappings.items():
            if entity in query_lower:
                context["detected_entities"].append(entity)
                context["potential_tables"].add(table)
        
        # Suggest joins if multiple tables detected
        tables = list(context["potential_tables"])
        if len(tables) > 1:
            for i in range(len(tables)):
                for j in range(i + 1, len(tables)):
                    join = self.get_join_path(tables[i], tables[j])
                    if join:
                        context["potential_joins"].append(join)
        
        return context
