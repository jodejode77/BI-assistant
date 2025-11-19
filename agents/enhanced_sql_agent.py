"""
Enhanced SQL Agent with universal database support and retry logic
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import time
from dataclasses import dataclass
from enum import Enum

from core.database_adapter import DatabaseAdapter, create_database_adapter
from core.llm_manager import LLMManager
from core.enhanced_rag_system import EnhancedRAGSystem
from core.dynamic_semantic_layer import DynamicSemanticLayer
from core.sql_generator import (
    DialectSpecificSQLGenerator,
    SQLGenerationStrategy,
    GeneratedSQL
)

logger = logging.getLogger(__name__)


class QueryProcessingStage(Enum):
    """Stages of query processing"""
    ANALYSIS = "analysis"
    DATA_SELECTION = "data_selection"
    SQL_GENERATION = "sql_generation"
    EXECUTION = "execution"
    VISUALIZATION_PREP = "visualization_prep"


@dataclass
class QueryAnalysisResult:
    """Result of query analysis"""
    query_type: str  # 'simple', 'aggregation', 'join', 'complex'
    intent: str  # 'retrieve', 'calculate', 'compare', 'trend'
    entities: List[str]
    metrics: List[str]
    requires_visualization: bool
    complexity_score: float
    suggested_strategy: SQLGenerationStrategy


@dataclass
class EnhancedSQLQueryResult:
    """Enhanced result with full processing metadata"""
    success: bool
    query: str
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    rows_returned: int = 0
    generation_metadata: Optional[GeneratedSQL] = None
    analysis_result: Optional[QueryAnalysisResult] = None
    retry_count: int = 0
    processing_stages: Dict[str, float] = None
    optimization_suggestions: List[str] = None
    visualization_ready: bool = False


class EnhancedSQLAgent:
    """
    Enhanced SQL Agent with universal database support
    
    Processing Pipeline:
    1. Analysis (optional) - Understand query intent
    2. Data Selection - Choose relevant data through RAG and semantic layer
    3. SQL Generation - Generate with LLM and retries
    4. Execution - Execute with monitoring
    5. Visualization Prep - Prepare data for visualization
    """
    
    def __init__(
        self,
        database_adapter: DatabaseAdapter,
        llm_manager: LLMManager,
        rag_system: Optional[EnhancedRAGSystem] = None,
        semantic_layer: Optional[DynamicSemanticLayer] = None,
        enable_analysis: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize Enhanced SQL Agent
        
        Args:
            database_adapter: Database adapter instance
            llm_manager: LLM manager for query generation
            rag_system: Optional RAG system (will be created if None)
            semantic_layer: Optional semantic layer (will be created if None)
            enable_analysis: Whether to enable query analysis stage
            max_retries: Maximum retry attempts
        """
        self.db_adapter = database_adapter
        self.llm = llm_manager
        self.enable_analysis = enable_analysis
        self.max_retries = max_retries
        
        # Initialize semantic layer if not provided
        if semantic_layer is None:
            logger.info("Creating dynamic semantic layer...")
            self.semantic_layer = DynamicSemanticLayer(database_adapter)
        else:
            self.semantic_layer = semantic_layer
        
        # Initialize RAG system if not provided
        if rag_system is None:
            logger.info("Creating enhanced RAG system...")
            self.rag_system = EnhancedRAGSystem(
                database_adapter=database_adapter,
                semantic_layer=self.semantic_layer
            )
        else:
            self.rag_system = rag_system
        
        # Initialize SQL generator
        self.sql_generator = DialectSpecificSQLGenerator(
            database_adapter=database_adapter,
            semantic_layer=self.semantic_layer,
            rag_system=self.rag_system,
            llm_manager=llm_manager,
            max_retries=max_retries
        )
        
        # Query history for learning
        self.query_history = []
        
        logger.info(f"[ENHANCED_SQL_AGENT] Initialized for {database_adapter.dialect.value}")
    
    def process_question(
        self,
        question: str,
        skip_analysis: bool = False,
        force_strategy: Optional[SQLGenerationStrategy] = None
    ) -> EnhancedSQLQueryResult:
        """
        Process a natural language question through the full pipeline
        
        Args:
            question: Natural language question
            skip_analysis: Skip the analysis stage
            force_strategy: Force a specific SQL generation strategy
            
        Returns:
            EnhancedSQLQueryResult with query results and metadata
        """
        logger.info(f"[ENHANCED_SQL_AGENT] Processing: '{question[:100]}...'")
        
        processing_stages = {}
        start_time = time.time()
        
        try:
            # Stage 1: Analysis (optional)
            analysis_result = None
            if self.enable_analysis and not skip_analysis:
                stage_start = time.time()
                logger.info("[ENHANCED_SQL_AGENT] Stage 1: Analyzing query...")
                analysis_result = self._analyze_query(question)
                processing_stages[QueryProcessingStage.ANALYSIS.value] = time.time() - stage_start
                logger.info(f"[ENHANCED_SQL_AGENT] Analysis complete: type={analysis_result.query_type}, intent={analysis_result.intent}")
            
            # Stage 2: Data Selection through RAG and Semantic Layer
            stage_start = time.time()
            logger.info("[ENHANCED_SQL_AGENT] Stage 2: Selecting relevant data...")
            
            # Get semantic context
            semantic_context = self.semantic_layer.enhance_query(question)
            logger.debug(f"[ENHANCED_SQL_AGENT] Semantic context: {len(semantic_context.get('suggested_tables', []))} tables")
            
            # Get schema context from RAG
            schema_context = self.rag_system.get_schema_context(
                question,
                semantic_context.get('suggested_tables', [])
            )
            logger.debug(f"[ENHANCED_SQL_AGENT] Schema context: {len(schema_context)} chars")
            
            # Get similar queries
            similar_queries = self.rag_system.get_similar_queries(question)
            logger.debug(f"[ENHANCED_SQL_AGENT] Found {len(similar_queries)} similar queries")
            
            processing_stages[QueryProcessingStage.DATA_SELECTION.value] = time.time() - stage_start
            
            # Stage 3: SQL Generation with retries
            stage_start = time.time()
            logger.info("[ENHANCED_SQL_AGENT] Stage 3: Generating SQL with LLM...")
            
            # Determine strategy
            if force_strategy:
                strategy = force_strategy
            elif analysis_result:
                strategy = analysis_result.suggested_strategy
            else:
                strategy = SQLGenerationStrategy.FULL_CONTEXT
            
            # Generate SQL
            generated_sql = self.sql_generator.generate_sql(question, strategy)
            
            processing_stages[QueryProcessingStage.SQL_GENERATION.value] = time.time() - stage_start
            logger.info(f"[ENHANCED_SQL_AGENT] SQL generated: confidence={generated_sql.confidence:.2f}, retries={generated_sql.retry_count}")
            
            # Stage 4: Execution
            stage_start = time.time()
            logger.info("[ENHANCED_SQL_AGENT] Stage 4: Executing SQL query...")
            
            # Validate before execution
            is_valid, optimized_sql, suggestions = self.sql_generator.validate_and_optimize(generated_sql.query)
            
            if not is_valid:
                return EnhancedSQLQueryResult(
                    success=False,
                    query=generated_sql.query,
                    error=f"Query validation failed: {suggestions[0] if suggestions else 'Unknown error'}",
                    analysis_result=analysis_result,
                    generation_metadata=generated_sql,
                    processing_stages=processing_stages
                )
            
            # Execute query
            exec_start = time.time()
            data = self.db_adapter.execute_query(optimized_sql)
            execution_time = time.time() - exec_start
            
            processing_stages[QueryProcessingStage.EXECUTION.value] = time.time() - stage_start
            logger.info(f"[ENHANCED_SQL_AGENT] Query executed: {len(data)} rows in {execution_time:.2f}s")
            
            # Stage 5: Visualization Preparation
            stage_start = time.time()
            visualization_ready = self._prepare_for_visualization(data, analysis_result)
            processing_stages[QueryProcessingStage.VISUALIZATION_PREP.value] = time.time() - stage_start
            
            # Learn from successful execution
            self._learn_from_execution(
                question=question,
                sql=optimized_sql,
                execution_time=execution_time,
                rows_returned=len(data)
            )
            
            # Store in history
            self.query_history.append({
                "question": question,
                "sql": optimized_sql,
                "rows": len(data),
                "time": execution_time,
                "strategy": strategy.value
            })
            
            # Total processing time
            total_time = time.time() - start_time
            logger.info(f"[ENHANCED_SQL_AGENT] Total processing time: {total_time:.2f}s")
            
            return EnhancedSQLQueryResult(
                success=True,
                query=optimized_sql,
                data=data,
                execution_time=execution_time,
                rows_returned=len(data),
                generation_metadata=generated_sql,
                analysis_result=analysis_result,
                retry_count=generated_sql.retry_count,
                processing_stages=processing_stages,
                optimization_suggestions=suggestions,
                visualization_ready=visualization_ready
            )
            
        except Exception as e:
            logger.error(f"[ENHANCED_SQL_AGENT] Error processing question: {e}", exc_info=True)
            return EnhancedSQLQueryResult(
                success=False,
                query="",
                error=str(e),
                processing_stages=processing_stages
            )
    
    def _analyze_query(self, question: str) -> QueryAnalysisResult:
        """
        Analyze query to understand intent and complexity
        
        Args:
            question: Natural language question
            
        Returns:
            QueryAnalysisResult
        """
        question_lower = question.lower()
        
        # Detect query type
        if any(word in question_lower for word in ['average', 'sum', 'count', 'total', 'maximum', 'minimum']):
            query_type = 'aggregation'
        elif any(word in question_lower for word in ['join', 'combine', 'relate', 'between tables']):
            query_type = 'join'
        elif any(word in question_lower for word in ['trend', 'over time', 'by month', 'by year']):
            query_type = 'complex'
        else:
            query_type = 'simple'
        
        # Detect intent
        if any(word in question_lower for word in ['calculate', 'compute', 'what is the']):
            intent = 'calculate'
        elif any(word in question_lower for word in ['compare', 'versus', 'difference']):
            intent = 'compare'
        elif any(word in question_lower for word in ['trend', 'pattern', 'over time']):
            intent = 'trend'
        else:
            intent = 'retrieve'
        
        # Get entities and metrics from semantic layer
        semantic_context = self.semantic_layer.enhance_query(question)
        entities = semantic_context.get('detected_entities', [])
        metrics = [m['name'] for m in semantic_context.get('suggested_metrics', [])]
        
        # Determine if visualization is needed
        requires_visualization = intent in ['compare', 'trend'] or query_type in ['aggregation', 'complex']
        
        # Calculate complexity score
        complexity_score = 0.0
        if query_type == 'simple':
            complexity_score = 0.3
        elif query_type == 'aggregation':
            complexity_score = 0.5
        elif query_type == 'join':
            complexity_score = 0.7
        else:
            complexity_score = 0.9
        
        # Suggest strategy based on complexity
        if complexity_score < 0.4:
            suggested_strategy = SQLGenerationStrategy.DIRECT
        elif complexity_score < 0.6:
            suggested_strategy = SQLGenerationStrategy.RAG_ENHANCED
        elif complexity_score < 0.8:
            suggested_strategy = SQLGenerationStrategy.SEMANTIC_GUIDED
        else:
            suggested_strategy = SQLGenerationStrategy.FULL_CONTEXT
        
        return QueryAnalysisResult(
            query_type=query_type,
            intent=intent,
            entities=entities,
            metrics=metrics,
            requires_visualization=requires_visualization,
            complexity_score=complexity_score,
            suggested_strategy=suggested_strategy
        )
    
    def _prepare_for_visualization(
        self,
        data: pd.DataFrame,
        analysis: Optional[QueryAnalysisResult]
    ) -> bool:
        """
        Prepare data for visualization
        
        Args:
            data: Query result data
            analysis: Query analysis result
            
        Returns:
            Whether data is ready for visualization
        """
        if data.empty:
            return False
        
        # Check if we have suitable data for visualization
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Basic visualization readiness check
        can_visualize = False
        
        if analysis and analysis.requires_visualization:
            if analysis.intent == 'trend' and datetime_columns:
                can_visualize = True
            elif analysis.intent == 'compare' and categorical_columns and numeric_columns:
                can_visualize = True
            elif analysis.query_type == 'aggregation' and numeric_columns:
                can_visualize = True
        
        # Simple heuristic if no analysis
        elif len(numeric_columns) > 0 and (len(categorical_columns) > 0 or len(datetime_columns) > 0):
            can_visualize = True
        
        return can_visualize
    
    def _learn_from_execution(
        self,
        question: str,
        sql: str,
        execution_time: float,
        rows_returned: int
    ):
        """Learn from successful query execution"""
        # Update RAG system
        self.rag_system.learn_from_query(
            question=question,
            sql=sql,
            execution_time=execution_time,
            rows_returned=rows_returned
        )
        
        # Update semantic layer
        self.semantic_layer.learn_from_query(
            question=question,
            sql=sql,
            feedback="successful"
        )
        
        logger.debug(f"[ENHANCED_SQL_AGENT] Learned from execution: {rows_returned} rows in {execution_time:.2f}s")
    
    def explain_query(self, sql: str) -> str:
        """
        Explain a SQL query in simple terms
        
        Args:
            sql: SQL query to explain
            
        Returns:
            Human-readable explanation
        """
        return self.sql_generator.explain_query(sql)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if not self.query_history:
            return {
                "total_queries": 0,
                "database_dialect": self.db_adapter.dialect.value
            }
        
        total_queries = len(self.query_history)
        avg_time = sum(q["time"] for q in self.query_history) / total_queries
        avg_rows = sum(q["rows"] for q in self.query_history) / total_queries
        
        # Strategy usage
        strategy_counts = {}
        for q in self.query_history:
            strategy = q.get("strategy", "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "total_queries": total_queries,
            "average_execution_time": round(avg_time, 3),
            "average_rows_returned": round(avg_rows, 1),
            "strategy_usage": strategy_counts,
            "database_dialect": self.db_adapter.dialect.value,
            "rag_documents": self.rag_system.get_database_summary()["total_documents"],
            "semantic_mappings": len(self.semantic_layer.entity_mappings),
            "last_query": self.query_history[-1]["sql"] if self.query_history else None
        }
    
    def export_knowledge(self, output_file: str):
        """Export accumulated knowledge"""
        import json
        
        knowledge = {
            "agent_stats": self.get_agent_stats(),
            "query_history": self.query_history[-100:],  # Last 100 queries
            "database_summary": self.rag_system.get_database_summary(),
            "semantic_knowledge": self.semantic_layer.export_knowledge_base()
        }
        
        with open(output_file, 'w') as f:
            json.dump(knowledge, f, indent=2, default=str)
        
        logger.info(f"[ENHANCED_SQL_AGENT] Exported knowledge to {output_file}")


def create_universal_sql_agent(
    connection_url: str,
    llm_manager: LLMManager,
    **kwargs
) -> EnhancedSQLAgent:
    """
    Factory function to create a universal SQL agent for any database
    
    Args:
        connection_url: Database connection URL
        llm_manager: LLM manager instance
        **kwargs: Additional parameters for the agent
        
    Returns:
        EnhancedSQLAgent configured for the database
    """
    # Create appropriate database adapter
    db_adapter = create_database_adapter(connection_url)
    
    # Create agent
    agent = EnhancedSQLAgent(
        database_adapter=db_adapter,
        llm_manager=llm_manager,
        **kwargs
    )
    
    return agent
