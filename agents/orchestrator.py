"""
Agent Orchestrator with strict pipeline:
1. Analysis (always first) - Understand query intent
2. SQL Generation with RAG (after analysis, with refined prompt)
3. Visualization (optional, if needed)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents.enhanced_sql_agent import EnhancedSQLAgent, EnhancedSQLQueryResult
from agents.analysis_agent import AnalysisAgent, AnalysisResult
from agents.visualization_agent import VisualizationAgent, VisualizationResult
from core.database_adapter import DatabaseAdapter
from core.llm_manager import LLMManager
from core.enhanced_rag_system import EnhancedRAGSystem
from core.dynamic_semantic_layer import DynamicSemanticLayer

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types for result classification"""
    QUERY = "query"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    COMPLEX = "complex"


@dataclass
class TaskResult:
    """Result of task processing"""
    success: bool
    task_type: TaskType
    sql_result: Optional[EnhancedSQLQueryResult] = None
    analysis_result: Optional[AnalysisResult] = None
    visualization_result: Optional[VisualizationResult] = None
    explanation: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgentOrchestrator:
    """
    Agent Orchestrator with strict pipeline execution
    
    Pipeline:
    1. Analysis (always) - Analyze query intent and refine prompt
    2. SQL Generation with RAG - Generate SQL using refined prompt and RAG context
    3. Visualization (optional) - Create visualization if needed
    """
    
    def __init__(
        self,
        database_adapter: DatabaseAdapter,
        llm_manager: LLMManager,
        rag_system: Optional[EnhancedRAGSystem] = None,
        semantic_layer: Optional[DynamicSemanticLayer] = None
    ):
        """
        Initialize orchestrator with new components
        
        Args:
            database_adapter: Database adapter instance
            llm_manager: LLM manager for all LLM operations
            rag_system: Optional enhanced RAG system (will be created if None)
            semantic_layer: Optional dynamic semantic layer (will be created if None)
        """
        self.db_adapter = database_adapter
        self.llm = llm_manager
        
        if semantic_layer is None:
            logger.info("[ORCHESTRATOR] Creating dynamic semantic layer...")
            self.semantic_layer = DynamicSemanticLayer(database_adapter)
        else:
            self.semantic_layer = semantic_layer
        
        if rag_system is None:
            logger.info("[ORCHESTRATOR] Creating enhanced RAG system...")
            self.rag_system = EnhancedRAGSystem(
                database_adapter=database_adapter,
                semantic_layer=self.semantic_layer
            )
        else:
            self.rag_system = rag_system
        
        self.sql_agent = EnhancedSQLAgent(
            database_adapter=database_adapter,
            llm_manager=llm_manager,
            rag_system=self.rag_system,
            semantic_layer=self.semantic_layer,
            enable_analysis=False
        )
        
        self.analysis_agent = AnalysisAgent(llm_manager, self.rag_system)
        self.visualization_agent = VisualizationAgent(llm_manager)
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orchestrator")
        
        logger.info("[ORCHESTRATOR] Initialized with strict pipeline: Analysis → SQL+RAG → Visualization")
    
    async def process_request(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Process user request through strict pipeline
        
        Pipeline:
        1. Analysis (always) - Understand query intent
        2. SQL Generation with RAG - Generate SQL with refined prompt
        3. Visualization (optional) - Create visualization if needed
        
        Args:
            user_input: User's natural language query
            context: Optional context from previous interactions
            
        Returns:
            TaskResult with all processing results
        """
        logger.info(f"[ORCHESTRATOR] Received request: '{user_input[:100]}...'")
        logger.info(f"[ORCHESTRATOR] Context: {context if context else 'none'}")
        
        try:
            processed_input, language_info = await asyncio.to_thread(
                self._prepare_input,
                user_input
            )
            
            if language_info.get("original_language") == "ru":
                logger.info(f"[ORCHESTRATOR] Russian detected, translated: '{processed_input[:100]}...'")
            else:
                logger.info("[ORCHESTRATOR] English input, no translation needed")
            
            logger.info("[ORCHESTRATOR] ===== STEP 1: ANALYSIS =====")
            analysis_result = await self._analyze_query_intent(processed_input, context)
            
            if not analysis_result:
                logger.error("[ORCHESTRATOR] Analysis failed, cannot proceed")
                return TaskResult(
                    success=False,
                    task_type=TaskType.QUERY,
                    error="Failed to analyze query intent"
                )
            
            refined_prompt = self._refine_prompt_from_analysis(processed_input, analysis_result)
            logger.info(f"[ORCHESTRATOR] Refined prompt: '{refined_prompt[:100]}...'")
            
            logger.info("[ORCHESTRATOR] ===== STEP 2: SQL GENERATION WITH RAG =====")
            sql_result = await asyncio.to_thread(
                self.sql_agent.process_question,
                refined_prompt,
                True
            )
            
            if not sql_result.success:
                logger.error(f"[ORCHESTRATOR] SQL generation failed: {sql_result.error}")
                return TaskResult(
                    success=False,
                    task_type=TaskType.QUERY,
                    sql_result=sql_result,
                    error=sql_result.error
                )
            
            logger.info(f"[ORCHESTRATOR] SQL executed: rows={sql_result.rows_returned}, time={sql_result.execution_time:.2f}s")
            
            visualization_result = None
            needs_visualization = self._needs_visualization(processed_input, analysis_result, sql_result)
            
            if needs_visualization and sql_result.data is not None and not sql_result.data.empty:
                logger.info("[ORCHESTRATOR] ===== STEP 3: VISUALIZATION =====")
                visualization_result = await asyncio.to_thread(
                    self.visualization_agent.create_visualization,
                    sql_result.data,
                    refined_prompt,
                    True
                )
                logger.info(f"[ORCHESTRATOR] Visualization created: success={visualization_result.success}, type={visualization_result.chart_type}")
            else:
                logger.info("[ORCHESTRATOR] Visualization skipped (not needed or no data)")
            
            if sql_result.data is not None and not sql_result.data.empty and self.analysis_agent:
                logger.info("[ORCHESTRATOR] Analyzing SQL results with AnalysisAgent...")
                try:
                    result_analysis = await asyncio.to_thread(
                        self.analysis_agent.analyze_results,
                        refined_prompt,
                        sql_result.query,
                        sql_result.data,
                        json.dumps(context) if context else None
                    )
                    analysis_result = result_analysis
                    logger.info(f"[ORCHESTRATOR] Result analysis complete: insights={len(analysis_result.insights)}")
                except Exception as e:
                    logger.warning(f"[ORCHESTRATOR] Failed to analyze results: {e}")
            
            explanation = self._generate_explanation(
                processed_input,
                sql_result,
                analysis_result,
                visualization_result
            )
            
            task_type = self._determine_task_type(needs_visualization, analysis_result)
            
            result = TaskResult(
                success=True,
                task_type=task_type,
                sql_result=sql_result,
                analysis_result=analysis_result,
                visualization_result=visualization_result,
                explanation=explanation,
                metadata={
                    **language_info,
                    "refined_prompt": refined_prompt,
                    "pipeline_steps": ["analysis", "sql_generation", "visualization" if needs_visualization else None]
                }
            )
            
            if language_info.get("original_language") == "ru":
                logger.info("[ORCHESTRATOR] Localizing result to Russian...")
                result = self._localize_result(result)
            
            logger.info(f"[ORCHESTRATOR] Pipeline completed: success={result.success}, task_type={result.task_type.value}")
            return result
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] ERROR processing request: {e}", exc_info=True)
            return TaskResult(
                success=False,
                task_type=TaskType.QUERY,
                error=str(e)
            )
    
    def _prepare_input(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect language and translate Russian inputs
        
        Args:
            user_input: User input text
            
        Returns:
            Tuple of (processed_input, language_info)
        """
        metadata = {"original_language": "en"}
        
        if re.search(r"[А-Яа-яЁё]", user_input):
            metadata["original_language"] = "ru"
            translated = self.llm.translate_text(user_input, target_language="English")
            metadata["translated_input"] = translated
            return translated, metadata
        
        return user_input, metadata
    
    async def _analyze_query_intent(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AnalysisResult]:
        """
        Analyze query intent (STEP 1)
        
        This step:
        - Understands what the user wants
        - Identifies entities, metrics, and relationships
        - Determines if visualization is needed
        - Refines the query for better SQL generation
        
        Args:
            user_input: User's natural language query
            context: Optional context
            
        Returns:
            AnalysisResult with query understanding
        """
        logger.info("[ORCHESTRATOR] [ANALYSIS] Analyzing query intent...")
        
        try:
            analysis_prompt = f"""Analyze this user query and provide structured understanding:

User Query: {user_input}

Provide analysis in the following format:
1. Query Type: [simple/aggregation/join/complex]
2. Intent: [retrieve/calculate/compare/trend/visualize]
3. Key Entities: [list of entities mentioned]
4. Metrics: [list of metrics to calculate]
5. Needs Visualization: [yes/no]
6. Refined Query: [improved version of the query for SQL generation]

Context: {json.dumps(context) if context else "None"}

Return only the analysis in the format above."""
            
            logger.debug("[ORCHESTRATOR] [ANALYSIS] Requesting LLM analysis...")
            def _call_llm():
                return self.llm.generate(
                    prompt=analysis_prompt,
                    temperature=0.3,
                    max_tokens=500
                )
            response = await asyncio.to_thread(_call_llm)
            
            analysis_text = response.content
            logger.debug(f"[ORCHESTRATOR] [ANALYSIS] LLM response: {analysis_text[:200]}...")
            
            analysis_result = AnalysisResult(
                interpretation=analysis_text,
                insights=self._extract_insights_from_analysis(analysis_text),
                statistics={},  # Empty statistics for query intent analysis
                recommendations=[],
                confidence_score=0.8
            )
            
            logger.info("[ORCHESTRATOR] [ANALYSIS] Analysis complete")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] [ANALYSIS] Error: {e}", exc_info=True)
            return None
    
    def _extract_insights_from_analysis(self, analysis_text: str) -> List[str]:
        """Extract insights from analysis text"""
        insights = []
        lines = analysis_text.split('\n')
        for line in lines:
            if ':' in line and any(keyword in line.lower() for keyword in ['entity', 'metric', 'intent', 'type']):
                insights.append(line.strip())
        return insights[:5]
    
    def _refine_prompt_from_analysis(
        self,
        original_prompt: str,
        analysis_result: AnalysisResult
    ) -> str:
        """
        Refine prompt based on analysis results
        
        Args:
            original_prompt: Original user query
            analysis_result: Analysis results
            
            Returns:
            Refined prompt for SQL generation
        """
        interpretation = analysis_result.interpretation
        
        if "Refined Query:" in interpretation:
            lines = interpretation.split('\n')
            for line in lines:
                if "Refined Query:" in line:
                    refined = line.split("Refined Query:")[-1].strip()
                    if refined:
                        logger.info(f"[ORCHESTRATOR] Using refined query from analysis: {refined[:100]}...")
                        return refined
        
        insights_text = "\n".join(analysis_result.insights[:3])
        refined = f"{original_prompt}\n\nContext: {insights_text}"
        
        return refined
    
    def _needs_visualization(
        self,
        user_input: str,
        analysis_result: Optional[AnalysisResult],
        sql_result: EnhancedSQLQueryResult
    ) -> bool:
        """
        Determine if visualization is needed
        
        Args:
            user_input: Original user query
            analysis_result: Analysis results
            sql_result: SQL execution results
            
        Returns:
            True if visualization is needed
        """
        viz_keywords = ["chart", "graph", "plot", "visualize", "draw", "display", "show", "диаграмма", "график", "визуализируй"]
        user_wants_viz = any(kw in user_input.lower() for kw in viz_keywords)
        
        if user_wants_viz:
            sql_upper = sql_result.query.upper() if sql_result.query else ""
            is_aggregated = any(kw in sql_upper for kw in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(', 'GROUP BY'])
            has_group_by = 'GROUP BY' in sql_upper
            
            if is_aggregated and len(sql_result.data) == 1 and not has_group_by:
                return False
            else:
                return True
        
        if analysis_result and "visualize" in analysis_result.interpretation.lower():
            return True
        
        if sql_result.visualization_ready:
            return True
        
        if sql_result.data is not None and not sql_result.data.empty:
            sql_upper = sql_result.query.upper() if sql_result.query else ""
            is_aggregated = any(kw in sql_upper for kw in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(', 'GROUP BY'])
            has_group_by = 'GROUP BY' in sql_upper
            
            if is_aggregated and has_group_by:
                return True
            elif len(sql_result.data) > 1 and len(sql_result.data.columns) <= 3:
                return True
            elif is_aggregated and len(sql_result.data) == 1 and not has_group_by:
                return False
        
        return False
    
    def _generate_explanation(
        self,
        user_input: str,
        sql_result: EnhancedSQLQueryResult,
        analysis_result: Optional[AnalysisResult],
        visualization_result: Optional[VisualizationResult]
    ) -> str:
        """
        Generate explanation for the results
        
        Args:
            user_input: Original user query
            sql_result: SQL execution results
            analysis_result: Analysis results
            visualization_result: Visualization results
            
        Returns:
            Explanation text
        """
        sql_upper = sql_result.query.upper() if sql_result.query else ""
        is_aggregated = any(kw in sql_upper for kw in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(', 'GROUP BY'])
        has_group_by = 'GROUP BY' in sql_upper
        
        explanation = f"<b>Результаты запроса:</b>\n"
        
        if is_aggregated and not has_group_by:
            explanation += f"Рассчитано агрегированное значение.\n"
        else:
            explanation += f"Найдено {sql_result.rows_returned} запис{'ей' if sql_result.rows_returned != 1 else 'ь'}.\n"
        
        explanation += f"Время выполнения: {sql_result.execution_time:.2f}с\n\n"
        
        if analysis_result:
            interpretation = analysis_result.interpretation
            if interpretation:
                interpretation = self._clean_markdown(interpretation)
            if len(interpretation) > 500:
                interpretation = interpretation[:500] + "..."
            explanation += f"<b>Анализ:</b>\n{interpretation}\n\n"
        
        if sql_result.data is not None and not sql_result.data.empty:
            if is_aggregated and len(sql_result.data) == 1 and not has_group_by:
                explanation += "<b>Результат:</b>\n"
                for col in sql_result.data.columns:
                    value = sql_result.data[col].iloc[0]
                    if pd.api.types.is_numeric_dtype(sql_result.data[col]):
                        formatted_value = f"{value:,.2f}".replace(",", " ") if isinstance(value, (int, float)) else str(value)
                        explanation += f"• {col}: {formatted_value}\n"
                    else:
                        explanation += f"• {col}: {value}\n"
            else:
                explanation += "<b>Сводка:</b>\n"
                numeric_cols = sql_result.data.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    for col in numeric_cols[:3]:
                        avg_value = sql_result.data[col].mean()
                        formatted_avg = f"{avg_value:,.2f}".replace(",", " ") if isinstance(avg_value, (int, float)) else str(avg_value)
                        explanation += f"• {col}: среднее={formatted_avg}\n"
        
        user_wants_viz = any(kw in user_input.lower() for kw in ["chart", "graph", "plot", "visualize", "draw", "display", "show", "диаграмма", "график", "визуализируй"])
        sql_upper = sql_result.query.upper() if sql_result.query else ""
        is_simple_aggregate = (
            sql_result.data is not None and 
            len(sql_result.data) == 1 and 
            any(kw in sql_upper for kw in ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(']) and
            'GROUP BY' not in sql_upper
        )
        
        if visualization_result and visualization_result.success:
            explanation += f"\n<b>Визуализация:</b> создана {visualization_result.chart_type}.\n"
        elif user_wants_viz and is_simple_aggregate:
            explanation += f"\n<b>Примечание:</b> Визуализация для одного агрегированного значения не создана, так как она не будет информативной. Для визуализации используйте запросы с группировкой (например, 'средний доход по категориям').\n"
        
        return explanation
    
    def _determine_task_type(
        self,
        has_visualization: bool,
        analysis_result: Optional[AnalysisResult]
    ) -> TaskType:
        """Determine task type based on results"""
        if has_visualization:
            return TaskType.VISUALIZATION
        elif analysis_result and len(analysis_result.insights) > 3:
            return TaskType.ANALYSIS
        else:
            return TaskType.QUERY
    
    def _localize_result(self, result: TaskResult) -> TaskResult:
        """
        Translate result back to Russian if needed
        
        Args:
            result: TaskResult to localize
            
        Returns:
            Localized TaskResult
        """
        def _translate_text(value: Optional[str]) -> Optional[str]:
            return self.llm.translate_text(value, target_language="Russian") if value else value
        
        def _translate_list(values: Optional[List[str]]) -> Optional[List[str]]:
            if not values:
                return values
            return [_translate_text(item) for item in values]
        
        result.explanation = _translate_text(result.explanation)
        
        if result.analysis_result:
            result.analysis_result.interpretation = _translate_text(
                result.analysis_result.interpretation
            )
            result.analysis_result.insights = _translate_list(
                result.analysis_result.insights
            ) or []
            result.analysis_result.recommendations = _translate_list(
                result.analysis_result.recommendations
            ) or []
        
        if result.visualization_result and getattr(result.visualization_result, "description", None):
            result.visualization_result.description = _translate_text(
                result.visualization_result.description
            )
        
        return result
    
    def _clean_markdown(self, text: str) -> str:
        """Remove escaped markdown characters from LLM responses"""
        if not text:
            return text
        
        text = re.sub(r'\\\*\\\*', '**', text)
        text = re.sub(r'\\_\\_', '__', text)
        text = re.sub(r'\\`\\`\\`', '```', text)
        text = re.sub(r'\\`\\`', '``', text)
        text = re.sub(r'\\{2,}([*_`])', r'\1', text)
        
        markdown_chars = r'[*_`\[\]()~>#+\-=|{}.!]'
        text = re.sub(rf'\\({markdown_chars})', r'\1', text)
        
        replacements = [
            ('\\.', '.'), ('\\-', '-'), ('\\=', '='), ('\\(', '('), ('\\)', ')'),
            ('\\#', '#'), ('\\*', '*'), ('\\_', '_'), ('\\[', '['), ('\\]', ']'),
            ('\\n', '\n'), ('\\t', '\t'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        
        for _ in range(3):
            text = re.sub(r'\\([*_`\[\]()~>#+\-=|{}.!])', r'\1', text)
        
        return text
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents
        
        Returns:
            Dictionary with status information
        """
        return {
            "orchestrator": "active",
            "pipeline": "strict",
            "async": True,
            "agents": {
                "sql_agent": {
                    "status": "active",
                    "type": "EnhancedSQLAgent"
                },
                "analysis_agent": {
                    "status": "active"
                },
                "visualization_agent": {
                    "status": "active"
                }
            },
            "rag_system": {
                "status": "active",
                "type": "EnhancedRAGSystem"
            },
            "semantic_layer": {
                "status": "active",
                "type": "DynamicSemanticLayer"
            },
            "database": {
                "connected": True,
                "dialect": self.db_adapter.dialect.value,
                "tables": len(self.db_adapter.get_table_list())
            }
        }
    
    def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("[ORCHESTRATOR] Shutting down...")
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("[ORCHESTRATOR] Thread pool executor shut down")
