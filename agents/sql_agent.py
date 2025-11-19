import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import re
from dataclasses import dataclass

from core.database import DatabaseManager
from core.llm_manager import LLMManager
from core.rag_system import DatabaseRAG
from core.semantic_layer import SemanticLayer

logger = logging.getLogger(__name__)


@dataclass
class SQLQueryResult:
    success: bool
    query: str
    data: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    rows_returned: int = 0


class SQLAgent:
    """
    Agent responsible for SQL query generation and execution
    """
    
    def __init__(
        self,
        database: DatabaseManager,
        llm_manager: LLMManager,
        rag_system: DatabaseRAG,
        semantic_layer: SemanticLayer
    ):
        """
        Initialize SQL Agent
        
        Args:
            database: Database manager instance
            llm_manager: LLM manager for query generation
            rag_system: RAG system for context retrieval
            semantic_layer: Semantic layer for term translation
        """
        self.database = database
        self.llm = llm_manager
        self.rag = rag_system
        self.semantic = semantic_layer
        
        self.query_history = []
        
        logger.info("[SQL_AGENT] Инициализирован SQL агент")
    
    def process_question(self, question: str) -> SQLQueryResult:
        """
        Process a natural language question and return query results
        
        Args:
            question: Natural language question
            
        Returns:
            SQLQueryResult with query and data
        """
        try:
            logger.info(f"[SQL_AGENT] Обработка вопроса: '{question[:80]}...'")
            
            logger.debug("[SQL_AGENT] → Получение семантического контекста")
            semantic_context = self.semantic.get_context_for_query(question)
            logger.debug(f"[SQL_AGENT] Семантический контекст: entities={len(semantic_context.get('detected_entities', []))}, tables={len(semantic_context.get('potential_tables', []))}")
            
            logger.debug("[SQL_AGENT] → Получение контекста схемы БД через RAG")
            schema_context = self.rag.get_relevant_schema_context(question)
            logger.debug(f"[SQL_AGENT] Контекст схемы получен: {len(schema_context)} символов")
            
            logger.debug("[SQL_AGENT] → Поиск похожих запросов через RAG")
            similar_queries = self.rag.get_similar_queries(question)
            logger.debug(f"[SQL_AGENT] Найдено похожих запросов: {len(similar_queries)}")
            
            logger.info("[SQL_AGENT] → Генерация SQL запроса через LLM")
            sql_query = self._generate_sql_query(
                question,
                semantic_context,
                schema_context,
                similar_queries
            )
            logger.info(f"[SQL_AGENT] SQL запрос сгенерирован: '{sql_query[:100]}...'")
            
            logger.debug("[SQL_AGENT] → Валидация SQL запроса")
            is_valid, error = self.database.validate_query(sql_query)
            if not is_valid:
                logger.warning(f"[SQL_AGENT] Валидация не пройдена: {error}")
                return SQLQueryResult(
                    success=False,
                    query=sql_query,
                    error=f"Query validation failed: {error}"
                )
            logger.debug("[SQL_AGENT] Валидация пройдена успешно")
            
            logger.info("[SQL_AGENT] → Выполнение SQL запроса в БД")
            import time
            start_time = time.time()
            data = self.database.execute_query(sql_query)
            execution_time = time.time() - start_time
            logger.info(f"[SQL_AGENT] Запрос выполнен: rows={len(data)}, time={execution_time:.2f}s")
            
            self.query_history.append({
                "question": question,
                "sql": sql_query,
                "rows": len(data),
                "time": execution_time
            })
            
            logger.info(f"[SQL_AGENT] Запрос успешно обработан: rows={len(data)}, time={execution_time:.2f}s")
            return SQLQueryResult(
                success=True,
                query=sql_query,
                data=data,
                execution_time=execution_time,
                rows_returned=len(data)
            )
            
        except Exception as e:
            logger.error(f"[SQL_AGENT] ОШИБКА при обработке вопроса: {e}", exc_info=True)
            return SQLQueryResult(
                success=False,
                query="",
                error=str(e)
            )
    
    def _generate_sql_query(
        self,
        question: str,
        semantic_context: Dict[str, Any],
        schema_context: str,
        similar_queries: List[Dict[str, str]]
    ) -> str:
        """
        Generate SQL query using LLM with context
        
        Args:
            question: User question
            semantic_context: Semantic layer context
            schema_context: Database schema context
            similar_queries: Similar query examples
            
        Returns:
            Generated SQL query
        """
        prompt = self._build_generation_prompt(
            question,
            semantic_context,
            schema_context,
            similar_queries
        )
        
        tables_info = self._get_detailed_tables_info(
            list(semantic_context.get("potential_tables", []))
        )
        logger.debug(f"[SQL_AGENT] Информация о таблицах подготовлена: {len(tables_info)} символов")
        
        logger.debug("[SQL_AGENT] → Вызов LLM для генерации SQL")
        sql = self.llm.generate_sql(
            question=question,
            schema_context=tables_info,
            examples=similar_queries[:3] if similar_queries else None
        )
        logger.debug(f"[SQL_AGENT] LLM вернул SQL: '{sql[:100]}...'")
        
        logger.debug("[SQL_AGENT] → Постобработка SQL запроса")
        sql = self._post_process_query(sql)
        logger.debug(f"[SQL_AGENT] SQL после постобработки: '{sql[:100]}...'")
        
        return sql
    
    def _build_generation_prompt(
        self,
        question: str,
        semantic_context: Dict[str, Any],
        schema_context: str,
        similar_queries: List[Dict[str, str]]
    ) -> str:
        prompt_parts = [f"Question: {question}"]
        
        if semantic_context.get("detected_entities"):
            prompt_parts.append(f"Detected entities: {', '.join(semantic_context['detected_entities'])}")
        
        if semantic_context.get("suggested_metrics"):
            metrics_str = "\n".join([
                f"- {m['name']}: {m['description']}"
                for m in semantic_context['suggested_metrics'][:3]
            ])
            prompt_parts.append(f"Relevant metrics:\n{metrics_str}")
        
        if schema_context:
            prompt_parts.append(f"Relevant schema:\n{schema_context}")
        
        if similar_queries:
            examples_str = "\n".join([
                f"Q: {ex['question']}\nSQL: {ex['sql']}"
                for ex in similar_queries[:2]
            ])
            prompt_parts.append(f"Similar examples:\n{examples_str}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_detailed_tables_info(self, tables: List[str]) -> str:
        if not tables:
            tables = ["application_train", "bureau", "previous_application"][:2]
        
        info_parts = []
        for table in tables:
            schema = self.database.get_table_schema(table)
            if schema:
                columns_info = []
                for col in schema[:10]:
                    columns_info.append(f"  - {col['name']} ({col['type']})")
                
                info_parts.append(f"Table: {table}\nColumns:\n" + "\n".join(columns_info))
        
        return "\n\n".join(info_parts)
    
    def _post_process_query(self, sql: str) -> str:
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        if "limit" not in sql.lower():
            sql = f"{sql} LIMIT 100"
        
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.replace(";", "")
        
        return sql.strip()
    
    def explain_query(self, sql: str) -> str:
        logger.info(f"[SQL_AGENT] Объяснение SQL запроса: '{sql[:80]}...'")
        logger.debug("[SQL_AGENT] → Генерация объяснения через LLM")
        prompt = f"""Explain this SQL query in simple terms:

```sql
{sql}
```

Provide a clear, concise explanation of:
1. What data is being retrieved
2. Which tables are being used
3. Any filters or conditions applied
4. How the results are organized"""
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=300
        )
        
        logger.info(f"[SQL_AGENT] Объяснение сгенерировано: {len(response.content)} символов")
        return response.content
    
    def suggest_optimizations(self, sql: str) -> List[str]:
        """
        Suggest optimizations for a SQL query
        
        Args:
            sql: SQL query to optimize
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        sql_lower = sql.lower()
        
        if "where" in sql_lower and "index" not in sql_lower:
            suggestions.append("Consider adding indexes on columns used in WHERE clause")
        
        if "select *" in sql_lower:
            suggestions.append("Specify only required columns instead of SELECT *")
        
        if "distinct" in sql_lower:
            suggestions.append("DISTINCT can be expensive; ensure it's necessary")
        
        if "select" in sql_lower[10:]:
            suggestions.append("Consider using JOINs instead of subqueries for better performance")
        
        if "limit" not in sql_lower:
            suggestions.append("Add LIMIT clause to restrict result size")
        
        return suggestions
    
    def get_query_stats(self) -> Dict[str, Any]:
        if not self.query_history:
            return {"total_queries": 0}
        
        total_queries = len(self.query_history)
        avg_time = sum(q["time"] for q in self.query_history) / total_queries
        avg_rows = sum(q["rows"] for q in self.query_history) / total_queries
        
        return {
            "total_queries": total_queries,
            "average_execution_time": round(avg_time, 3),
            "average_rows_returned": round(avg_rows, 1),
            "last_query": self.query_history[-1]["sql"] if self.query_history else None
        }
