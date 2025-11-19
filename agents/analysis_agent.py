import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

from plotly.figure_factory._distplot import scipy_stats
from sklearn.linear_model import LinearRegression

from core.llm_manager import LLMManager
from core.rag_system import RAGSystem

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    interpretation: str
    insights: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class AnalysisAgent:
    """
    Agent responsible for analyzing and interpreting data results.

    Сохранена публичная архитектура:
      - analyze_results(question, sql_query, data, context) -> AnalysisResult
      - generate_summary_report(analysis_result, include_details=False) -> str

    Добавлен режим анализа (use_llm_analysis):
      - 'llm'    -> LLM получает данные и сам делает детекцию/интерпретацию
      - 'hybrid' -> rule-based + LLM интерпретация (по-умолчанию гибридный режим)

    Важно: LLMManager должен реализовывать метод interpret_results(question, sql_query, results, context)
    """
    def __init__(
        self,
        llm_manager: LLMManager,
        rag_system: RAGSystem,
        use_llm_analysis: str = "hybrid"
    ):
        self.llm = llm_manager
        self.rag = rag_system
        self.mode = use_llm_analysis if use_llm_analysis in {"rule", "llm", "hybrid"} else "hybrid"

        logger.info(f"Initialized Analysis Agent (mode={self.mode})")
    
    def analyze_results(
        self,
        question: str,
        sql_query: str,
        data: pd.DataFrame,
        context: Optional[str] = None
    ) -> AnalysisResult:
        """Analyze query results and provide interpretation.
        
        Performs comprehensive analysis including statistics generation, pattern
        detection, insight extraction, and recommendation generation.
        
        Args:
            question: Original user question.
            sql_query: Executed SQL query.
            data: Query results as DataFrame.
            context: Optional additional context.
            
        Returns:
            AnalysisResult with interpretation, insights, statistics,
            recommendations, and confidence score.
        """
        try:
            logger.info(f"[ANALYSIS_AGENT] Начало анализа: question='{question[:60]}...', rows={len(data)}, cols={len(data.columns)}")

            logger.debug("[ANALYSIS_AGENT] → Генерация статистики")
            statistics = self._generate_statistics(data)
            logger.debug(
                f"[ANALYSIS_AGENT] Статистика сгенерирована: numeric_cols={len(statistics.get('numeric_columns', {}))}")
            logger.debug("[ANALYSIS_AGENT] → Обнаружение паттернов")

            patterns = {}
            if self.mode == "hybrid":
                patterns = self._detect_patterns(data)
            elif self.mode == "llm":
                patterns = {}
            logger.debug(f"[ANALYSIS_AGENT] Паттерны обнаружены: {len(patterns)}")

            logger.info("[ANALYSIS_AGENT] → Генерация интерпретации через LLM")
            interpretation = self._generate_interpretation(question, sql_query, data, statistics, patterns, context)
            logger.info(f"[ANALYSIS_AGENT] Интерпретация сгенерирована: {len(interpretation)} символов")

            logger.debug("[ANALYSIS_AGENT] → Извлечение инсайтов")

            insights = self._extract_insights(data, statistics, patterns, sql_query)
            logger.debug(f"[ANALYSIS_AGENT] Инсайты извлечены: {len(insights)}")

            logger.debug("[ANALYSIS_AGENT] → Генерация рекомендаций")
            recommendations = self._generate_recommendations(question, insights, statistics)
            logger.debug(f"[ANALYSIS_AGENT] Рекомендации сгенерированы: {len(recommendations)}")

            logger.debug("[ANALYSIS_AGENT] → Расчет уровня уверенности")
            confidence = self._calculate_confidence(data, statistics)
            logger.info(f"[ANALYSIS_AGENT] Анализ завершен: insights={len(insights)}, recommendations={len(recommendations)}, confidence={confidence:.2f}")

            return AnalysisResult(
                interpretation=interpretation,
                insights=insights,
                statistics=statistics,
                recommendations=recommendations,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"[ANALYSIS_AGENT] ОШИБКА при анализе: {e}", exc_info=True)
            return AnalysisResult(
                interpretation=f"Analysis failed: {str(e)}",
                insights=[],
                statistics={},
                recommendations=[],
                confidence_score=0.0
            )
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary of the data.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Dictionary containing row count, column count, and statistics
            for numeric and categorical columns.
        """
        
        stats = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "columns": list(data.columns)
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = {}
            for col in numeric_cols:
                if not data[col].empty:
                    numeric_stats[col] = {
                        "mean": float(data[col].mean()) if not data[col].isna().all() else None,
                        "median": float(data[col].median()) if not data[col].isna().all() else None,
                        "std": float(data[col].std()) if not data[col].isna().all() else None,
                        "min": float(data[col].min()) if not data[col].isna().all() else None,
                        "max": float(data[col].max()) if not data[col].isna().all() else None,
                        "nulls": int(data[col].isna().sum())
                    }
            stats["numeric_columns"] = numeric_stats
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols[:5]:
                if not data[col].empty:
                    value_counts = data[col].value_counts()
                    categorical_stats[col] = {
                        "unique_values": int(data[col].nunique()),
                        "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    }
            stats["categorical_columns"] = categorical_stats
        
        return stats

    def _detect_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns and anomalies in the data"""

        patterns = {
            "trends": [],
            "anomalies": [],
            "correlations": []
        }

        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            ser = data[col].dropna()
            if len(ser) < 5:
                continue

            slope, slope_p = self._estimate_trend_slope(ser)

            try:
                change_percent = (
                        (float(ser.iloc[-1]) - float(ser.iloc[0])) /
                        (abs(float(ser.iloc[0])) + 1e-9) * 100
                )
            except Exception:
                change_percent = None

            if slope is not None:
                if abs(slope) > 1e-9 and (slope_p is None or slope_p < 0.1 or abs(change_percent or 0) > 10):
                    trend_type = "increasing" if slope > 0 else "decreasing"
                    patterns["trends"].append({
                        "column": col,
                        "slope": float(slope),
                        "trend_type": trend_type,
                        "slope_p_value": float(slope_p) if slope_p is not None else None,
                        "change_percent": float(round(change_percent, 3)) if change_percent is not None else None,
                        "n_points": int(len(ser))
                    })

        for col in numeric_cols:
            if not data[col].isna().all():
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1

                if pd.notna(Q1) and pd.notna(Q3) and IQR > 0:
                    outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)]

                    if len(outliers) > 0:
                        patterns["anomalies"].append({
                            "column": col,
                            "outlier_count": len(outliers),
                            "outlier_percentage": round(len(outliers) / len(data) * 100, 2)
                        })

        if len(numeric_cols) > 1 and len(data) > 10:
            corr_matrix = data[numeric_cols].corr()

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    if pd.notna(corr_value) and abs(corr_value) > 0.7:
                        patterns["correlations"].append({
                            "column1": numeric_cols[i],
                            "column2": numeric_cols[j],
                            "correlation": round(corr_value, 3),
                            "strength": "strong positive" if corr_value > 0 else "strong negative"
                        })

        return patterns

    def _estimate_trend_slope(self, series: pd.Series):
        """
        Возвращает (slope, p_value) оценки тренда по индексу.
        Если scipy/sklearn недоступны, возвращает простую оценку slope по линейной регрессии numpy.
        """
        try:
            y = series.values.astype(float)
            X = np.arange(len(y)).reshape(-1, 1)
            if len(y) >= 5:
                model = LinearRegression().fit(X, y)
                slope = float(model.coef_[0])
                slope_res = scipy_stats.linregress(np.arange(len(y)), y)
                slope_p = float(slope_res.pvalue)
                return slope, slope_p
            else:
                coeffs = np.polyfit(np.arange(len(y)), y, 1)
                slope = float(coeffs[0])
                return slope, None
        except Exception as e:
            logger.debug("Trend estimation failed: %s", e)
            return None, None

    def _generate_interpretation(
        self,
        question: str,
        sql_query: str,
        data: pd.DataFrame,
        statistics: Dict[str, Any],
        patterns: Dict[str, Any],
        context: Optional[str]
    ) -> str:
        """Generate natural language interpretation of results.
        
        Args:
            question: Original user question.
            sql_query: Executed SQL query.
            data: Query results DataFrame.
            statistics: Generated statistics.
            patterns: Detected patterns.
            context: Optional additional context.
            
        Returns:
            Natural language interpretation of the results.
        """
        stat_json = json.dumps(statistics, indent=2, ensure_ascii=False)
        patterns_json = json.dumps(patterns, indent=2, ensure_ascii=False)

        data_summary = data.head(10).to_string() if len(data) > 0 else "No data returned"

        interpretation_context = f"""
        Question: {question}

        Statistics (json):
        {stat_json}

        Sample Data:
        {data_summary}

        Patterns (json):
        {patterns_json}
        """

        if context:
            interpretation_context += f"\nAdditional Context: {context}"

        interpretation = self.llm.interpret_results(
            question=question,
            sql_query=sql_query,
            results=interpretation_context,
            context=context
        )

        return interpretation

    def _extract_insights(
        self,
        data: pd.DataFrame,
        statistics: Dict[str, Any],
        patterns: Dict[str, Any],
        sql_query: Optional[str] = None
    ) -> List[str]:
        """Extract key insights from the analysis.
        
        Args:
            data: Query results DataFrame.
            statistics: Generated statistics.
            patterns: Detected patterns.
            sql_query: Optional SQL query to detect aggregation.

        Returns:
            List of insight strings.
        """
        insights = []
        
        # Check if query is aggregated
        is_aggregated = False
        if sql_query:
            sql_upper = sql_query.upper()
            # Check for aggregation functions
            aggregation_keywords = ['AVG(', 'SUM(', 'COUNT(', 'MAX(', 'MIN(', 'GROUP BY', 'HAVING']
            is_aggregated = any(kw in sql_upper for kw in aggregation_keywords)

        if statistics['row_count'] == 0:
            insights.append("No data found matching the query criteria")
        elif statistics['row_count'] == 1:
            if is_aggregated:
                # For aggregated queries, 1 row is normal
                insights.append("Aggregated result calculated from underlying data")
            else:
                insights.append("Single record found - represents a unique case")
        elif statistics['row_count'] > 90:
            insights.append("Large dataset returned - results may be truncated")
        
        if "numeric_columns" in statistics:
            for col, stats in statistics["numeric_columns"].items():
                if stats.get("std") and stats.get("mean"):
                    cv = stats["std"] / abs(stats["mean"]) * 100 if stats["mean"] != 0 else 0
                    if cv > 100:
                        insights.append(f"High variability in {col} (CV: {cv:.1f}%)")
                
                if stats.get("nulls", 0) > statistics['row_count'] * 0.2:
                    insights.append(f"Significant missing data in {col} ({stats['nulls']} nulls)")

        for trend in patterns.get("trends", [])[:3]:
            trend_type = trend.get("trend_type", "trend")
            col = trend.get("column", "unknown")
            change = trend.get("change_percent")

            if change is not None:
                insights.append(f"{trend_type.capitalize()} trend in {col} ({change:.1f}% change)")
            else:
                insights.append(f"{trend_type.capitalize()} trend detected in {col}")
        
        for anomaly in patterns.get("anomalies", [])[:2]:
            insights.append(
                f"{anomaly['outlier_count']} outliers detected in {anomaly['column']} "
                f"({anomaly['outlier_percentage']:.1f}% of data)"
            )
        
        for correlation in patterns.get("correlations", [])[:2]:
            insights.append(
                f"{correlation['strength'].capitalize()} correlation between "
                f"{correlation['column1']} and {correlation['column2']} ({correlation['correlation']:.2f})"
            )
        
        return insights
    
    def _generate_recommendations(
        self,
        question: str,
        insights: List[str],
        statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis.
        
        Args:
            question: Original user question.
            insights: Extracted insights.
            statistics: Generated statistics.
            
        Returns:
            List of recommendation strings (max 5).
        """
        recommendations = []
        
        if statistics['row_count'] == 0:
            recommendations.append("Try broadening your search criteria or checking data availability")
        elif statistics['row_count'] >= 100:
            recommendations.append("Consider adding more specific filters to narrow down results")
        
        if "numeric_columns" in statistics:
            high_null_cols = [
                col for col, stats in statistics["numeric_columns"].items()
                if stats.get("nulls", 0) > statistics['row_count'] * 0.3
            ]
            if high_null_cols:
                recommendations.append(
                    f"Consider data quality improvements for: {', '.join(high_null_cols[:3])}"
                )
        
        if any("outlier" in insight.lower() for insight in insights):
            recommendations.append("Investigate outliers for data quality issues or special cases")
        
        if any("correlation" in insight.lower() for insight in insights):
            recommendations.append("Consider using correlated features for predictive modeling")
        
        if len(recommendations) < 2:
            recommendations.append("Consider visualizing this data for better insights")
            recommendations.append("Regular monitoring of these metrics is recommended")
        
        return recommendations[:5]
    
    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        statistics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis.
        
        Confidence is reduced for small datasets and high missing data rates.
        
        Args:
            data: Query results DataFrame.
            statistics: Generated statistics.
            
        Returns:
            Confidence score between 0.1 and 1.0.
        """
        confidence = 1.0
        
        if len(data) < 5:
            confidence *= 0.7
        elif len(data) < 10:
            confidence *= 0.85
        
        if "numeric_columns" in statistics:
            null_ratios = [
                stats.get("nulls", 0) / statistics['row_count']
                for stats in statistics["numeric_columns"].values()
                if statistics['row_count'] > 0
            ]
            if null_ratios:
                avg_null_ratio = sum(null_ratios) / len(null_ratios)
                confidence *= (1 - avg_null_ratio * 0.5)
        
        confidence = max(0.1, min(1.0, confidence))
        
        return round(confidence, 2)
    
    def generate_summary_report(
        self,
        analysis_result: AnalysisResult,
        include_details: bool = False
    ) -> str:
        """Generate a formatted summary report.
        
        Args:
            analysis_result: Analysis result to summarize.
            include_details: Whether to include detailed statistics (default: False).
            
        Returns:
            Formatted report as string with markdown formatting.
        """
        report_parts = [
            "**Analysis Report**\n",
            f"**Confidence Score**: {analysis_result.confidence_score:.0%}\n",
            "**Interpretation**:",
            analysis_result.interpretation,
            "\n**Key Insights**:"
        ]
        
        for insight in analysis_result.insights:
            report_parts.append(f"• {insight}")
        
        if analysis_result.recommendations:
            report_parts.append("\n**Recommendations**:")
            for rec in analysis_result.recommendations:
                report_parts.append(f"• {rec}")
        
        if include_details and analysis_result.statistics:
            report_parts.append("\n**Statistical Summary**:")
            report_parts.append(f"• Records: {analysis_result.statistics.get('row_count', 0)}")
            report_parts.append(f"• Columns: {analysis_result.statistics.get('column_count', 0)}")
        
        return "\n".join(report_parts)
