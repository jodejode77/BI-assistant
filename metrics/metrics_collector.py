import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    query_count: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    average_rows_returned: float = 0.0


@dataclass
class LLMMetrics:
    total_requests: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    average_tokens_per_request: float = 0.0
    error_rate: float = 0.0


@dataclass 
class UserMetrics:
    total_users: int = 0
    active_users: int = 0
    total_messages: int = 0
    average_session_length: float = 0.0
    user_satisfaction_score: float = 0.0
    retention_rate: float = 0.0


@dataclass
class SystemMetrics:
    uptime: float = 0.0
    total_requests: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    peak_load: int = 0
    memory_usage: float = 0.0
    

@dataclass
class QualityMetrics:
    sql_accuracy: float = 0.0
    analysis_confidence: float = 0.0
    visualization_success_rate: float = 0.0
    rag_relevance_score: float = 0.0
    user_feedback_positive: float = 0.0


class MetricsCollector:
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.query_metrics = QueryMetrics()
        self.llm_metrics = LLMMetrics()
        self.user_metrics = UserMetrics()
        self.system_metrics = SystemMetrics()
        self.quality_metrics = QualityMetrics()
        
        self.sessions = {}
        self.start_time = datetime.now()
        
        self._load_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def record_query(
        self,
        success: bool,
        execution_time: float,
        rows_returned: int,
        query: str = None
    ):
        self.query_metrics.query_count += 1
        
        if success:
            self.query_metrics.successful_queries += 1
        else:
            self.query_metrics.failed_queries += 1
        
        self.query_metrics.total_execution_time += execution_time
        self.query_metrics.max_execution_time = max(
            self.query_metrics.max_execution_time,
            execution_time
        )
        self.query_metrics.min_execution_time = min(
            self.query_metrics.min_execution_time,
            execution_time
        )
        
        if self.query_metrics.query_count > 0:
            self.query_metrics.average_execution_time = (
                self.query_metrics.total_execution_time / self.query_metrics.query_count
            )
            
            total_rows = self.query_metrics.average_rows_returned * (self.query_metrics.query_count - 1)
            self.query_metrics.average_rows_returned = (total_rows + rows_returned) / self.query_metrics.query_count
    
    def record_llm_request(
        self,
        tokens_used: int,
        response_time: float,
        success: bool = True
    ):
        self.llm_metrics.total_requests += 1
        self.llm_metrics.total_tokens_used += tokens_used
        
        prev_avg_time = self.llm_metrics.average_response_time
        self.llm_metrics.average_response_time = (
            (prev_avg_time * (self.llm_metrics.total_requests - 1) + response_time) /
            self.llm_metrics.total_requests
        )
        
        self.llm_metrics.average_tokens_per_request = (
            self.llm_metrics.total_tokens_used / self.llm_metrics.total_requests
        )
        
        if not success:
            self.llm_metrics.error_rate = (
                (self.llm_metrics.error_rate * (self.llm_metrics.total_requests - 1) + 1) /
                self.llm_metrics.total_requests
            )
    
    def record_user_interaction(
        self,
        user_id: str,
        message_type: str = "query",
        session_start: bool = False,
        session_end: bool = False
    ):
        self.user_metrics.total_messages += 1
        
        if session_start:
            self.sessions[user_id] = {
                "start": datetime.now(),
                "messages": 1
            }
            self.user_metrics.total_users = len(self.sessions)
            self.user_metrics.active_users = len([
                s for s in self.sessions.values()
                if datetime.now() - s["start"] < timedelta(hours=1)
            ])
        elif user_id in self.sessions:
            self.sessions[user_id]["messages"] += 1
            
        if session_end and user_id in self.sessions:
            session_length = (datetime.now() - self.sessions[user_id]["start"]).seconds / 60
            
            total_sessions = len([s for s in self.sessions.values() if "end" in s]) + 1
            prev_avg = self.user_metrics.average_session_length
            self.user_metrics.average_session_length = (
                (prev_avg * (total_sessions - 1) + session_length) / total_sessions
            )
            
            self.sessions[user_id]["end"] = datetime.now()
    
    def record_quality_metric(
        self,
        metric_type: str,
        value: float
    ):
        if metric_type == "sql_accuracy":
            self.quality_metrics.sql_accuracy = (
                self.quality_metrics.sql_accuracy * 0.9 + value * 0.1
            )
        elif metric_type == "analysis_confidence":
            self.quality_metrics.analysis_confidence = (
                self.quality_metrics.analysis_confidence * 0.9 + value * 0.1
            )
        elif metric_type == "visualization_success":
            self.quality_metrics.visualization_success_rate = (
                self.quality_metrics.visualization_success_rate * 0.9 + value * 0.1
            )
        elif metric_type == "rag_relevance":
            self.quality_metrics.rag_relevance_score = (
                self.quality_metrics.rag_relevance_score * 0.9 + value * 0.1
            )
        elif metric_type == "user_feedback":
            self.quality_metrics.user_feedback_positive = (
                self.quality_metrics.user_feedback_positive * 0.9 + value * 0.1
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": round(uptime, 2),
            "query_metrics": {
                "total_queries": self.query_metrics.query_count,
                "success_rate": (
                    self.query_metrics.successful_queries / max(self.query_metrics.query_count, 1) * 100
                ),
                "avg_execution_time": round(self.query_metrics.average_execution_time, 3),
                "avg_rows_returned": round(self.query_metrics.average_rows_returned, 1)
            },
            "llm_metrics": {
                "total_requests": self.llm_metrics.total_requests,
                "total_tokens": self.llm_metrics.total_tokens_used,
                "avg_response_time": round(self.llm_metrics.average_response_time, 3),
                "avg_tokens_per_request": round(self.llm_metrics.average_tokens_per_request, 1),
                "error_rate": round(self.llm_metrics.error_rate * 100, 2)
            },
            "user_metrics": {
                "total_users": self.user_metrics.total_users,
                "active_users": self.user_metrics.active_users,
                "total_messages": self.user_metrics.total_messages,
                "avg_session_length_minutes": round(self.user_metrics.average_session_length, 1)
            },
            "quality_metrics": {
                "sql_accuracy": round(self.quality_metrics.sql_accuracy * 100, 1),
                "analysis_confidence": round(self.quality_metrics.analysis_confidence * 100, 1),
                "visualization_success_rate": round(self.quality_metrics.visualization_success_rate * 100, 1),
                "rag_relevance_score": round(self.quality_metrics.rag_relevance_score, 2),
                "user_satisfaction": round(self.quality_metrics.user_feedback_positive * 100, 1)
            }
        }
    
    def generate_report(self) -> str:
        summary = self.get_metrics_summary()
        
        report = f"""
═══════════════════════════════════════════════════════════════
                    METRICS REPORT
                {summary['timestamp']}
═══════════════════════════════════════════════════════════════

QUERY PERFORMANCE
─────────────────────────────
• Total Queries: {summary['query_metrics']['total_queries']}
• Success Rate: {summary['query_metrics']['success_rate']:.1f}%
• Avg Execution Time: {summary['query_metrics']['avg_execution_time']}s
• Avg Rows Returned: {summary['query_metrics']['avg_rows_returned']}

LLM USAGE
─────────────────────────────
• Total Requests: {summary['llm_metrics']['total_requests']}
• Total Tokens Used: {summary['llm_metrics']['total_tokens']:,}
• Avg Response Time: {summary['llm_metrics']['avg_response_time']}s
• Avg Tokens/Request: {summary['llm_metrics']['avg_tokens_per_request']}
• Error Rate: {summary['llm_metrics']['error_rate']}%

USER ENGAGEMENT
─────────────────────────────
• Total Users: {summary['user_metrics']['total_users']}
• Active Users: {summary['user_metrics']['active_users']}
• Total Messages: {summary['user_metrics']['total_messages']}
• Avg Session Length: {summary['user_metrics']['avg_session_length_minutes']} min

QUALITY METRICS
─────────────────────────────
• SQL Accuracy: {summary['quality_metrics']['sql_accuracy']}%
• Analysis Confidence: {summary['quality_metrics']['analysis_confidence']}%
• Visualization Success: {summary['quality_metrics']['visualization_success_rate']}%
• RAG Relevance: {summary['quality_metrics']['rag_relevance_score']}/5
• User Satisfaction: {summary['quality_metrics']['user_satisfaction']}%

SYSTEM UPTIME: {summary['uptime_hours']} hours
═══════════════════════════════════════════════════════════════
"""
        
        return report
    
    def save_metrics(self):
        metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.get_metrics_summary(), f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def _load_metrics(self):
        try:
            metrics_files = sorted(self.metrics_dir.glob("metrics_*.json"))
            
            if metrics_files:
                with open(metrics_files[-1], 'r') as f:
                    historical = json.load(f)
                
                if "query_metrics" in historical:
                    self.query_metrics.query_count = historical["query_metrics"].get("total_queries", 0)
                
                logger.info(f"Loaded historical metrics from {metrics_files[-1]}")
        except Exception as e:
            logger.warning(f"Could not load historical metrics: {e}")
    
    def calculate_performance_score(self) -> float:
        weights = {
            "success_rate": 0.3,
            "response_time": 0.2,
            "quality": 0.3,
            "user_satisfaction": 0.2
        }
        
        success_rate = self.query_metrics.successful_queries / max(self.query_metrics.query_count, 1)
        
        response_time_score = max(0, 1 - (self.query_metrics.average_execution_time / 10))
        
        quality_score = (
            self.quality_metrics.sql_accuracy * 0.4 +
            self.quality_metrics.analysis_confidence * 0.3 +
            self.quality_metrics.visualization_success_rate * 0.3
        )
        
        user_score = self.quality_metrics.user_feedback_positive
        
        score = (
            success_rate * weights["success_rate"] +
            response_time_score * weights["response_time"] +
            quality_score * weights["quality"] +
            user_score * weights["user_satisfaction"]
        )
        
        return round(score * 100, 1)
