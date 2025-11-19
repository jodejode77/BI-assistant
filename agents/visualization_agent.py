import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image
from dataclasses import dataclass

from core.llm_manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class VisualizationResult:
    success: bool
    chart_type: str
    image_data: Optional[bytes] = None
    plotly_json: Optional[str] = None
    error: Optional[str] = None
    description: Optional[str] = None


class VisualizationAgent:
    """
    Agent responsible for creating data visualizations
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize Visualization Agent
        
        Args:
            llm_manager: LLM manager for generating visualization specs
        """
        self.llm = llm_manager
        
        # Set default styles
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
        logger.info("[VISUALIZATION_AGENT] Инициализирован агент визуализации")
    
    def create_visualization(
        self,
        data: pd.DataFrame,
        question: Optional[str] = None,
        viz_type: Optional[str] = None,
        auto_select: bool = True
    ) -> VisualizationResult:
        """
        Create a visualization from data
        
        Args:
            data: Data to visualize
            question: Original user question for context
            viz_type: Specific visualization type requested
            auto_select: Whether to auto-select visualization type
            
        Returns:
            VisualizationResult with the generated visualization
        """
        try:
            logger.info(f"[VISUALIZATION_AGENT] Создание визуализации: rows={len(data)}, cols={len(data.columns)}, question='{question[:60] if question else 'N/A'}...'")
            
            if data.empty:
                logger.warning("[VISUALIZATION_AGENT] Данные пусты, визуализация невозможна")
                return VisualizationResult(
                    success=False,
                    chart_type="none",
                    error="No data available for visualization"
                )
            
            logger.debug("[VISUALIZATION_AGENT] → Определение типа визуализации")
            if viz_type:
                chart_type = viz_type
                viz_spec = {"chart_type": viz_type}
                logger.debug(f"[VISUALIZATION_AGENT] Тип визуализации задан явно: {viz_type}")
            elif auto_select:
                logger.debug("[VISUALIZATION_AGENT] Автоматический выбор типа через LLM")
                viz_spec = self._auto_select_visualization(data, question)
                chart_type = viz_spec.get("chart_type", "bar")
                logger.info(f"[VISUALIZATION_AGENT] Автоматически выбран тип: {chart_type}")
            else:
                chart_type = "table"
                viz_spec = {"chart_type": "table"}
                logger.debug("[VISUALIZATION_AGENT] Используется тип по умолчанию: table")
            
            logger.info(f"[VISUALIZATION_AGENT] → Создание визуализации типа: {chart_type}")
            if chart_type == "table":
                result = self._create_table(data)
            elif chart_type == "bar":
                result = self._create_bar_chart(data, viz_spec)
            elif chart_type == "line":
                result = self._create_line_chart(data, viz_spec)
            elif chart_type == "scatter":
                result = self._create_scatter_plot(data, viz_spec)
            elif chart_type == "pie":
                result = self._create_pie_chart(data, viz_spec)
            elif chart_type == "heatmap":
                result = self._create_heatmap(data, viz_spec)
            elif chart_type == "box":
                result = self._create_box_plot(data, viz_spec)
            elif chart_type == "histogram":
                result = self._create_histogram(data, viz_spec)
            elif chart_type == "multi":
                result = self._create_multi_chart(data, viz_spec)
            else:
                result = self._create_bar_chart(data, viz_spec)
            
            logger.debug("[VISUALIZATION_AGENT] → Генерация описания визуализации")
            result.description = self._generate_description(data, chart_type, question)
            
            logger.info(f"[VISUALIZATION_AGENT] Визуализация создана: success={result.success}, type={result.chart_type}, image_size={len(result.image_data) if result.image_data else 0} bytes")
            return result
            
        except Exception as e:
            logger.error(f"[VISUALIZATION_AGENT] ОШИБКА при создании визуализации: {e}", exc_info=True)
            return VisualizationResult(
                success=False,
                chart_type="error",
                error=str(e)
            )
    
    def _auto_select_visualization(
        self,
        data: pd.DataFrame,
        question: Optional[str] = None
    ) -> Dict[str, Any]:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        data_info = f"""
Data shape: {data.shape}
Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:5])}
Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}
Datetime columns ({len(datetime_cols)}): {', '.join(datetime_cols[:3])}
Sample values: {data.head(3).to_dict()}
"""
        
        # Use LLM to generate visualization spec
        viz_spec = self.llm.generate_visualization_spec(
            data_description=data_info,
            user_request=question
        )
        
        # Validate and adjust spec
        viz_spec = self._validate_viz_spec(viz_spec, data)
        
        return viz_spec
    
    def _validate_viz_spec(self, spec: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        if "chart_type" not in spec:
            spec["chart_type"] = "bar"
        
        # Validate column names
        if "x_axis" in spec and spec["x_axis"] not in data.columns:
            spec["x_axis"] = data.columns[0]
        
        if "y_axis" in spec and spec["y_axis"] not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            spec["y_axis"] = numeric_cols[0] if len(numeric_cols) > 0 else data.columns[1]
        
        return spec
    
    def _create_bar_chart(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a bar chart"""
        
        try:
            # Determine columns
            x_col = spec.get("x_axis")
            y_col = spec.get("y_axis")
            
            if not x_col or not y_col:
                # Auto-select columns
                if len(data.columns) >= 2:
                    x_col = data.columns[0]
                    y_col = data.columns[1]
                else:
                    x_col = data.index.name or "index"
                    y_col = data.columns[0]
                    data = data.reset_index()
            
            # Limit data for visualization
            if len(data) > 20:
                data = data.head(20)
            
            # Create Plotly figure
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                title=spec.get("title", f"{y_col} by {x_col}"),
                color=spec.get("color"),
                text_auto=True
            )
            
            fig.update_layout(
                showlegend=True,
                xaxis_tickangle=-45,
                height=500
            )
            
            # Convert to image - try kaleido first, fallback to HTML if fails
            try:
                img_bytes = fig.to_image(format="png")
            except (ValueError, ImportError) as e:
                # Fallback: use HTML representation or matplotlib
                logger.warning(f"[VISUALIZATION_AGENT] Kaleido export failed: {e}, using HTML fallback")
                # Return plotly JSON instead of image
                return VisualizationResult(
                    success=True,
                    chart_type="bar",
                    image_data=None,
                    plotly_json=fig.to_json(),
                    description="Chart created (image export unavailable, using interactive format)"
                )
            
            return VisualizationResult(
                success=True,
                chart_type="bar",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            logger.error(f"[VISUALIZATION_AGENT] ОШИБКА при создании bar chart: {e}", exc_info=True)
            return VisualizationResult(
                success=False,
                chart_type="bar",
                error=str(e)
            )
    
    def _create_line_chart(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a line chart"""
        
        try:
            x_col = spec.get("x_axis", data.columns[0])
            y_col = spec.get("y_axis", data.columns[1] if len(data.columns) > 1 else data.columns[0])
            
            fig = px.line(
                data,
                x=x_col,
                y=y_col,
                title=spec.get("title", f"{y_col} over {x_col}"),
                color=spec.get("color"),
                markers=True
            )
            
            fig.update_layout(height=500)
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="line",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="line",
                error=str(e)
            )
    
    def _create_scatter_plot(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a scatter plot"""
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return VisualizationResult(
                    success=False,
                    chart_type="scatter",
                    error="Need at least 2 numeric columns for scatter plot"
                )
            
            x_col = spec.get("x_axis", numeric_cols[0])
            y_col = spec.get("y_axis", numeric_cols[1])
            
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                title=spec.get("title", f"{y_col} vs {x_col}"),
                color=spec.get("color"),
                size=spec.get("size"),
                hover_data=data.columns
            )
            
            # Add trendline if appropriate
            if len(data) > 10:
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    )
                )
            
            fig.update_layout(height=500)
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="scatter",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="scatter",
                error=str(e)
            )
    
    def _create_pie_chart(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a pie chart"""
        
        try:
            # Select appropriate columns
            if len(data.columns) >= 2:
                labels_col = spec.get("labels", data.columns[0])
                values_col = spec.get("values", data.columns[1])
            else:
                labels_col = data.index.name or "index"
                values_col = data.columns[0]
                data = data.reset_index()
            
            # Limit to top 10 categories
            if len(data) > 10:
                data = data.nlargest(10, values_col)
            
            fig = px.pie(
                data,
                names=labels_col,
                values=values_col,
                title=spec.get("title", f"Distribution of {values_col}")
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="pie",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="pie",
                error=str(e)
            )
    
    def _create_heatmap(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a heatmap"""
        
        try:
            # Get numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return VisualizationResult(
                    success=False,
                    chart_type="heatmap",
                    error="No numeric data for heatmap"
                )
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                title=spec.get("title", "Correlation Heatmap"),
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            
            fig.update_layout(height=600)
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="heatmap",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="heatmap",
                error=str(e)
            )
    
    def _create_box_plot(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a box plot"""
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return VisualizationResult(
                    success=False,
                    chart_type="box",
                    error="No numeric data for box plot"
                )
            
            y_col = spec.get("y_axis", numeric_cols[0])
            x_col = spec.get("x_axis")
            
            fig = px.box(
                data,
                x=x_col,
                y=y_col,
                title=spec.get("title", f"Distribution of {y_col}"),
                color=spec.get("color")
            )
            
            fig.update_layout(height=500)
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="box",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="box",
                error=str(e)
            )
    
    def _create_histogram(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create a histogram"""
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return VisualizationResult(
                    success=False,
                    chart_type="histogram",
                    error="No numeric data for histogram"
                )
            
            x_col = spec.get("x_axis", numeric_cols[0])
            
            fig = px.histogram(
                data,
                x=x_col,
                title=spec.get("title", f"Distribution of {x_col}"),
                nbins=spec.get("bins", 30),
                color=spec.get("color")
            )
            
            fig.update_layout(height=500, showlegend=True)
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="histogram",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="histogram",
                error=str(e)
            )
    
    def _create_table(self, data: pd.DataFrame) -> VisualizationResult:
        """Create a formatted table visualization"""
        
        try:
            # Limit rows for display
            display_data = data.head(20)
            
            fig = go.Figure(data=[
                go.Table(
                    header=dict(
                        values=list(display_data.columns),
                        fill_color='paleturquoise',
                        align='left',
                        font=dict(size=12)
                    ),
                    cells=dict(
                        values=[display_data[col] for col in display_data.columns],
                        fill_color='lavender',
                        align='left',
                        font=dict(size=11)
                    )
                )
            ])
            
            fig.update_layout(
                title="Data Table",
                height=400 + len(display_data) * 20
            )
            
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="table",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="table",
                error=str(e)
            )
    
    def _create_multi_chart(self, data: pd.DataFrame, spec: Dict[str, Any]) -> VisualizationResult:
        """Create multiple charts in a subplot"""
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]
            
            if len(numeric_cols) < 2:
                return self._create_bar_chart(data, spec)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f"Distribution of {col}" for col in numeric_cols],
                specs=[[{"type": "histogram"}, {"type": "box"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Add histograms
            for i, col in enumerate(numeric_cols[:2]):
                fig.add_trace(
                    go.Histogram(x=data[col], name=col),
                    row=1, col=i+1
                )
            
            # Add box plots
            if len(numeric_cols) > 2:
                fig.add_trace(
                    go.Box(y=data[numeric_cols[2]], name=numeric_cols[2]),
                    row=2, col=1
                )
            
            # Add bar chart
            if len(numeric_cols) > 3:
                fig.add_trace(
                    go.Bar(x=data.index[:10], y=data[numeric_cols[3]][:10], name=numeric_cols[3]),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=False, title_text="Multi-Chart Analysis")
            img_bytes = fig.to_image(format="png")
            
            return VisualizationResult(
                success=True,
                chart_type="multi",
                image_data=img_bytes,
                plotly_json=fig.to_json()
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                chart_type="multi",
                error=str(e)
            )
    
    def _generate_description(
        self,
        data: pd.DataFrame,
        chart_type: str,
        question: Optional[str] = None
    ) -> str:
        """Generate a description of the visualization"""
        
        descriptions = {
            "bar": f"Bar chart showing {data.shape[0]} data points",
            "line": f"Line chart tracking trends across {data.shape[0]} points",
            "scatter": f"Scatter plot revealing relationships in {data.shape[0]} observations",
            "pie": f"Pie chart displaying distribution across {data.shape[0]} categories",
            "heatmap": f"Heatmap showing correlations between {data.shape[1]} variables",
            "box": f"Box plot illustrating data distribution and outliers",
            "histogram": f"Histogram showing frequency distribution",
            "table": f"Table view of {data.shape[0]} rows and {data.shape[1]} columns",
            "multi": f"Multi-chart dashboard analyzing {data.shape[1]} dimensions"
        }
        
        base_description = descriptions.get(chart_type, f"{chart_type} visualization")
        
        if question:
            return f"{base_description} to answer: {question}"
        
        return base_description
