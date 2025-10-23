"""
Base chart classes for the visualization engine.
Provides common functionality and styling to reduce duplication.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod


class BaseChart(ABC):
    """Abstract base class for all charts"""
    
    def __init__(self):
        # Define color scheme
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40',
            'buy': '#28a745',
            'sell': '#dc3545',
            'neutral': '#6c757d'
        }
        
        # Common layout template (only valid Layout properties)
        self.layout_template = {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff', 'size': 12},
            'xaxis': {'gridcolor': '#444444', 'color': '#ffffff', 'showgrid': True},
            'yaxis': {'gridcolor': '#444444', 'color': '#ffffff', 'showgrid': True},
            'colorway': [self.colors['primary'], self.colors['secondary'],
                        self.colors['success'], self.colors['danger'],
                        self.colors['warning'], self.colors['info']]
        }
    
    @abstractmethod
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create the chart figure"""
        pass
    
    def _create_empty_chart(self, title: str, message: str = "No data available") -> go.Figure:
        """Create an empty chart with error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=self.colors['neutral'])
        )
        fig.update_layout(
            title=title,
            **self.layout_template,
            height=400
        )
        return fig
    
    def _create_error_chart(self, title: str, error_message: str) -> go.Figure:
        """Create an error chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=self.colors['danger'])
        )
        fig.update_layout(
            title=title,
            **self.layout_template,
            height=400
        )
        return fig
    
    def _apply_common_layout(self, fig: go.Figure, title: str, height: int = 400) -> go.Figure:
        """Apply common layout settings to a figure"""
        fig.update_layout(
            title=title,
            **self.layout_template,
            height=height
        )
        return fig
    
    def _create_hover_template(self, chart_type: str) -> str:
        """Create hover template based on chart type"""
        templates = {
            'ladder': '<b>%{fullData.name}</b><br>' +
                     'Depth: %{x:.2f}%<br>' +
                     'Price: $%{y:.4f}<br>' +
                     'Quantity: %{customdata[0]:.4f}<br>' +
                     'Notional: $%{customdata[1]:.2f}<br>' +
                     '<extra></extra>',
            'probability': '<b>%{fullData.name}</b><br>' +
                         'Depth: %{x:.2f}%<br>' +
                         'Probability: %{y:.3f}<br>' +
                         '<extra></extra>',
            'kpi': '<b>%{fullData.name}</b><br>' +
                  'Value: %{y:.2f}<br>' +
                  '<extra></extra>',
            'default': '<b>%{fullData.name}</b><br>' +
                      'X: %{x}<br>' +
                      'Y: %{y}<br>' +
                      '<extra></extra>'
        }
        return templates.get(chart_type, templates['default'])


class LadderChart(BaseChart):
    """Base class for ladder-related charts"""
    
    def _extract_ladder_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract common ladder data from the data dictionary"""
        try:
            buy_depths = data.get('buy_depths', np.array([]))
            sell_depths = data.get('sell_depths', np.array([]))
            buy_quantities = data.get('buy_quantities', np.array([]))
            sell_quantities = data.get('sell_quantities', np.array([]))
            current_price = data.get('current_price', 100.0)
            
            # Calculate prices
            buy_prices = current_price * (1 - buy_depths / 100)
            sell_prices = current_price * (1 + sell_depths / 100)
            
            return buy_depths, sell_depths, buy_prices, sell_prices
        except Exception as e:
            print(f"Error extracting ladder data: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])


class ProbabilityChart(BaseChart):
    """Base class for probability-related charts"""
    
    def _extract_probability_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract probability data from the data dictionary"""
        try:
            depths = data.get('buy_depths', np.array([]))
            probabilities = data.get('touch_probabilities', np.array([]))
            weibull_params = data.get('weibull_params', {})
            
            return depths, probabilities, weibull_params
        except Exception as e:
            print(f"Error extracting probability data: {e}")
            return np.array([]), np.array([]), {}


class KPIChart(BaseChart):
    """Base class for KPI charts"""
    
    def _extract_kpi_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract KPI data from the data dictionary"""
        try:
            kpis = data.get('kpis', {})
            return kpis
        except Exception as e:
            print(f"Error extracting KPI data: {e}")
            return {}


class LadderConfigurationChart(LadderChart):
    """Ladder configuration chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create ladder configuration chart"""
        try:
            buy_depths, sell_depths, buy_prices, sell_prices = self._extract_ladder_data(data)
            
            if len(buy_depths) == 0:
                return self._create_empty_chart("Ladder Configuration", "No ladder data available")
            
            fig = go.Figure()
            
            # Add buy orders
            fig.add_trace(go.Scatter(
                x=buy_depths,
                y=buy_prices,
                mode='markers+lines',
                name='Buy Orders',
                marker=dict(size=8, color=self.colors['buy']),
                line=dict(color=self.colors['buy'], width=2),
                hovertemplate=self._create_hover_template('ladder'),
                customdata=np.column_stack([
                    data.get('buy_quantities', np.zeros_like(buy_depths)),
                    buy_prices * data.get('buy_quantities', np.zeros_like(buy_depths))
                ])
            ))
            
            # Add sell orders
            fig.add_trace(go.Scatter(
                x=sell_depths,
                y=sell_prices,
                mode='markers+lines',
                name='Sell Orders',
                marker=dict(size=8, color=self.colors['sell']),
                line=dict(color=self.colors['sell'], width=2),
                hovertemplate=self._create_hover_template('ladder'),
                customdata=np.column_stack([
                    data.get('sell_quantities', np.zeros_like(sell_depths)),
                    sell_prices * data.get('sell_quantities', np.zeros_like(sell_depths))
                ])
            ))
            
            # Add current price line
            current_price = data.get('current_price', 100.0)
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color=self.colors['neutral'],
                annotation_text=f"Current Price: ${current_price:.4f}"
            )
            
            return self._apply_common_layout(fig, f"Ladder Configuration ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Ladder Configuration", str(e))


class TouchProbabilityChart(ProbabilityChart):
    """Touch probability curves chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create touch probability curves chart"""
        try:
            depths, probabilities, weibull_params = self._extract_probability_data(data)
            
            if len(depths) == 0:
                return self._create_empty_chart("Touch Probability Curves", "No probability data available")
            
            fig = go.Figure()
            
            # Add probability curve
            fig.add_trace(go.Scatter(
                x=depths,
                y=probabilities,
                mode='lines+markers',
                name='Touch Probability',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6),
                hovertemplate=self._create_hover_template('probability')
            ))
            
            # Add Weibull fit if available
            if weibull_params and 'buy' in weibull_params:
                theta = weibull_params['buy']['theta']
                p = weibull_params['buy']['p']
                
                # Generate smooth curve for Weibull fit
                smooth_depths = np.linspace(depths.min(), depths.max(), 100)
                from analysis import weibull_touch_probability
                weibull_probs = np.array([weibull_touch_probability(d, theta, p) for d in smooth_depths])
                
                fig.add_trace(go.Scatter(
                    x=smooth_depths,
                    y=weibull_probs,
                    mode='lines',
                    name='Weibull Fit',
                    line=dict(color=self.colors['secondary'], width=2, dash='dash'),
                    hovertemplate=self._create_hover_template('probability')
                ))
            
            return self._apply_common_layout(fig, f"Touch Probability Curves ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Touch Probability Curves", str(e))


class RungTouchProbabilitiesChart(ProbabilityChart):
    """Rung touch probabilities chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create rung touch probabilities chart"""
        try:
            depths, probabilities, weibull_params = self._extract_probability_data(data)
            
            if len(depths) == 0:
                return self._create_empty_chart("Rung Touch Probabilities", "No probability data available")
            
            fig = go.Figure()
            
            # Create bar chart
            fig.add_trace(go.Bar(
                x=[f"Rung {i+1}" for i in range(len(depths))],
                y=probabilities,
                name='Touch Probability',
                marker_color=self.colors['info'],
                hovertemplate=self._create_hover_template('probability')
            ))
            
            return self._apply_common_layout(fig, f"Rung Touch Probabilities ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Rung Touch Probabilities", str(e))


class HistoricalTouchFrequencyChart(BaseChart):
    """Historical touch frequency chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create historical touch frequency chart"""
        try:
            buy_touch_probs = data.get('buy_touch_probs', [])
            sell_touch_probs = data.get('sell_touch_probs', [])
            
            if len(buy_touch_probs) == 0:
                return self._create_empty_chart("Historical Touch Frequency", "No touch probability data available")
            
            fig = go.Figure()
            
            # Create rung labels
            rung_labels = [f"Rung {i+1}" for i in range(len(buy_touch_probs))]
            
            # Add buy touch probabilities
            fig.add_trace(go.Bar(
                x=rung_labels,
                y=buy_touch_probs,
                name='Buy Touch Probability',
                marker_color=self.colors['buy'],
                hovertemplate='<b>%{x}</b><br>Buy Touch Prob: %{y:.2%}<extra></extra>'
            ))
            
            # Add sell touch probabilities
            if len(sell_touch_probs) > 0:
                fig.add_trace(go.Bar(
                    x=rung_labels,
                    y=sell_touch_probs,
                    name='Sell Touch Probability',
                    marker_color=self.colors['sell'],
                    hovertemplate='<b>%{x}</b><br>Sell Touch Prob: %{y:.2%}<extra></extra>'
                ))
            
            return self._apply_common_layout(fig, f"Historical Touch Frequency ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Historical Touch Frequency", str(e))


class ProfitDistributionChart(BaseChart):
    """Profit distribution chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create profit distribution chart"""
        try:
            actual_profits = data.get('actual_profits', [])
            profit_per_pair = data.get('profit_per_pair', [])
            
            if len(actual_profits) == 0:
                return self._create_empty_chart("Profit Distribution", "No profit data available")
            
            fig = go.Figure()
            
            # Create rung labels
            rung_labels = [f"Rung {i+1}" for i in range(len(actual_profits))]
            
            # Add actual profits
            fig.add_trace(go.Bar(
                x=rung_labels,
                y=actual_profits,
                name='Actual Profit',
                marker_color=self.colors['success'],
                hovertemplate='<b>%{x}</b><br>Actual Profit: $%{y:.2f}<extra></extra>'
            ))
            
            # Add profit per pair if available
            if len(profit_per_pair) > 0:
                fig.add_trace(go.Bar(
                    x=rung_labels,
                    y=profit_per_pair,
                    name='Profit per Pair',
                    marker_color=self.colors['info'],
                    hovertemplate='<b>%{x}</b><br>Profit per Pair: $%{y:.2f}<extra></extra>'
                ))
            
            return self._apply_common_layout(fig, f"Profit Distribution ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Profit Distribution", str(e))


class RiskReturnProfileChart(BaseChart):
    """Risk-return profile chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create risk-return profile chart"""
        try:
            buy_touch_probs = data.get('buy_touch_probs', [])
            actual_profits = data.get('actual_profits', [])
            
            if len(buy_touch_probs) == 0 or len(actual_profits) == 0:
                return self._create_empty_chart("Risk-Return Profile", "No risk-return data available")
            
            fig = go.Figure()
            
            # Calculate risk as inverse of touch probability (higher touch prob = lower risk)
            risks = 1 - buy_touch_probs
            returns = actual_profits
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=risks,
                y=returns,
                mode='markers+lines',
                name='Risk-Return',
                marker=dict(
                    color=self.colors['primary'],
                    size=10,
                    line=dict(width=1, color='white')
                ),
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='<b>Rung %{pointIndex+1}</b><br>Risk: %{x:.2%}<br>Return: $%{y:.2f}<extra></extra>'
            ))
            
            return self._apply_common_layout(fig, f"Risk-Return Profile ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Risk-Return Profile", str(e))


class TouchVsTimeChart(BaseChart):
    """Touch vs time chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create touch vs time chart"""
        try:
            buy_touch_probs = data.get('buy_touch_probs', [])
            sell_touch_probs = data.get('sell_touch_probs', [])
            
            if len(buy_touch_probs) == 0:
                return self._create_empty_chart("Touch vs Time", "No touch probability data available")
            
            fig = go.Figure()
            
            # Create rung labels
            rung_labels = [f"Rung {i+1}" for i in range(len(buy_touch_probs))]
            
            # Add buy touch probabilities over time
            fig.add_trace(go.Scatter(
                x=rung_labels,
                y=buy_touch_probs,
                mode='lines+markers',
                name='Buy Touch Probability',
                line=dict(color=self.colors['buy'], width=2),
                marker=dict(size=6),
                hovertemplate='<b>%{x}</b><br>Buy Touch Prob: %{y:.2%}<extra></extra>'
            ))
            
            # Add sell touch probabilities over time
            if len(sell_touch_probs) > 0:
                fig.add_trace(go.Scatter(
                    x=rung_labels,
                    y=sell_touch_probs,
                    mode='lines+markers',
                    name='Sell Touch Probability',
                    line=dict(color=self.colors['sell'], width=2),
                    marker=dict(size=6),
                    hovertemplate='<b>%{x}</b><br>Sell Touch Prob: %{y:.2%}<extra></extra>'
                ))
            
            return self._apply_common_layout(fig, f"Touch vs Time ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Touch vs Time", str(e))


class AllocationDistributionChart(BaseChart):
    """Allocation distribution chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create allocation distribution chart"""
        try:
            allocations = data.get('buy_quantities', np.array([]))
            
            if len(allocations) == 0:
                return self._create_empty_chart("Allocation Distribution", "No allocation data available")
            
            fig = go.Figure()
            
            # Create pie chart
            fig.add_trace(go.Pie(
                labels=[f"Rung {i+1}" for i in range(len(allocations))],
                values=allocations,
                name="Allocation",
                hovertemplate=self._create_hover_template('default')
            ))
            
            return self._apply_common_layout(fig, f"Allocation Distribution ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Allocation Distribution", str(e))


class FitQualityDashboardChart(BaseChart):
    """Fit quality dashboard chart"""
    
    def create(self, data: Dict[str, Any], timeframe_display: str = "") -> go.Figure:
        """Create fit quality dashboard chart"""
        try:
            weibull_params = data.get('weibull_params', {})
            expected_monthly_fills = data.get('expected_monthly_fills', 0)
            expected_monthly_profit = data.get('expected_monthly_profit', 0)
            expected_profit_per_dollar = data.get('expected_profit_per_dollar', 0)
            
            if not weibull_params:
                return self._create_empty_chart("Fit Quality Dashboard", "No quality data available")
            
            fig = go.Figure()
            
            # Create quality metrics
            metrics = []
            values = []
            
            # Add Weibull parameters
            if 'buy' in weibull_params:
                buy_params = weibull_params['buy']
                metrics.extend(['Buy Theta', 'Buy P'])
                values.extend([buy_params.get('theta', 0), buy_params.get('p', 0)])
            
            if 'sell' in weibull_params:
                sell_params = weibull_params['sell']
                metrics.extend(['Sell Theta', 'Sell P'])
                values.extend([sell_params.get('theta', 0), sell_params.get('p', 0)])
            
            # Add performance metrics
            metrics.extend(['Monthly Fills', 'Monthly Profit', 'Profit per $'])
            values.extend([float(expected_monthly_fills), float(expected_monthly_profit), float(expected_profit_per_dollar)])
            
            # Add bars
            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                name='Quality Metrics',
                marker_color=self.colors['primary'],
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
            ))
            
            return self._apply_common_layout(fig, f"Fit Quality Dashboard ({timeframe_display})")
            
        except Exception as e:
            return self._create_error_chart("Fit Quality Dashboard", str(e))


class ChartFactory:
    """Factory for creating chart instances"""
    
    _charts = {
        'ladder-configuration-chart': LadderConfigurationChart,
        'touch-probability-curves': TouchProbabilityChart,
        'rung-touch-probabilities': RungTouchProbabilitiesChart,
        'historical-touch-frequency': HistoricalTouchFrequencyChart,
        'profit-distribution': ProfitDistributionChart,
        'risk-return-profile': RiskReturnProfileChart,
        'touch-vs-time': TouchVsTimeChart,
        'allocation-distribution': AllocationDistributionChart,
        'fit-quality-dashboard': FitQualityDashboardChart,
    }
    
    @classmethod
    def create_chart(cls, chart_type: str) -> BaseChart:
        """Create a chart instance"""
        chart_class = cls._charts.get(chart_type)
        if not chart_class:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        return chart_class()
    
    @classmethod
    def get_available_charts(cls) -> list:
        """Get list of available chart types"""
        return list(cls._charts.keys())
