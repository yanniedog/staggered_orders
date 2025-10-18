"""
GUI Visualization Engine
Creates all interactive charts and visualizations for the ladder GUI.
Uses Plotly for professional, interactive visualizations.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class VisualizationEngine:
    """Creates all visualizations for the interactive GUI"""
    
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
        
        # Chart styling
        self.layout_template = {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff', 'size': 12},
            'xaxis': {'gridcolor': '#444444', 'color': '#ffffff'},
            'yaxis': {'gridcolor': '#444444', 'color': '#ffffff'},
            'colorway': [self.colors['primary'], self.colors['secondary'], 
                        self.colors['success'], self.colors['danger'], 
                        self.colors['warning'], self.colors['info']]
        }
    
    def create_all_charts(self, ladder_data: Dict, timeframe_hours: int) -> Tuple:
        """Create all visualization charts"""
        try:
            charts = (
                self.create_ladder_configuration_chart(ladder_data),
                self.create_touch_probability_curves(ladder_data),
                self.create_rung_touch_probabilities_chart(ladder_data),
                self.create_historical_touch_frequency_chart(ladder_data, timeframe_hours),
                self.create_profit_distribution_chart(ladder_data),
                self.create_risk_return_profile_chart(ladder_data),
                self.create_touch_vs_time_chart(ladder_data, timeframe_hours),
                self.create_allocation_distribution_chart(ladder_data),
                self.create_fit_quality_dashboard(ladder_data)
            )
            return charts
        except Exception as e:
            print(f"Error creating charts: {e}")
            return self._create_error_charts()
    
    def create_ladder_configuration_chart(self, ladder_data: Dict) -> go.Figure:
        """Create interactive ladder configuration scatter plot"""
        try:
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            buy_prices = ladder_data['buy_prices']
            sell_prices = ladder_data['sell_prices']
            buy_allocations = ladder_data['buy_allocations']
            current_price = ladder_data['current_price']
            
            # Create figure
            fig = go.Figure()
            
            # Add current price line
            fig.add_hline(y=current_price, line_dash="dash", line_color="white", 
                         annotation_text="Current Price", annotation_position="top right")
            
            # Add buy orders (scatter)
            fig.add_trace(go.Scatter(
                x=buy_prices,
                y=buy_allocations,
                mode='markers+lines',
                name='Buy Orders',
                marker=dict(
                    size=12,
                    color=self.colors['buy'],
                    line=dict(width=2, color='white')
                ),
                line=dict(color=self.colors['buy'], width=3),
                hovertemplate='<b>Buy Order</b><br>' +
                             'Price: $%{x:.2f}<br>' +
                             'Allocation: $%{y:,.0f}<br>' +
                             'Depth: %{customdata:.2f}%<extra></extra>',
                customdata=buy_depths
            ))
            
            # Add sell orders (scatter)
            fig.add_trace(go.Scatter(
                x=sell_prices,
                y=buy_allocations,  # Same allocation for visualization
                mode='markers+lines',
                name='Sell Orders',
                marker=dict(
                    size=12,
                    color=self.colors['sell'],
                    line=dict(width=2, color='white')
                ),
                line=dict(color=self.colors['sell'], width=3),
                hovertemplate='<b>Sell Order</b><br>' +
                             'Price: $%{x:.2f}<br>' +
                             'Allocation: $%{y:,.0f}<br>' +
                             'Depth: %{customdata:.2f}%<extra></extra>',
                customdata=sell_depths
            ))
            
            # Update layout
            fig.update_layout(
                title="Ladder Configuration",
                xaxis_title="Price ($)",
                yaxis_title="Allocation ($)",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating ladder configuration chart: {e}")
            return self._create_empty_chart("Ladder Configuration", "Error loading data")
    
    def create_touch_probability_curves(self, ladder_data: Dict) -> go.Figure:
        """Create touch probability curves for buy and sell sides"""
        try:
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            weibull_params = ladder_data['weibull_params']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Buy-Side Touch Probability', 'Sell-Side Touch Probability'),
                horizontal_spacing=0.1
            )
            
            # Buy-side plot
            fig.add_trace(go.Scatter(
                x=buy_depths,
                y=buy_touch_probs,
                mode='markers+lines',
                name='Buy Touch Prob',
                marker=dict(color=self.colors['buy'], size=8),
                line=dict(color=self.colors['buy'], width=3)
            ), row=1, col=1)
            
            # Sell-side plot
            fig.add_trace(go.Scatter(
                x=sell_depths,
                y=sell_touch_probs,
                mode='markers+lines',
                name='Sell Touch Prob',
                marker=dict(color=self.colors['sell'], size=8),
                line=dict(color=self.colors['sell'], width=3)
            ), row=1, col=2)
            
            # Update layout
            fig.update_layout(
                title="Touch Probability Curves",
                **self.layout_template,
                height=400,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Depth (%)", row=1, col=1)
            fig.update_xaxes(title_text="Depth (%)", row=1, col=2)
            fig.update_yaxes(title_text="Probability", row=1, col=1)
            fig.update_yaxes(title_text="Probability", row=1, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch probability curves: {e}")
            return self._create_empty_chart("Touch Probability Curves", "Error loading data")
    
    def create_rung_touch_probabilities_chart(self, ladder_data: Dict) -> go.Figure:
        """Create bar chart of individual rung touch probabilities"""
        try:
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            joint_probs = ladder_data['joint_probs']
            
            rung_numbers = list(range(1, len(buy_touch_probs) + 1))
            
            fig = go.Figure()
            
            # Add buy probabilities
            fig.add_trace(go.Bar(
                x=rung_numbers,
                y=buy_touch_probs,
                name='Buy Touch Prob',
                marker_color=self.colors['buy'],
                opacity=0.8
            ))
            
            # Add sell probabilities
            fig.add_trace(go.Bar(
                x=rung_numbers,
                y=sell_touch_probs,
                name='Sell Touch Prob',
                marker_color=self.colors['sell'],
                opacity=0.8
            ))
            
            # Add joint probabilities
            fig.add_trace(go.Bar(
                x=rung_numbers,
                y=joint_probs,
                name='Joint Prob',
                marker_color=self.colors['warning'],
                opacity=0.8
            ))
            
            fig.update_layout(
                title="Individual Rung Touch Probabilities",
                xaxis_title="Rung Number",
                yaxis_title="Probability",
                barmode='group',
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating rung touch probabilities chart: {e}")
            return self._create_empty_chart("Rung Touch Probabilities", "Error loading data")
    
    def create_historical_touch_frequency_chart(self, ladder_data: Dict, timeframe_hours: int) -> go.Figure:
        """Create histogram of historical touch frequency"""
        try:
            buy_depths = ladder_data['buy_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            
            # Convert probabilities to expected frequencies
            expected_frequencies = buy_touch_probs * timeframe_hours
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[f"{depth:.1f}%" for depth in buy_depths],
                y=expected_frequencies,
                name='Expected Touches',
                marker_color=self.colors['info'],
                opacity=0.8,
                hovertemplate='<b>Depth: %{x}</b><br>' +
                             'Expected Touches: %{y:.1f}<br>' +
                             'Timeframe: {timeframe_hours}h<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Historical Touch Frequency ({timeframe_hours}h window)",
                xaxis_title="Depth Level",
                yaxis_title="Expected Touches",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating historical touch frequency chart: {e}")
            return self._create_empty_chart("Historical Touch Frequency", "Error loading data")
    
    def create_profit_distribution_chart(self, ladder_data: Dict) -> go.Figure:
        """Create profit distribution visualization"""
        try:
            profit_per_pair = ladder_data['profit_per_pair']
            actual_profits = ladder_data['actual_profits']
            buy_allocations = ladder_data['buy_allocations']
            
            # Calculate weighted profits
            weighted_profits = profit_per_pair * buy_allocations / np.sum(buy_allocations)
            
            fig = go.Figure()
            
            # Add profit distribution
            fig.add_trace(go.Box(
                y=profit_per_pair,
                name='Profit per Pair (%)',
                marker_color=self.colors['success'],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig.update_layout(
                title="Profit Distribution",
                yaxis_title="Profit (%)",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating profit distribution chart: {e}")
            return self._create_empty_chart("Profit Distribution", "Error loading data")
    
    def create_risk_return_profile_chart(self, ladder_data: Dict) -> go.Figure:
        """Create risk-return profile scatter plot"""
        try:
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            joint_probs = ladder_data['joint_probs']
            profit_per_pair = ladder_data['profit_per_pair']
            buy_depths = ladder_data['buy_depths']
            
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=joint_probs,
                y=profit_per_pair,
                mode='markers+text',
                name='Risk-Return Profile',
                marker=dict(
                    size=15,
                    color=buy_depths,
                    colorscale='Viridis',
                    colorbar=dict(title="Depth (%)"),
                    line=dict(width=2, color='white')
                ),
                text=[f"Rung {i+1}" for i in range(len(buy_depths))],
                textposition="top center",
                hovertemplate='<b>Rung %{text}</b><br>' +
                             'Joint Prob: %{x:.3f}<br>' +
                             'Profit: %{y:.1f}%<br>' +
                             'Depth: %{marker.color:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="Risk-Return Profile",
                xaxis_title="Joint Touch Probability",
                yaxis_title="Expected Profit (%)",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating risk-return profile chart: {e}")
            return self._create_empty_chart("Risk-Return Profile", "Error loading data")
    
    def create_touch_vs_time_chart(self, ladder_data: Dict, timeframe_hours: int) -> go.Figure:
        """Create touch vs time analysis chart"""
        try:
            buy_depths = ladder_data['buy_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            
            # Create mock time series data
            hours = np.arange(0, timeframe_hours, max(1, timeframe_hours // 50))
            cumulative_touches = []
            
            for depth, prob in zip(buy_depths, buy_touch_probs):
                # Mock cumulative touches over time
                touches = np.cumsum(np.random.poisson(prob * 0.1, len(hours)))
                cumulative_touches.append(touches)
            
            fig = go.Figure()
            
            # Add traces for each depth level
            for i, (depth, touches) in enumerate(zip(buy_depths, cumulative_touches)):
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=touches,
                    mode='lines',
                    name=f'{depth:.1f}% depth',
                    line=dict(width=2),
                    hovertemplate=f'<b>{depth:.1f}% Depth</b><br>' +
                                 'Time: %{x}h<br>' +
                                 'Cumulative Touches: %{y}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Touch vs Time Analysis",
                xaxis_title="Time (hours)",
                yaxis_title="Cumulative Touches",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch vs time chart: {e}")
            return self._create_empty_chart("Touch vs Time", "Error loading data")
    
    def create_allocation_distribution_chart(self, ladder_data: Dict) -> go.Figure:
        """Create allocation distribution pie chart"""
        try:
            buy_allocations = ladder_data['buy_allocations']
            buy_depths = ladder_data['buy_depths']
            
            # Create labels
            labels = [f"Rung {i+1}<br>({depth:.1f}%)" for i, depth in enumerate(buy_depths)]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=buy_allocations,
                hovertemplate='<b>%{label}</b><br>' +
                             'Allocation: $%{value:,.0f}<br>' +
                             'Percentage: %{percent}<extra></extra>',
                textinfo='label+percent',
                textfont_size=10
            )])
            
            fig.update_layout(
                title="Capital Allocation Distribution",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating allocation distribution chart: {e}")
            return self._create_empty_chart("Allocation Distribution", "Error loading data")
    
    def create_fit_quality_dashboard(self, ladder_data: Dict) -> go.Figure:
        """Create fit quality dashboard with mini plots"""
        try:
            weibull_params = ladder_data['weibull_params']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Buy-Side R²', 'Sell-Side R²', 'Buy-Side RMSE', 'Sell-Side RMSE'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Get fit metrics
            buy_fit = weibull_params['buy'].get('fit_metrics', {})
            sell_fit = weibull_params['sell'].get('fit_metrics', {})
            
            buy_r2 = buy_fit.get('r_squared', 0.0)
            sell_r2 = sell_fit.get('r_squared', 0.0)
            buy_rmse = buy_fit.get('rmse', 0.0)
            sell_rmse = sell_fit.get('rmse', 0.0)
            
            # Add indicators
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=buy_r2,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Buy R²"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': self.colors['buy']},
                       'steps': [{'range': [0, 0.8], 'color': "lightgray"},
                                {'range': [0.8, 1], 'color': "gray"}]}
            ), row=1, col=1)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=sell_r2,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sell R²"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': self.colors['sell']},
                       'steps': [{'range': [0, 0.8], 'color': "lightgray"},
                                {'range': [0.8, 1], 'color': "gray"}]}
            ), row=1, col=2)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=buy_rmse,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Buy RMSE"},
                gauge={'axis': {'range': [None, 0.1]},
                       'bar': {'color': self.colors['warning']},
                       'steps': [{'range': [0, 0.05], 'color': "lightgray"},
                                {'range': [0.05, 0.1], 'color': "gray"}]}
            ), row=2, col=1)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=sell_rmse,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sell RMSE"},
                gauge={'axis': {'range': [None, 0.1]},
                       'bar': {'color': self.colors['warning']},
                       'steps': [{'range': [0, 0.05], 'color': "lightgray"},
                                {'range': [0.05, 0.1], 'color': "gray"}]}
            ), row=2, col=2)
            
            fig.update_layout(
                title="Fit Quality Dashboard",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating fit quality dashboard: {e}")
            return self._create_empty_chart("Fit Quality Dashboard", "Error loading data")
    
    def _create_empty_chart(self, title: str, message: str) -> go.Figure:
        """Create empty chart with error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title=title,
            **self.layout_template,
            height=400
        )
        return fig
    
    def _create_error_charts(self) -> Tuple:
        """Create error charts when main creation fails"""
        error_chart = self._create_empty_chart("Error", "Failed to load data")
        return (error_chart,) * 9
