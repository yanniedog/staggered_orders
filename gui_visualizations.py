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
from utils import format_timeframe

warnings.filterwarnings('ignore')

class VisualizationEngine:
    """Creates all visualizations for the interactive GUI"""
    
    def __init__(self, historical_analyzer=None):
        self.historical_analyzer = historical_analyzer
        
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
            timeframe_display = format_timeframe(timeframe_hours)
            charts = (
                self.create_ladder_configuration_chart(ladder_data, timeframe_display),
                self.create_touch_probability_curves(ladder_data, timeframe_display),
                self.create_rung_touch_probabilities_chart(ladder_data, timeframe_display),
                self.create_historical_touch_frequency_chart(ladder_data, timeframe_hours, timeframe_display),
                self.create_profit_distribution_chart(ladder_data, timeframe_display),
                self.create_risk_return_profile_chart(ladder_data, timeframe_display),
                self.create_touch_vs_time_chart(ladder_data, timeframe_hours, timeframe_display),
                self.create_allocation_distribution_chart(ladder_data, timeframe_display),
                self.create_fit_quality_dashboard(ladder_data, timeframe_display)
            )
            return charts
        except Exception as e:
            print(f"Error creating charts: {e}")
            return self._create_error_charts()
    
    def create_ladder_configuration_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create interactive ladder configuration scatter plot showing price levels"""
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
            
            # Calculate marker sizes based on allocation (normalized)
            max_allocation = np.max(buy_allocations)
            min_marker_size = 8
            max_marker_size = 25
            buy_marker_sizes = min_marker_size + (buy_allocations / max_allocation) * (max_marker_size - min_marker_size)
            
            # Add buy orders (scatter) - price on y-axis, allocation as marker size
            fig.add_trace(go.Scatter(
                x=list(range(len(buy_prices))),  # Use index for x-axis
                y=buy_prices,
                mode='markers',
                name='Buy Orders',
                marker=dict(
                    size=buy_marker_sizes,
                    color=self.colors['buy'],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>Buy Order</b><br>' +
                             'Price: $%{y:.2f}<br>' +
                             'Allocation: $%{customdata:,.0f}<br>' +
                             'Depth: %{text:.2f}%<extra></extra>',
                customdata=buy_allocations,
                text=buy_depths
            ))
            
            # Add sell orders (scatter)
            sell_allocations = ladder_data.get('sell_allocations', ladder_data['sell_quantities'] * sell_prices)
            sell_marker_sizes = min_marker_size + (sell_allocations / max_allocation) * (max_marker_size - min_marker_size)
            
            fig.add_trace(go.Scatter(
                x=list(range(len(sell_prices))),  # Use index for x-axis
                y=sell_prices,
                mode='markers',
                name='Sell Orders',
                marker=dict(
                    size=sell_marker_sizes,
                    color=self.colors['sell'],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>Sell Order</b><br>' +
                             'Price: $%{y:.2f}<br>' +
                             'Allocation: $%{customdata:,.0f}<br>' +
                             'Depth: %{text:.2f}%<extra></extra>',
                customdata=sell_allocations,
                text=sell_depths
            ))
            
            # Update layout
            title = "Ladder Configuration"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                xaxis_title="Order Index",
                yaxis_title="Price ($)",
                **self.layout_template,
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating ladder configuration chart: {e}")
            return self._create_empty_chart("Ladder Configuration", "Error loading data")
    
    def create_touch_probability_curves(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create touch probability curves for buy and sell sides using actual price levels"""
        try:
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            buy_prices = ladder_data['buy_prices']
            sell_prices = ladder_data['sell_prices']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Buy-Side Touch Probability', 'Sell-Side Touch Probability'),
                horizontal_spacing=0.1
            )
            
            # Buy-side plot - use actual prices on x-axis
            fig.add_trace(go.Scatter(
                x=buy_prices,
                y=buy_touch_probs,
                mode='markers+lines',
                name='Buy Touch Prob',
                marker=dict(color=self.colors['buy'], size=8),
                line=dict(color=self.colors['buy'], width=3)
            ), row=1, col=1)
            
            # Sell-side plot - use actual prices on x-axis
            fig.add_trace(go.Scatter(
                x=sell_prices,
                y=sell_touch_probs,
                mode='markers+lines',
                name='Sell Touch Prob',
                marker=dict(color=self.colors['sell'], size=8),
                line=dict(color=self.colors['sell'], width=3)
            ), row=1, col=2)
            
            # Update layout
            title = "Touch Probability Curves"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                **self.layout_template,
                height=400,
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Price ($)", row=1, col=1)
            fig.update_xaxes(title_text="Price ($)", row=1, col=2)
            fig.update_yaxes(title_text="Probability", row=1, col=1)
            fig.update_yaxes(title_text="Probability", row=1, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch probability curves: {e}")
            return self._create_empty_chart("Touch Probability Curves", "Error loading data")
    
    def create_rung_touch_probabilities_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create bar chart of individual rung touch probabilities using actual price levels"""
        try:
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            joint_probs = ladder_data['joint_probs']
            buy_prices = ladder_data['buy_prices']
            sell_prices = ladder_data['sell_prices']
            
            # Create price labels for x-axis
            price_labels = [f"${price:.2f}" for price in buy_prices]
            
            fig = go.Figure()
            
            # Add buy probabilities
            fig.add_trace(go.Bar(
                x=price_labels,
                y=buy_touch_probs,
                name='Buy Touch Prob',
                marker_color=self.colors['buy'],
                opacity=0.8,
                hovertemplate='<b>Price: %{x}</b><br>' +
                             'Buy Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add sell probabilities
            fig.add_trace(go.Bar(
                x=[f"${price:.2f}" for price in sell_prices],
                y=sell_touch_probs,
                name='Sell Touch Prob',
                marker_color=self.colors['sell'],
                opacity=0.8,
                hovertemplate='<b>Price: %{x}</b><br>' +
                             'Sell Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add joint probabilities
            fig.add_trace(go.Bar(
                x=price_labels,
                y=joint_probs,
                name='Joint Prob',
                marker_color=self.colors['warning'],
                opacity=0.8,
                hovertemplate='<b>Price: %{x}</b><br>' +
                             'Joint Prob: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            title = "Individual Price Level Touch Probabilities"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                xaxis_title="Price Level",
                yaxis_title="Probability",
                barmode='group',
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating rung touch probabilities chart: {e}")
            return self._create_empty_chart("Individual Rung Touch Probabilities", "Error loading data")
    
    def create_historical_touch_frequency_chart(self, ladder_data: Dict, timeframe_hours: int, timeframe_display: str = "") -> go.Figure:
        """Create histogram of historical touch frequency using actual price levels"""
        try:
            buy_depths = ladder_data['buy_depths']
            buy_prices = ladder_data['buy_prices']
            current_price = ladder_data['current_price']
            
            # Use HistoricalAnalyzer to get real touch frequency data
            if self.historical_analyzer:
                touch_data = self.historical_analyzer.analyze_touch_frequency(
                    buy_depths, timeframe_hours, current_price
                )
                frequencies = touch_data['frequencies_per_day']
                depths = touch_data['depths']
                
                # Convert depths to prices for display
                price_levels = [f"${current_price * (1 - depth / 100):.2f}" for depth in depths]
            else:
                # Fallback to Weibull-based calculation
                buy_touch_probs = ladder_data['buy_touch_probs']
                frequencies = buy_touch_probs * timeframe_hours / 24  # Convert to per day
                price_levels = [f"${price:.2f}" for price in buy_prices]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=price_levels,
                y=frequencies,
                name='Expected Touches per Day',
                marker_color=self.colors['info'],
                opacity=0.8,
                hovertemplate='<b>Price: %{x}</b><br>' +
                             'Expected Touches/Day: %{y:.1f}<br>' +
                             f'Timeframe: {timeframe_hours}h<extra></extra>'
            ))
            
            title = f"Historical Touch Frequency ({timeframe_display} window)"
            if timeframe_display:
                title = f"Historical Touch Frequency ({timeframe_display} window)"

            fig.update_layout(
                title=title,
                xaxis_title="Price Level",
                yaxis_title="Expected Touches per Day",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating historical touch frequency chart: {e}")
            return self._create_empty_chart("Historical Touch Frequency", "Error loading data")
    
    def create_profit_distribution_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create profit distribution visualization"""
        try:
            profit_per_pair = ladder_data['profit_per_pair']
            joint_probs = ladder_data['joint_probs']
            buy_allocations = ladder_data['buy_allocations']
            
            # Calculate weighted profits (expected profit per dollar invested)
            weighted_profits = profit_per_pair * joint_probs
            
            # Create histogram bins for profit distribution
            profit_bins = np.linspace(0, np.max(profit_per_pair), 20)
            histogram_values = []
            
            for i in range(len(profit_bins) - 1):
                bin_start = profit_bins[i]
                bin_end = profit_bins[i + 1]
                
                # Find profits in this bin
                bin_mask = (profit_per_pair >= bin_start) & (profit_per_pair < bin_end)
                if np.any(bin_mask):
                    # Weight by joint probability and allocation
                    bin_weight = np.sum(joint_probs[bin_mask] * buy_allocations[bin_mask])
                    histogram_values.append(bin_weight)
                else:
                    histogram_values.append(0)
            
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Bar(
                x=[f"{profit_bins[i]:.1f}-{profit_bins[i+1]:.1f}%" for i in range(len(profit_bins)-1)],
                y=histogram_values,
                name='Expected Profit Distribution',
                marker_color=self.colors['success'],
                opacity=0.8,
                hovertemplate='<b>Profit Range: %{x}</b><br>' +
                             'Weighted Probability: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
            
            # Add individual profit points as scatter
            fig.add_trace(go.Scatter(
                x=profit_per_pair,
                y=weighted_profits,
                mode='markers',
                name='Individual Rungs',
                marker=dict(
                    size=8,
                    color=self.colors['warning'],
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>Rung Profit: %{x:.1f}%</b><br>' +
                             'Weighted Probability: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
            
            title = "Profit Distribution (Probability Weighted)"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                xaxis_title="Profit per Pair (%)",
                yaxis_title="Weighted Probability",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating profit distribution chart: {e}")
            return self._create_empty_chart("Profit Distribution", "Error loading data")
    
    def create_risk_return_profile_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create risk-return profile scatter plot using actual prices"""
        try:
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            joint_probs = ladder_data['joint_probs']
            profit_per_pair = ladder_data['profit_per_pair']
            buy_prices = ladder_data['buy_prices']
            
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=joint_probs,
                y=profit_per_pair,
                mode='markers',
                name='Risk-Return Profile',
                marker=dict(
                    size=15,
                    color=buy_prices,  # Use actual prices for color
                    colorscale='Viridis',
                    colorbar=dict(title="Price ($)"),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate='<b>Price: $%{marker.color:.2f}</b><br>' +
                             'Joint Prob: %{x:.3f}<br>' +
                             'Profit: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
            
            title = "Risk-Return Profile"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                xaxis_title="Joint Touch Probability",
                yaxis_title="Expected Profit (%)",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating risk-return profile chart: {e}")
            return self._create_empty_chart("Risk-Return Profile", "Error loading data")
    
    def create_touch_vs_time_chart(self, ladder_data: Dict, timeframe_hours: int, timeframe_display: str = "") -> go.Figure:
        """Create vertical histogram showing touch frequency at each price level"""
        try:
            buy_depths = ladder_data['buy_depths']
            buy_prices = ladder_data['buy_prices']
            buy_touch_probs = ladder_data['buy_touch_probs']
            current_price = ladder_data['current_price']
            
            fig = go.Figure()
            
            # Use HistoricalAnalyzer to get real touch frequency data
            if self.historical_analyzer:
                touch_data = self.historical_analyzer.analyze_touch_frequency(
                    buy_depths, timeframe_hours, current_price
                )
                frequencies = touch_data['frequencies_per_day']
                depths = touch_data['depths']
                
                # Convert depths to prices for display
                price_levels = [f"${current_price * (1 - depth / 100):.2f}" for depth in depths]
                
                fig.add_trace(go.Bar(
                    x=price_levels,
                    y=frequencies,
                    name='Historical Touch Frequency',
                    marker_color=self.colors['info'],
                    opacity=0.8,
                    hovertemplate='<b>Price: %{x}</b><br>' +
                                 'Touches per Day: %{y:.1f}<br>' +
                                 f'Timeframe: {timeframe_display}<extra></extra>'
                ))
            else:
                # Fallback to Weibull-based calculation
                # Convert probabilities to expected frequencies per day
                frequencies_per_day = buy_touch_probs * timeframe_hours / 24
                price_levels = [f"${price:.2f}" for price in buy_prices]
                
                fig.add_trace(go.Bar(
                    x=price_levels,
                    y=frequencies_per_day,
                    name='Expected Touch Frequency',
                    marker_color=self.colors['info'],
                    opacity=0.8,
                    hovertemplate='<b>Price: %{x}</b><br>' +
                                 'Expected Touches/Day: %{y:.1f}<br>' +
                                 f'Timeframe: {timeframe_display}<extra></extra>'
                ))
            
            title = f"Touch Frequency by Price Level ({timeframe_display} window)"
            if timeframe_display:
                title = f"Touch Frequency by Price Level ({timeframe_display} window)"

            fig.update_layout(
                title=title,
                xaxis_title="Price Level",
                yaxis_title="Touches per Day",
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch vs time chart: {e}")
            return self._create_empty_chart("Touch vs Time", "Error loading data")
    
    def create_allocation_distribution_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create allocation distribution pie chart using actual price levels"""
        try:
            buy_allocations = ladder_data['buy_allocations']
            buy_prices = ladder_data['buy_prices']
            
            # Create labels with price levels
            labels = [f"${price:.2f}" for price in buy_prices]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=buy_allocations,
                hovertemplate='<b>Price: %{label}</b><br>' +
                             'Allocation: $%{value:,.0f}<br>' +
                             'Percentage: %{percent}<extra></extra>',
                textinfo='label+percent',
                textfont_size=10
            )])
            
            title = "Capital Allocation Distribution"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
                **self.layout_template,
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating allocation distribution chart: {e}")
            return self._create_empty_chart("Allocation Distribution", "Error loading data")
    
    def create_fit_quality_dashboard(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
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
            
            title = "Fit Quality Dashboard"
            if timeframe_display:
                title += f" ({timeframe_display} analysis)"

            fig.update_layout(
                title=title,
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
