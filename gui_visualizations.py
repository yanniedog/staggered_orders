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
        
        # Chart styling with performance optimizations
        self.layout_template = {
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': '#ffffff', 'size': 12},
            'xaxis': {'gridcolor': '#444444', 'color': '#ffffff', 'showgrid': True},
            'yaxis': {'gridcolor': '#444444', 'color': '#ffffff', 'showgrid': True},
            'colorway': [self.colors['primary'], self.colors['secondary'],
                        self.colors['success'], self.colors['danger'],
                        self.colors['warning'], self.colors['info']],
            # Performance optimizations
            'modebar': {'remove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']},
            'config': {'displayModeBar': False, 'responsive': True}
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

            # Performance optimization: limit to max 50 points per side for very large ladders
            max_points = 50
            if len(buy_prices) > max_points:
                step = len(buy_prices) // max_points
                buy_indices = range(0, len(buy_prices), step)[:max_points]
                sell_indices = range(0, len(sell_prices), step)[:max_points]
            else:
                buy_indices = range(len(buy_prices))
                sell_indices = range(len(sell_prices))
            
            # Create figure
            fig = go.Figure()
            
            # Add current price line
            fig.add_hline(y=current_price, line_dash="dash", line_color="white", 
                         annotation_text="Current Price", annotation_position="top right")
            
            # Use filtered data for performance
            buy_prices_filtered = [buy_prices[i] for i in buy_indices]
            sell_prices_filtered = [sell_prices[i] for i in sell_indices]
            buy_depths_filtered = [buy_depths[i] for i in buy_indices]
            sell_depths_filtered = [sell_depths[i] for i in sell_indices]
            buy_allocations_filtered = [buy_allocations[i] for i in buy_indices]
            sell_allocations_filtered = [ladder_data.get('sell_allocations', ladder_data['sell_quantities'] * sell_prices)[i] for i in sell_indices]

            # Calculate marker sizes based on allocation (normalized)
            max_allocation = np.max(buy_allocations) if buy_allocations.size > 0 else 1
            min_marker_size = 8
            max_marker_size = 25
            buy_marker_sizes = [min_marker_size + (alloc / max_allocation) * (max_marker_size - min_marker_size)
                              for alloc in buy_allocations_filtered]

            # Calculate buy volumes (quantity of coins)
            buy_volumes = [alloc / price for alloc, price in zip(buy_allocations_filtered, buy_prices_filtered)]

            # Add buy orders (scatter) - price on y-axis, allocation as marker size
            fig.add_trace(go.Scatter(
                x=list(range(len(buy_prices_filtered))),  # Use index for x-axis
                y=buy_prices_filtered,
                mode='markers',
                name='Buy Orders',
                marker=dict(
                    size=buy_marker_sizes,
                    color=self.colors['buy'],
                    line=dict(width=1, color='white'),  # Reduced line width for performance
                    opacity=0.8
                ),
                hovertemplate='<b>Buy Order</b><br>' +
                             'Price: $%{y:.2f}<br>' +
                             'Allocation: $%{customdata[0]:,.0f}<br>' +
                             'Volume: %{customdata[1]:,.2f} coins<br>' +
                             'Depth: %{text:.2f}%<extra></extra>',
                customdata=np.column_stack([buy_allocations_filtered, buy_volumes]),
                text=buy_depths_filtered
            ))

            # Add sell orders (scatter)
            sell_volumes = [ladder_data['sell_quantities'][i] for i in sell_indices]
            sell_marker_sizes = [min_marker_size + (alloc / max_allocation) * (max_marker_size - min_marker_size)
                               for alloc in sell_allocations_filtered]

            fig.add_trace(go.Scatter(
                x=list(range(len(sell_prices_filtered))),  # Use index for x-axis
                y=sell_prices_filtered,
                mode='markers',
                name='Sell Orders',
                marker=dict(
                    size=sell_marker_sizes,
                    color=self.colors['sell'],
                    line=dict(width=1, color='white'),  # Reduced line width for performance
                    opacity=0.8
                ),
                hovertemplate='<b>Sell Order</b><br>' +
                             'Price: $%{y:.2f}<br>' +
                             'Allocation: $%{customdata[0]:,.0f}<br>' +
                             'Volume: %{customdata[1]:,.2f} coins<br>' +
                             'Depth: %{text:.2f}%<extra></extra>',
                customdata=np.column_stack([sell_allocations_filtered, sell_volumes]),
                text=sell_depths_filtered
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
        """Create mirrored touch probability chart with y-axis in center"""
        try:
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            current_price = ladder_data['current_price']
            
            fig = go.Figure()
            
            # Buy side (green) - negative x-values (left of y-axis)
            # Mirror the depths so they appear on left
            buy_x = -buy_depths  # Negative for left side
            
            fig.add_trace(go.Scatter(
                x=buy_x,
                y=buy_touch_probs,
                mode='lines+markers',
                name='Buy Touch Probability',
                line=dict(color=self.colors['buy'], width=3),
                marker=dict(size=8, color=self.colors['buy']),
                hovertemplate='<b>Buy Side</b><br>' +
                             'Depth: %{customdata:.2f}%<br>' +
                             'Price: $%{text:.2f}<br>' +
                             'Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=buy_depths,
                text=current_price * (1 - buy_depths / 100)
            ))
            
            # Sell side (red) - positive x-values (right of y-axis)
            fig.add_trace(go.Scatter(
                x=sell_depths,
                y=sell_touch_probs,
                mode='lines+markers',
                name='Sell Touch Probability',
                line=dict(color=self.colors['sell'], width=3),
                marker=dict(size=8, color=self.colors['sell']),
                hovertemplate='<b>Sell Side</b><br>' +
                             'Depth: +%{x:.2f}%<br>' +
                             'Price: $%{customdata:.2f}<br>' +
                             'Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=current_price * (1 + sell_depths / 100)
            ))
            
            # Add vertical line at x=0 (current price / y-axis)
            fig.add_vline(x=0, line_dash="dash", line_color="white", 
                         annotation_text=f"Current: ${current_price:.2f}", 
                         annotation_position="top")
            
            title = "Touch Probability Distribution (Mirrored)"
            if timeframe_display:
                title += f" ({timeframe_display})"

            fig.update_layout(
                title=title,
                xaxis_title="← Buy Depth (%) | Current Price | Sell Depth (%) →",
                yaxis_title="Touch Probability",
                **self.layout_template,
                height=400,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='#ffffff',
                    borderwidth=1
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch probability curves: {e}")
            return self._create_empty_chart("Touch Probability Curves", "Error loading data")
    
    def create_rung_touch_probabilities_chart(self, ladder_data: Dict, timeframe_display: str = "") -> go.Figure:
        """Create combined buy/sell touch probability curves"""
        try:
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            current_price = ladder_data['current_price']
            
            fig = go.Figure()
            
            # Add buy curve (green) - depths below current price
            fig.add_trace(go.Scatter(
                x=buy_depths,
                y=buy_touch_probs,
                mode='lines+markers',
                name='Buy Touch Probability',
                line=dict(color=self.colors['buy'], width=3),
                marker=dict(size=8, color=self.colors['buy']),
                hovertemplate='<b>Buy Side</b><br>' +
                             'Depth: -%{x:.2f}%<br>' +
                             'Price: $%{customdata:.2f}<br>' +
                             'Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=current_price * (1 - buy_depths / 100)
            ))
            
            # Add sell curve (red) - depths above current price
            fig.add_trace(go.Scatter(
                x=sell_depths,
                y=sell_touch_probs,
                mode='lines+markers',
                name='Sell Touch Probability',
                line=dict(color=self.colors['sell'], width=3),
                marker=dict(size=8, color=self.colors['sell']),
                hovertemplate='<b>Sell Side</b><br>' +
                             'Depth: +%{x:.2f}%<br>' +
                             'Price: $%{customdata:.2f}<br>' +
                             'Touch Prob: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=current_price * (1 + sell_depths / 100)
            ))
            
            # Add vertical line at current price (x=0 representing 0% depth)
            fig.add_vline(x=0, line_dash="dash", line_color="white", 
                         annotation_text=f"Current: ${current_price:.2f}", 
                         annotation_position="top")
            
            title = "Touch Probability Curves (Buy & Sell)"
            if timeframe_display:
                title += f" ({timeframe_display})"

            fig.update_layout(
                title=title,
                xaxis_title="Depth from Current Price (%)",
                yaxis_title="Touch Probability",
                **self.layout_template,
                height=400,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='#ffffff',
                    borderwidth=1
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating touch probability curves: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_chart("Touch Probability Curves", f"Error: {str(e)}")
    
    def create_historical_touch_frequency_chart(self, ladder_data: Dict, timeframe_hours: int, timeframe_display: str = "") -> go.Figure:
        """Create histogram of historical touch frequency for both buy and sell ranges"""
        try:
            buy_depths = ladder_data['buy_depths']
            sell_depths = ladder_data['sell_depths']
            buy_prices = ladder_data['buy_prices']
            sell_prices = ladder_data['sell_prices']
            current_price = ladder_data['current_price']
            
            fig = go.Figure()
            
            # Use HistoricalAnalyzer to get real touch frequency data if available
            if self.historical_analyzer:
                # Buy side frequencies
                buy_touch_data = self.historical_analyzer.analyze_touch_frequency(
                    buy_depths, timeframe_hours, current_price
                )
                buy_frequencies = buy_touch_data['frequencies_per_day']
                
                # Sell side frequencies
                sell_touch_data = self.historical_analyzer.analyze_touch_frequency(
                    sell_depths, timeframe_hours, current_price
                )
                sell_frequencies = sell_touch_data['frequencies_per_day']
            else:
                # Fallback to Weibull-based calculation
                buy_touch_probs = ladder_data['buy_touch_probs']
                sell_touch_probs = ladder_data['sell_touch_probs']
                buy_frequencies = buy_touch_probs * timeframe_hours / 24
                sell_frequencies = sell_touch_probs * timeframe_hours / 24
            
            # Create price labels for x-axis (lowest to highest)
            # Reverse buy prices since they're stored high to low
            buy_prices_sorted = buy_prices[::-1]
            buy_frequencies_sorted = buy_frequencies[::-1]
            
            buy_price_labels = [f"${price:.2f}" for price in buy_prices_sorted]
            sell_price_labels = [f"${price:.2f}" for price in sell_prices]
            
            # Combine all price levels in order (lowest to highest)
            all_prices = list(buy_prices_sorted) + [current_price] + list(sell_prices)
            all_price_labels = buy_price_labels + [f"${current_price:.2f}"] + sell_price_labels
            all_frequencies = list(buy_frequencies_sorted) + [0] + list(sell_frequencies)
            
            # Create bar colors (green for buy, white for current, red for sell)
            bar_colors = [self.colors['buy']] * len(buy_prices) + ['white'] + [self.colors['sell']] * len(sell_prices)
            
            # Add bars
            fig.add_trace(go.Bar(
                x=all_price_labels,
                y=all_frequencies,
                name='Touch Frequency',
                marker_color=bar_colors,
                opacity=0.8,
                hovertemplate='<b>Price: %{x}</b><br>' +
                             'Expected Touches/Day: %{y:.2f}<br>' +
                             f'Timeframe: {timeframe_hours}h<extra></extra>'
            ))
            
            # Add vertical line at current price
            current_price_index = len(buy_prices)
            fig.add_vline(
                x=current_price_index,
                line_dash="solid",
                line_color="white",
                line_width=3,
                annotation_text=f"Current: ${current_price:.2f}",
                annotation_position="top"
            )
            
            title = f"Historical Touch Frequency - Buy & Sell Ranges ({timeframe_display})"

            fig.update_layout(
                title=title,
                xaxis_title="Price Level (Buy ← Current → Sell)",
                yaxis_title="Expected Touches per Day",
                **self.layout_template,
                height=400
            )
            
            fig.update_xaxes(tickangle=-45)
            
            return fig
            
        except Exception as e:
            print(f"Error creating historical touch frequency chart: {e}")
            import traceback
            traceback.print_exc()
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
        """Create cumulative expected profit and probability metrics"""
        try:
            buy_prices = ladder_data['buy_prices']
            buy_allocations = ladder_data['buy_allocations']
            buy_touch_probs = ladder_data['buy_touch_probs']
            joint_probs = ladder_data['joint_probs']
            profit_per_pair = ladder_data['profit_per_pair']
            
            # Reverse arrays to go from lowest to highest price
            buy_prices_sorted = buy_prices[::-1]
            buy_allocations_sorted = buy_allocations[::-1]
            joint_probs_sorted = joint_probs[::-1]
            profit_per_pair_sorted = profit_per_pair[::-1]
            
            # Calculate expected profit per rung (now in low-to-high order)
            expected_profit_per_rung = joint_probs_sorted * profit_per_pair_sorted * buy_allocations_sorted
            
            # Calculate cumulative metrics
            cumulative_profit = np.cumsum(expected_profit_per_rung)
            cumulative_capital = np.cumsum(buy_allocations_sorted)
            cumulative_roi = (cumulative_profit / cumulative_capital) * 100
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cumulative Expected Profit', 'Cumulative ROI'),
                vertical_spacing=0.15
            )
            
            # Cumulative profit line
            fig.add_trace(
                go.Scatter(
                    x=[f"${price:.2f}" for price in buy_prices_sorted],
                    y=cumulative_profit,
                    mode='lines+markers',
                    name='Cumulative Profit',
                    line=dict(color=self.colors['success'], width=3),
                    marker=dict(size=8, color=self.colors['success']),
                    fill='tozeroy',
                    hovertemplate='<b>Up to Price: %{x}</b><br>' +
                                 'Total Profit: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Cumulative ROI line
            fig.add_trace(
                go.Scatter(
                    x=[f"${price:.2f}" for price in buy_prices_sorted],
                    y=cumulative_roi,
                    mode='lines+markers',
                    name='Cumulative ROI',
                    line=dict(color=self.colors['warning'], width=3),
                    marker=dict(size=8, color=self.colors['warning']),
                    hovertemplate='<b>Up to Price: %{x}</b><br>' +
                                 'ROI: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                row=2, col=1
            )
            
            title = "Cumulative Performance Metrics"
            if timeframe_display:
                title += f" ({timeframe_display})"

            fig.update_layout(
                title=title,
                **self.layout_template,
                height=500,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="", tickangle=-45, row=1, col=1)
            fig.update_xaxes(title_text="Price Level", tickangle=-45, row=2, col=1)
            fig.update_yaxes(title_text="Expected Profit ($)", row=1, col=1)
            fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating cumulative metrics chart: {e}")
            return self._create_empty_chart("Cumulative Metrics", "Error loading data")
    
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
                
                # Convert depths to prices and sort from lowest to highest price
                prices = [current_price * (1 - depth / 100) for depth in depths]
                # Reverse to get lowest to highest price
                prices_sorted = prices[::-1]
                frequencies_sorted = frequencies[::-1]
                price_levels = [f"${price:.2f}" for price in prices_sorted]
                
                fig.add_trace(go.Bar(
                    x=price_levels,
                    y=frequencies_sorted,
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
                
                # Sort from lowest to highest price
                buy_prices_sorted = buy_prices[::-1]
                frequencies_sorted = frequencies_per_day[::-1]
                price_levels = [f"${price:.2f}" for price in buy_prices_sorted]
                
                fig.add_trace(go.Bar(
                    x=price_levels,
                    y=frequencies_sorted,
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
        """Create expected value and capital efficiency visualization"""
        try:
            buy_prices = ladder_data['buy_prices']
            buy_allocations = ladder_data['buy_allocations']
            buy_touch_probs = ladder_data['buy_touch_probs']
            sell_touch_probs = ladder_data['sell_touch_probs']
            joint_probs = ladder_data['joint_probs']
            profit_per_pair = ladder_data['profit_per_pair']
            
            # Calculate expected value for each rung
            expected_values = joint_probs * profit_per_pair * buy_allocations
            
            # Calculate capital efficiency (expected profit per dollar)
            capital_efficiency = (joint_probs * profit_per_pair) * 100  # As percentage
            
            # Sort from lowest to highest price (reverse arrays)
            buy_prices_sorted = buy_prices[::-1]
            expected_values_sorted = expected_values[::-1]
            capital_efficiency_sorted = capital_efficiency[::-1]
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add expected value bars
            fig.add_trace(
                go.Bar(
                    x=[f"${price:.2f}" for price in buy_prices_sorted],
                    y=expected_values_sorted,
                    name='Expected Value ($)',
                    marker_color=self.colors['success'],
                    opacity=0.8,
                    hovertemplate='<b>Price: %{x}</b><br>' +
                                 'Expected Value: $%{y:.2f}<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )
            
            # Add capital efficiency line
            fig.add_trace(
                go.Scatter(
                    x=[f"${price:.2f}" for price in buy_prices_sorted],
                    y=capital_efficiency_sorted,
                    name='Capital Efficiency (%)',
                    line=dict(color=self.colors['warning'], width=3),
                    marker=dict(size=8, color=self.colors['warning']),
                    hovertemplate='<b>Price: %{x}</b><br>' +
                                 'Efficiency: %{y:.3f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=True
            )
            
            title = "Expected Value & Capital Efficiency by Price Level"
            if timeframe_display:
                title += f" ({timeframe_display})"

            fig.update_layout(
                title=title,
                **self.layout_template,
                height=400,
                showlegend=True
            )
            
            fig.update_xaxes(tickangle=-45)
            
            fig.update_xaxes(title_text="Price Level")
            fig.update_yaxes(title_text="Expected Value ($)", secondary_y=False)
            fig.update_yaxes(title_text="Capital Efficiency (%)", secondary_y=True)
            
            return fig
            
        except Exception as e:
            print(f"Error creating expected value chart: {e}")
            return self._create_empty_chart("Expected Value & Efficiency", "Error loading data")
    
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
