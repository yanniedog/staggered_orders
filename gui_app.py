"""
Interactive Staggered Order Ladder GUI
Main Dash application with real-time interactive visualizations.
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from functools import lru_cache
import warnings

# Import our custom modules
from gui_calculator import LadderCalculator
from gui_visualizations import VisualizationEngine
from gui_historical import HistoricalAnalyzer
from data_fetcher import get_current_price
from config import load_config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class InteractiveLadderGUI:
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=['assets/style.css']
        )
        
        # Initialize calculation engines
        self.calculator = LadderCalculator()
        self.historical = HistoricalAnalyzer()
        self.visualizer = VisualizationEngine(self.historical)

        # Track current timeframe for data interval management
        self.current_timeframe_hours = 720
        
        # Load configuration
        self.config = load_config()
        
        # State management
        self.last_update_time = 0
        self.update_debounce_ms = 300
        
        # Precalculation system
        self.precalc_cache = {}
        self.precalc_thread = None
        self.precalc_running = False
        self.precalc_progress = 0
        self.precalc_total = 0
        self.precalc_paused = False  # For user priority
        
        # Usage tracking system
        self.usage_stats = {}
        self.usage_file = 'usage_stats.json'
        self.load_usage_stats()
        
        self.setup_layout()
        self.setup_callbacks()
        self.start_precalculations()
    
    def setup_layout(self):
        """Create the main application layout"""
        self.app.layout = html.Div([
            # Left sidebar - Controls (Fixed/Floating)
            html.Div([
                # Floating indicator
                html.Div([
                    html.Span("⚙️", style={'fontSize': '20px', 'marginRight': '10px'}),
                    html.Span("Configuration Panel", style={'fontWeight': 'bold', 'color': '#ffffff'})
                ], style={
                    'backgroundColor': '#007bff',
                    'color': '#ffffff',
                    'padding': '10px',
                    'textAlign': 'center',
                    'borderRadius': '8px 8px 0 0',
                    'marginBottom': '0px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.3)'
                }),
                
                self.create_control_panel()
            ], style={
                'position': 'fixed',
                'top': '0px',
                'left': '0px',
                'width': '25%',
                'height': '100vh',
                'overflowY': 'auto',
                'backgroundColor': '#1a1a1a',
                'borderRight': '2px solid #444444',
                'zIndex': '1000',
                'padding': '0px',
                'boxShadow': '2px 0 15px rgba(0, 0, 0, 0.4)'
            }),
            
            # Right side - Visualizations (with left margin to account for fixed sidebar)
            html.Div([
                # Header (positioned to work with fixed sidebar)
                html.Div([
                    html.H1("Interactive Staggered Order Ladder", 
                           style={'textAlign': 'center', 'color': '#007bff', 'marginBottom': '20px'}),
                    html.P("Real-time visualization and analysis of order ladder configurations",
                          style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '30px'}),
                    
                    # Status indicators in top right
                    html.Div([
                        html.Div(id='precalc-status', 
                                style={'color': '#28a745', 'fontSize': '12px', 'textAlign': 'right', 
                                       'padding': '5px', 'backgroundColor': '#2d2d2d', 'borderRadius': '4px',
                                       'marginBottom': '5px'}),
                        html.Div(id='user-request-status', 
                                style={'color': '#ffc107', 'fontSize': '12px', 'textAlign': 'right', 
                                       'padding': '5px', 'backgroundColor': '#2d2d2d', 'borderRadius': '4px',
                                       'display': 'none'})  # Hidden by default
                    ], style={'position': 'absolute', 'top': '20px', 'right': '20px', 'width': '300px'})
                ], style={'padding': '20px', 'backgroundColor': '#1a1a1a', 'borderBottom': '2px solid #444444', 'marginBottom': '20px', 'position': 'relative'}),
                
                # Visualization content
                self.create_visualization_area()
            ], style={
                'marginLeft': '25%',
                'width': '75%',
                'paddingLeft': '20px',
                'paddingRight': '20px',
                'minHeight': '100vh',
                'backgroundColor': '#1a1a1a'
            }),
            
            # Hidden interval component for status updates
            dcc.Interval(
                id='precalc-interval',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            ),

            # Loading overlay
            dcc.Loading(
                id="loading",
                children=[html.Div(id="loading-output")],
                type="default",
                fullscreen=True
            ),
            
        # Store for caching calculations
        dcc.Store(id='calculation-cache'),
        dcc.Store(id='historical-cache'),
        dcc.Store(id='cache-buster', data={'timestamp': time.time()}),
            
            # Interval for periodic updates
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
        ], style={'padding': '20px', 'backgroundColor': '#1a1a1a', 'minHeight': '100vh'})
    
    def create_control_panel(self):
        """Create the left sidebar control panel"""
        return html.Div([
            html.Div([
                # Aggression Level Slider
                html.Div([
                    html.Label("Aggression Level", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Slider(
                        id='aggression-slider',
                        min=1, max=5, step=1, value=3,
                        marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Controls depth range: 1=Conservative, 5=Very Aggressive", 
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),
                
                # Number of Rungs Slider
                html.Div([
                    html.Label("Number of Rungs", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Slider(
                        id='rungs-slider',
                        min=5, max=50, step=5, value=20,
                        marks={5: '5', 10: '10', 15: '15', 20: '20', 30: '30', 40: '40', 50: '50'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("More rungs = finer granularity, smaller allocations", 
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),
                
                # Timeframe Slider
                html.Div([
                    html.Label("Analysis Timeframe", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Slider(
                        id='timeframe-slider',
                        min=0, max=7, step=1, value=2,
                        marks={
                            0: "1d",
                            1: "1w",
                            2: "1m",
                            3: "6m",
                            4: "1y",
                            5: "3y",
                            6: "5y",
                            7: "max"
                        },
                        tooltip={
                            "placement": "bottom", 
                            "always_visible": True,
                            "template": "{value}"
                        }
                    ),
                    html.Div(id='timeframe-label', style={'color': '#ffffff', 'marginTop': '10px', 'fontSize': '14px'}),
                    html.Small("Historical analysis window", 
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),
                
                # Budget Input
                html.Div([
                    html.Label("Total Budget (USD)", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Input(
                        id='budget-input',
                        type='number',
                        value=self.config['budget_usd'],
                        min=1000, max=1000000, step=1000,
                        style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                               'backgroundColor': '#3d3d3d', 'border': '1px solid #555555',
                               'color': '#ffffff', 'borderRadius': '4px'}
                    ),
                    html.Small("Total capital for ladder orders",
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),

                # Quantity Distribution Method
                html.Div([
                    html.Label("Quantity Distribution Method", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='quantity-distribution-dropdown',
                        options=self.get_sorted_quantity_options(),
                        value='kelly_optimized',
                        style={'width': '100%', 'marginBottom': '10px', 'borderRadius': '4px'},
                        className='dark-dropdown'
                    ),
                    html.Small("How quantities are distributed across ladder rungs",
                             style={'color': '#6c757d'}),
                    html.Div(id='quantity-distribution-explanation', 
                            style={'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 'fontStyle': 'italic'})
                ], style={'marginBottom': '30px'}),

                # Rung Positioning Method
                html.Div([
                    html.Label("Rung Positioning Method", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='rung-positioning-dropdown',
                        options=self.get_sorted_positioning_options(),
                        value='linear',
                        style={'width': '100%', 'marginBottom': '10px', 'borderRadius': '4px'},
                        className='dark-dropdown'
                    ),
                    html.Small("How ladder rungs are positioned across price levels",
                             style={'color': '#6c757d'}),
                    html.Div(id='rung-positioning-explanation', 
                            style={'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 'fontStyle': 'italic'})
                ], style={'marginBottom': '30px'}),
                
                # Current Price Display
                html.Div([
                    html.Label("Current Price", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    html.Div(id='current-price-display',
                           style={'fontSize': '24px', 'color': '#28a745', 'marginBottom': '10px'})
                ], style={'marginBottom': '30px'}),

                # Trading Parameters
                html.Div([
                    html.Label("Trading Parameters", style={'fontWeight': 'bold', 'color': '#ffffff', 'marginBottom': '10px'}),
                    
                    # Trading Fee
                    html.Div([
                        html.Label("Trading Fee (%)", style={'fontWeight': 'bold', 'color': '#ffffff', 'fontSize': '14px'}),
                        dcc.Input(
                            id='trading-fee-input',
                            type='number',
                            value=0.075,
                            step=0.001,
                            min=0,
                            max=1,
                            style={'width': '100%', 'padding': '8px', 'marginBottom': '5px',
                                   'backgroundColor': '#3d3d3d', 'border': '1px solid #555555',
                                   'color': '#ffffff', 'borderRadius': '4px'}
                        ),
                        html.Small("Default: 0.075% (Binance spot trading fee)",
                             style={'color': '#6c757d'})
                    ], style={'marginBottom': '15px'}),
                    
                    # Minimum Notional
                    html.Div([
                        html.Label("Minimum Notional per Trade ($)", style={'fontWeight': 'bold', 'color': '#ffffff', 'fontSize': '14px'}),
                        dcc.Input(
                            id='min-notional-input',
                            type='number',
                            value=10.0,
                            step=1,
                            min=1,
                            max=1000,
                            style={'width': '100%', 'padding': '8px', 'marginBottom': '5px',
                                   'backgroundColor': '#3d3d3d', 'border': '1px solid #555555',
                                   'color': '#ffffff', 'borderRadius': '4px'}
                        ),
                        html.Small("Default: $10 (minimum order size)",
                                 style={'color': '#6c757d'})
                    ], style={'marginBottom': '15px'})
                ], style={'marginBottom': '30px'}),
                
                # Order Tables Section
                html.Div([
                    html.Label("Order Summary", style={'fontWeight': 'bold', 'color': '#ffffff', 'fontSize': '16px', 'marginBottom': '15px'}),
                    
                    # Buy Orders Table
                    html.Div([
                        html.H5("Buy Orders", style={'color': '#28a745', 'marginBottom': '10px'}),
                        html.Div(id='buy-orders-table', style={'marginBottom': '20px'})
                    ]),
                    
                    # Sell Orders Table
                    html.Div([
                        html.H5("Sell Orders", style={'color': '#dc3545', 'marginBottom': '10px'}),
                        html.Div(id='sell-orders-table', style={'marginBottom': '20px'})
                    ]),
                    
                    # Download Button
                    html.Div([
                        html.Button(
                            'Download Orders to CSV',
                            id='download-csv-btn',
                            style={
                                'backgroundColor': '#007bff',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'fontSize': '14px',
                                'width': '100%'
                            }
                        ),
                        dcc.Download(id="download-csv")
                    ], style={'marginTop': '20px'})
                ], style={'marginBottom': '30px'}),
                
                # Cryptocurrency Selection
                html.Div([
                    html.Label("Cryptocurrency", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='crypto-dropdown',
                        className='dark-dropdown',
                        options=self.get_sorted_crypto_options(),
                        value='SOLUSDT',
                        style={'width': '100%', 'marginBottom': '10px', 'borderRadius': '4px'}
                    ),
                    html.Small("Select cryptocurrency for ladder analysis",
                             style={'color': '#6c757d'})
                ])
            ], style={'backgroundColor': '#2d2d2d', 'padding': '20px', 'borderRadius': '0 0 8px 8px', 
                     'border': '1px solid #444444', 'margin': '0px'})
        ])
    
    def create_visualization_area(self):
        """Create the main visualization grid"""
        return html.Div([
            # Row 1: Core Ladder Visualizations
            html.Div([
                html.Div([
                    dcc.Graph(id='ladder-configuration-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='touch-probability-curves')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),
            
            # Row 2: Probability Analysis
            html.Div([
                html.Div([
                    dcc.Graph(id='rung-touch-probabilities')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='historical-touch-frequency')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),
            
            # Row 3: Profitability Metrics
            html.Div([
                html.Div([
                    dcc.Graph(id='profit-distribution')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='risk-return-profile')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),
            
            # Row 4: Time-based Analysis
            html.Div([
                html.Div([
                    dcc.Graph(id='touch-vs-time')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    self.create_kpi_cards()
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ], style={'marginBottom': '20px'}),
            
            # Row 5: Sensitivity & Validation
            html.Div([
                html.Div([
                    dcc.Graph(id='allocation-distribution')
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='fit-quality-dashboard')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
            ])
        ])
    
    def create_kpi_cards(self):
        """Create KPI cards for performance metrics"""
        return html.Div([
            html.Div([
                html.H5("Performance Metrics", style={'marginBottom': '20px', 'color': '#ffffff'})
            ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H6("Total Expected Profit", style={'color': '#6c757d', 'marginBottom': '5px'}),
                        html.H4(id='total-profit-kpi', style={'color': '#28a745', 'marginBottom': '20px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    html.Div([
                        html.H6("Monthly Fills", style={'color': '#6c757d', 'marginBottom': '5px'}),
                        html.H4(id='monthly-fills-kpi', style={'color': '#17a2b8', 'marginBottom': '20px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H6("Capital Efficiency", style={'color': '#6c757d', 'marginBottom': '5px'}),
                        html.H4(id='capital-efficiency-kpi', style={'color': '#ffc107', 'marginBottom': '20px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                    html.Div([
                        html.H6("Expected Timeframe", style={'color': '#6c757d', 'marginBottom': '5px'}),
                        html.H4(id='timeframe-kpi', style={'color': '#007bff', 'marginBottom': '20px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'})
                ])
            ], style={'backgroundColor': '#2d2d2d', 'padding': '20px', 'borderRadius': '8px', 
                     'border': '1px solid #444444'})
        ])
    
    def setup_callbacks(self):
        """Set up all Dash callbacks"""
        
        # Main calculation callback - updates all visualizations
        @self.app.callback(
            [Output('ladder-configuration-chart', 'figure'),
             Output('touch-probability-curves', 'figure'),
             Output('rung-touch-probabilities', 'figure'),
             Output('historical-touch-frequency', 'figure'),
             Output('profit-distribution', 'figure'),
             Output('risk-return-profile', 'figure'),
             Output('touch-vs-time', 'figure'),
             Output('allocation-distribution', 'figure'),
             Output('fit-quality-dashboard', 'figure'),
             Output('total-profit-kpi', 'children'),
             Output('monthly-fills-kpi', 'children'),
             Output('capital-efficiency-kpi', 'children'),
             Output('timeframe-kpi', 'children'),
             Output('buy-orders-table', 'children'),
             Output('sell-orders-table', 'children'),
             Output('calculation-cache', 'data')],
            [Input('aggression-slider', 'value'),
             Input('rungs-slider', 'value'),
             Input('timeframe-slider', 'value'),
             Input('budget-input', 'value'),
             Input('quantity-distribution-dropdown', 'value'),
             Input('crypto-dropdown', 'value'),
             Input('rung-positioning-dropdown', 'value'),
             Input('trading-fee-input', 'value'),
             Input('min-notional-input', 'value'),
             Input('cache-buster', 'data')],
            [State('calculation-cache', 'data')]
        )
        def update_all_visualizations_callback(aggression_level, num_rungs, timeframe_slider,
                                             budget, quantity_distribution, crypto_symbol, rung_positioning,
                                             trading_fee, min_notional, cache_buster, cache_data):
            # Map slider position to actual hours
            timeframe_map = {0: 24, 1: 168, 2: 720, 3: 4320, 4: 8760, 5: 26280, 6: 43800, 7: 87600}
            timeframe_hours = timeframe_map.get(timeframe_slider, 720)
            
            return self.update_all_visualizations(aggression_level, num_rungs, timeframe_hours,
                                                 budget, quantity_distribution, crypto_symbol, rung_positioning,
                                                 trading_fee, min_notional, cache_data)
        
        # Current price callback
        @self.app.callback(
            Output('current-price-display', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('crypto-dropdown', 'value')]
        )
        def update_current_price_callback(interval_n, crypto_symbol):
            return self.update_current_price(interval_n, crypto_symbol)
        
        # Timeframe label callback
        @self.app.callback(
            Output('timeframe-label', 'children'),
            [Input('timeframe-slider', 'value')]
        )
        def update_timeframe_label(slider_value):
            timeframe_labels = {0: "1 day", 1: "1 week", 2: "1 month", 3: "6 months", 
                              4: "1 year", 5: "3 years", 6: "5 years", 7: "max"}
            return f"Selected: {timeframe_labels.get(slider_value, 'Unknown')}"
        
        # CSV download callback
        @self.app.callback(
            Output("download-csv", "data"),
            [Input("download-csv-btn", "n_clicks")],
            [State('calculation-cache', 'data')]
        )
        def download_csv(n_clicks, cache_data):
            if n_clicks is None or not cache_data or 'ladder_data' not in cache_data:
                return dash.no_update
            
            try:
                ladder_data = cache_data['ladder_data']
                trading_fee = cache_data.get('trading_fee', 0.075)
                min_notional = cache_data.get('min_notional', 10.0)
                crypto_symbol = ladder_data.get('crypto_symbol', 'SOLUSDT')
                
                # Generate timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create CSV content
                csv_content = self._generate_csv_content(ladder_data, trading_fee, min_notional, crypto_symbol)
                
                # Create filename
                filename = f"ladder_orders_{crypto_symbol}_{timestamp}.csv"
                
                return dict(content=csv_content, filename=filename)
                
            except Exception as e:
                print(f"Error generating CSV: {e}")
                return dash.no_update
        
        # Quantity Distribution Method Explanation Callback
        @self.app.callback(
            Output('quantity-distribution-explanation', 'children'),
            [Input('quantity-distribution-dropdown', 'value')]
        )
        def update_quantity_distribution_explanation(method):
            explanations = {
                'kelly_optimized': "Kelly Criterion optimizes position sizes based on win probability and payoff ratio for maximum long-term growth.",
                'adaptive_kelly': "Adaptive Kelly adjusts position sizes dynamically based on recent market performance and volatility.",
                'volatility_weighted': "Volatility-weighted allocation reduces position sizes in high-volatility areas to manage risk.",
                'sharpe_maximizing': "Sharpe-maximizing allocation optimizes the risk-adjusted return ratio across all ladder positions.",
                'fibonacci_weighted': "Fibonacci-weighted allocation uses Fibonacci sequence ratios to distribute capital across rungs.",
                'risk_parity': "Risk-parity allocation equalizes risk contribution from each ladder position.",
                'price_weighted': "Price-weighted allocation distributes more capital to lower-priced rungs for better entry points.",
                'equal_notional': "Equal notional allocation distributes the same dollar amount to each ladder rung.",
                'equal_quantity': "Equal quantity allocation distributes the same number of coins to each ladder rung.",
                'linear_increase': "Linear increase allocation gradually increases position sizes from lowest to highest rungs.",
                'exponential_increase': "Exponential increase allocation exponentially increases position sizes across rungs.",
                'probability_weighted': "Probability-weighted allocation allocates more capital to rungs with higher touch probabilities."
            }
            return explanations.get(method, "Method description not available.")
        
        # Rung Positioning Method Explanation Callback
        @self.app.callback(
            Output('rung-positioning-explanation', 'children'),
            [Input('rung-positioning-dropdown', 'value')]
        )
        def update_rung_positioning_explanation(method):
            explanations = {
                'linear': "Linear spacing places rungs at equal percentage intervals from current price for consistent coverage.",
                'support_resistance': "Support/resistance clustering positions rungs near key technical levels where price often reverses.",
                'volume_profile': "Volume profile weighted positioning places rungs where high trading volume typically occurs.",
                'touch_pattern': "Touch pattern analysis positions rungs based on historical price touch frequency patterns.",
                'adaptive_probability': "Adaptive probability adjusts rung spacing based on real-time market volatility and conditions.",
                'expected_value': "Expected value optimization positions rungs to maximize the expected profit from each position.",
                'quantile': "Quantile-based positioning uses statistical quantiles to distribute rungs across price ranges.",
                'risk_weighted': "Risk-weighted positioning adjusts rung spacing based on the risk profile of each price level.",
                'exponential': "Exponential spacing places rungs closer to current price with increasing intervals further out.",
                'logarithmic': "Logarithmic spacing uses logarithmic intervals for natural price distribution patterns.",
                'fibonacci': "Fibonacci levels positioning places rungs at key Fibonacci retracement and extension levels.",
                'dynamic_density': "Dynamic density adjusts rung density based on market volatility and trading activity."
            }
            return explanations.get(method, "Method description not available.")
    
    def get_sorted_quantity_options(self):
        """Get quantity distribution options sorted by usage frequency"""
        analytics = self.get_usage_analytics()
        method_usage = analytics.get('method_distribution', {}).get('quantity', {})
        
        # Default options with usage counts
        options = [
            {'label': 'Kelly-Optimized (Recommended)', 'value': 'kelly_optimized'},
            {'label': 'Adaptive Kelly', 'value': 'adaptive_kelly'},
            {'label': 'Volatility-Weighted', 'value': 'volatility_weighted'},
            {'label': 'Sharpe-Maximizing', 'value': 'sharpe_maximizing'},
            {'label': 'Fibonacci-Weighted', 'value': 'fibonacci_weighted'},
            {'label': 'Risk-Parity', 'value': 'risk_parity'},
            {'label': 'Price-Weighted', 'value': 'price_weighted'},
            {'label': 'Equal Notional', 'value': 'equal_notional'},
            {'label': 'Equal Quantity', 'value': 'equal_quantity'},
            {'label': 'Linear Increase', 'value': 'linear_increase'},
            {'label': 'Exponential Increase', 'value': 'exponential_increase'},
            {'label': 'Probability-Weighted', 'value': 'probability_weighted'}
        ]
        
        # Sort by usage frequency (descending)
        def sort_key(option):
            usage_count = method_usage.get(option['value'], 0)
            # Put recommended first, then by usage
            if option['value'] == 'kelly_optimized':
                return (1, -usage_count)  # Recommended first
            return (0, -usage_count)  # Then by usage
        
        return sorted(options, key=sort_key)
    
    def get_sorted_positioning_options(self):
        """Get rung positioning options sorted by usage frequency"""
        analytics = self.get_usage_analytics()
        method_usage = analytics.get('method_distribution', {}).get('positioning', {})
        
        # Default options with usage counts
        options = [
            {'label': 'Linear Spacing (Recommended)', 'value': 'linear'},
            {'label': 'Support/Resistance Clustering', 'value': 'support_resistance'},
            {'label': 'Volume Profile Weighted', 'value': 'volume_profile'},
            {'label': 'Touch Pattern Analysis', 'value': 'touch_pattern'},
            {'label': 'Adaptive Probability', 'value': 'adaptive_probability'},
            {'label': 'Expected Value Optimization', 'value': 'expected_value'},
            {'label': 'Quantile-Based', 'value': 'quantile'},
            {'label': 'Risk-Weighted', 'value': 'risk_weighted'},
            {'label': 'Exponential Spacing', 'value': 'exponential'},
            {'label': 'Logarithmic Spacing', 'value': 'logarithmic'},
            {'label': 'Fibonacci Levels', 'value': 'fibonacci'},
            {'label': 'Dynamic Density', 'value': 'dynamic_density'}
        ]
        
        # Sort by usage frequency (descending)
        def sort_key(option):
            usage_count = method_usage.get(option['value'], 0)
            # Put recommended first, then by usage
            if option['value'] == 'linear':
                return (1, -usage_count)  # Recommended first
            return (0, -usage_count)  # Then by usage
        
        return sorted(options, key=sort_key)
    
    def get_sorted_crypto_options(self):
        """Get cryptocurrency options sorted by usage frequency"""
        analytics = self.get_usage_analytics()
        crypto_usage = analytics.get('crypto_distribution', {})
        
        # Default options with usage counts
        options = [
            {'label': 'Solana (SOLUSDT)', 'value': 'SOLUSDT'},
            {'label': 'Bitcoin (BTCUSDT)', 'value': 'BTCUSDT'},
            {'label': 'Ethereum (ETHUSDT)', 'value': 'ETHUSDT'},
            {'label': 'Cardano (ADAUSDT)', 'value': 'ADAUSDT'},
            {'label': 'Polkadot (DOTUSDT)', 'value': 'DOTUSDT'},
            {'label': 'Chainlink (LINKUSDT)', 'value': 'LINKUSDT'},
            {'label': 'Uniswap (UNIUSDT)', 'value': 'UNIUSDT'},
            {'label': 'Litecoin (LTCUSDT)', 'value': 'LTCUSDT'},
            {'label': 'Binance Coin (BNBUSDT)', 'value': 'BNBUSDT'},
            {'label': 'Polygon (MATICUSDT)', 'value': 'MATICUSDT'}
        ]
        
        # Sort by usage frequency (descending)
        def sort_key(option):
            usage_count = crypto_usage.get(option['value'], 0)
            # Put SOLUSDT first (default), then by usage
            if option['value'] == 'SOLUSDT':
                return (1, -usage_count)  # Default first
            return (0, -usage_count)  # Then by usage
        
        return sorted(options, key=sort_key)
        
        # Precalculation Status Callback
        @self.app.callback(
            Output('precalc-status', 'children'),
            [Input('precalc-interval', 'n_intervals')]
        )
        def update_precalc_status(n_intervals):
            return self.get_precalc_status()

        # User Request Status Callback
        @self.app.callback(
            Output('user-request-status', 'children'),
            [Input('aggression-slider', 'value'),
             Input('rungs-slider', 'value'),
             Input('timeframe-slider', 'value'),
             Input('budget-input', 'value'),
             Input('quantity-distribution-dropdown', 'value'),
             Input('crypto-dropdown', 'value'),
             Input('rung-positioning-dropdown', 'value')]
        )
        def update_user_request_status(aggression, rungs, timeframe, budget, qty_method, crypto, pos_method):
            # Show processing status when any parameter changes
            if any([aggression, rungs, timeframe, budget, qty_method, crypto, pos_method]):
                return "🔄 Processing your request..."
            return "✓ Ready"

        # User Request Status Text when Hidden
        @self.app.callback(
            Output('user-request-status', 'children'),
            [Input('user-request-status', 'style')]
        )
        def update_user_request_text_when_hidden(style):
            if style.get('display') == 'none':
                return "✓ Ready"
            return dash.no_update

        # User Request Status Visibility Callback (triggered by calculation callback)
        @self.app.callback(
            Output('user-request-status', 'style'),
            [Input('ladder-configuration-chart', 'figure'),
             Input('touch-probability-curves', 'figure'),
             Input('rung-touch-probabilities', 'figure'),
             Input('historical-touch-frequency', 'figure'),
             Input('profit-distribution', 'figure'),
             Input('risk-return-profile', 'figure'),
             Input('touch-vs-time', 'figure'),
             Input('allocation-distribution', 'figure'),
             Input('fit-quality-dashboard', 'figure')]
        )
        def update_user_request_visibility(*figures):
            # Show processing status when figures are being updated (indicates calculation in progress)
            return {'color': '#ffc107', 'fontSize': '12px', 'textAlign': 'right',
                   'padding': '5px', 'backgroundColor': '#2d2d2d', 'borderRadius': '4px',
                   'display': 'block', 'animation': 'pulse 1s infinite'}

        # Hide User Request Status when calculation completes
        @self.app.callback(
            Output('user-request-status', 'style'),
            [Input('calculation-cache', 'data')]
        )
        def hide_user_request_status(cache_data):
            # Hide status when cache is updated (calculation complete)
            return {'display': 'none'}

    
    def update_all_visualizations(self, aggression_level, num_rungs, timeframe_hours,
                                 budget, quantity_distribution, crypto_symbol, rung_positioning,
                                 trading_fee, min_notional, cache_data):
        """Main callback that updates all visualizations"""
        # Pause precalculation for user priority
        self.pause_precalculation()
        
        # Debounce updates
        current_time = time.time() * 1000
        if current_time - self.last_update_time < self.update_debounce_ms:
            self.resume_precalculation()
            return dash.no_update
        self.last_update_time = current_time
        
        try:
            # Validate inputs
            if not all([aggression_level, num_rungs, timeframe_hours, budget]):
                print("Warning: Missing input parameters")
                return dash.no_update
            
            # Try to get precalculated result first
            ladder_data = self.get_precalculated_result(
                aggression_level, num_rungs, timeframe_hours, budget, 
                quantity_distribution, crypto_symbol, rung_positioning
            )
            
            # If not precalculated, calculate now
            if ladder_data is None:
                print(f"Calculating on-demand: {crypto_symbol} {aggression_level} {num_rungs} {timeframe_hours}h {budget} {quantity_distribution} {rung_positioning}")
                ladder_data = self.calculator.calculate_ladder_configuration(
                    aggression_level, num_rungs, timeframe_hours, budget, quantity_distribution,
                    crypto_symbol, rung_positioning
                )
            else:
                print(f"Using precalculated result: {crypto_symbol} {aggression_level} {num_rungs} {timeframe_hours}h {budget} {quantity_distribution} {rung_positioning}")
            
            # Track usage of this configuration
            self.track_configuration_usage(
                aggression_level, num_rungs, timeframe_hours, budget,
                quantity_distribution, crypto_symbol, rung_positioning
            )
            
            # Validate ladder data
            if not ladder_data or 'buy_depths' not in ladder_data:
                print("Error: Invalid ladder data returned")
                return self._get_error_response(cache_data)
            
            # Update current timeframe for data interval management
            self.current_timeframe_hours = timeframe_hours

            # Generate all visualizations
            figures = self.visualizer.create_all_charts(ladder_data, timeframe_hours)
            
            # Calculate KPIs
            kpis = self.calculator.calculate_kpis(ladder_data)
            
            # Create order tables
            buy_table = self._create_buy_orders_table(ladder_data, trading_fee, min_notional)
            sell_table = self._create_sell_orders_table(ladder_data, trading_fee, min_notional)
            
            # Update cache
            cache_data = {
                'timestamp': current_time,
                'ladder_data': ladder_data,
                'kpis': kpis,
                'trading_fee': trading_fee,
                'min_notional': min_notional
            }
            
            # Resume precalculation after user request
            self.resume_precalculation()

            # Trigger status update to hide processing indicator
            from dash import no_update
            return (*figures, *kpis.values(), buy_table, sell_table, cache_data)
            
        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()
            # Resume precalculation even on error
            self.resume_precalculation()
            return self._get_error_response(cache_data)
    
    def _get_error_response(self, cache_data):
        """Get error response with empty figures"""
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Error loading data - check console for details", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        empty_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#ffffff'},
            height=400
        )
        empty_figs = (empty_fig,) * 9
        empty_table = html.Div("No data available", style={'color': '#6c757d'})
        return (*empty_figs, "N/A", "N/A", "N/A", "N/A", empty_table, empty_table, cache_data)
    
    def update_current_price(self, interval_n, crypto_symbol='SOLUSDT'):
        """Update current price display"""
        try:
            current_price = get_current_price(crypto_symbol)
            return f"${current_price:.2f}"
        except Exception as e:
            return f"Error: {e}"
    
    def _create_buy_orders_table(self, ladder_data, trading_fee, min_notional):
        """Create buy orders table"""
        try:
            buy_prices = ladder_data['buy_prices']
            buy_quantities = ladder_data['buy_quantities']
            buy_allocations = ladder_data['buy_allocations']
            
            # Sort by price (lowest to highest)
            sorted_indices = np.argsort(buy_prices)
            prices = buy_prices[sorted_indices]
            quantities = buy_quantities[sorted_indices]
            allocations = buy_allocations[sorted_indices]
            
            # Create table rows
            rows = []
            for i, (price, qty, alloc) in enumerate(zip(prices, quantities, allocations), 1):
                # Check if order meets minimum notional
                meets_min = alloc >= min_notional
                row_style = {'backgroundColor': '#2d2d2d'} if meets_min else {'backgroundColor': '#3d2d2d', 'opacity': '0.7'}
                
                rows.append(html.Tr([
                    html.Td(f"#{i}", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    html.Td(f"${price:.2f}", style={'color': '#28a745', 'fontWeight': 'bold'}),
                    html.Td(f"{qty:.4f}", style={'color': '#ffffff'}),
                    html.Td(f"${alloc:.2f}", style={'color': '#ffffff', 'fontWeight': 'bold'})
                ], style=row_style))
            
            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Order", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Price", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Quantity", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Notional", style={'color': '#ffffff', 'textAlign': 'center'})
                    ])
                ]),
                html.Tbody(rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'})
            
            return table
            
        except Exception as e:
            return html.Div(f"Error creating buy table: {e}", style={'color': '#dc3545'})
    
    def _create_sell_orders_table(self, ladder_data, trading_fee, min_notional):
        """Create sell orders table"""
        try:
            sell_prices = ladder_data['sell_prices']
            sell_quantities = ladder_data['sell_quantities']
            sell_allocations = ladder_data['sell_allocations']
            
            # Sort by price (lowest to highest)
            sorted_indices = np.argsort(sell_prices)
            prices = sell_prices[sorted_indices]
            quantities = sell_quantities[sorted_indices]
            allocations = sell_allocations[sorted_indices]
            
            # Create table rows
            rows = []
            for i, (price, qty, alloc) in enumerate(zip(prices, quantities, allocations), 1):
                # Check if order meets minimum notional
                meets_min = alloc >= min_notional
                row_style = {'backgroundColor': '#2d2d2d'} if meets_min else {'backgroundColor': '#3d2d2d', 'opacity': '0.7'}
                
                rows.append(html.Tr([
                    html.Td(f"#{i}", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    html.Td(f"${price:.2f}", style={'color': '#dc3545', 'fontWeight': 'bold'}),
                    html.Td(f"{qty:.4f}", style={'color': '#ffffff'}),
                    html.Td(f"${alloc:.2f}", style={'color': '#ffffff', 'fontWeight': 'bold'})
                ], style=row_style))
            
            table = html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Order", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Price", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Quantity", style={'color': '#ffffff', 'textAlign': 'center'}),
                        html.Th("Notional", style={'color': '#ffffff', 'textAlign': 'center'})
                    ])
                ]),
                html.Tbody(rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'})
            
            return table

        except Exception as e:
            return html.Div(f"Error creating sell table: {e}", style={'color': '#dc3545'})
    
    def _generate_csv_content(self, ladder_data, trading_fee, min_notional, crypto_symbol):
        """Generate CSV content for download"""
        try:
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'Order Type', 'Order #', 'Price ($)', 'Quantity', 'Notional ($)', 
                'Trading Fee ($)', 'Net Notional ($)', 'Meets Min Notional', 
                'Cryptocurrency', 'Timestamp'
            ])
            
            # Get current timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Process buy orders
            buy_prices = ladder_data['buy_prices']
            buy_quantities = ladder_data['buy_quantities']
            buy_allocations = ladder_data['buy_allocations']
            
            # Sort buy orders by price (lowest to highest)
            buy_sorted_indices = np.argsort(buy_prices)
            for i, idx in enumerate(buy_sorted_indices, 1):
                price = buy_prices[idx]
                qty = buy_quantities[idx]
                notional = buy_allocations[idx]
                fee = notional * (trading_fee / 100)
                net_notional = notional - fee
                meets_min = notional >= min_notional
                
                writer.writerow([
                    'BUY', i, f"{price:.2f}", f"{qty:.4f}", f"{notional:.2f}",
                    f"{fee:.2f}", f"{net_notional:.2f}", meets_min,
                    crypto_symbol, timestamp
                ])
            
            # Process sell orders
            sell_prices = ladder_data['sell_prices']
            sell_quantities = ladder_data['sell_quantities']
            sell_allocations = ladder_data['sell_allocations']
            
            # Sort sell orders by price (lowest to highest)
            sell_sorted_indices = np.argsort(sell_prices)
            for i, idx in enumerate(sell_sorted_indices, 1):
                price = sell_prices[idx]
                qty = sell_quantities[idx]
                notional = sell_allocations[idx]
                fee = notional * (trading_fee / 100)
                net_notional = notional - fee
                meets_min = notional >= min_notional
                
                writer.writerow([
                    'SELL', i, f"{price:.2f}", f"{qty:.4f}", f"{notional:.2f}",
                    f"{fee:.2f}", f"{net_notional:.2f}", meets_min,
                    crypto_symbol, timestamp
                ])
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error generating CSV content: {e}")
            return "Error generating CSV content"
    
    def load_usage_stats(self):
        """Load usage statistics from persistent storage"""
        try:
            import json
            import os
            
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    self.usage_stats = json.load(f)
                print(f"Loaded usage stats: {len(self.usage_stats)} configurations tracked")
            else:
                self.usage_stats = {}
                print("No existing usage stats found, starting fresh")
                
        except Exception as e:
            print(f"Error loading usage stats: {e}")
            self.usage_stats = {}
    
    def save_usage_stats(self):
        """Save usage statistics to persistent storage"""
        try:
            import json
            
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
            print(f"Saved usage stats: {len(self.usage_stats)} configurations")
            
        except Exception as e:
            print(f"Error saving usage stats: {e}")
    
    def track_configuration_usage(self, aggression_level, num_rungs, timeframe_hours, budget, quantity_distribution, crypto_symbol, rung_positioning):
        """Track usage of a configuration"""
        config_key = self._generate_cache_key({
            'aggression_level': aggression_level,
            'num_rungs': num_rungs,
            'timeframe_hours': timeframe_hours,
            'budget': budget,
            'quantity_distribution': quantity_distribution,
            'crypto_symbol': crypto_symbol,
            'rung_positioning': rung_positioning
        })
        
        # Increment usage count
        if config_key in self.usage_stats:
            self.usage_stats[config_key]['count'] += 1
            self.usage_stats[config_key]['last_used'] = time.time()
        else:
            self.usage_stats[config_key] = {
                'count': 1,
                'first_used': time.time(),
                'last_used': time.time(),
                'config': {
                    'aggression_level': aggression_level,
                    'num_rungs': num_rungs,
                    'timeframe_hours': timeframe_hours,
                    'budget': budget,
                    'quantity_distribution': quantity_distribution,
                    'crypto_symbol': crypto_symbol,
                    'rung_positioning': rung_positioning
                }
            }
        
        # Save stats periodically (every 10 uses)
        if self.usage_stats[config_key]['count'] % 10 == 0:
            self.save_usage_stats()
    
    def get_most_used_configurations(self, limit=20):
        """Get the most frequently used configurations"""
        if not self.usage_stats:
            return []
        
        # Sort by usage count (descending)
        sorted_configs = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Return top configurations
        return sorted_configs[:limit]
    
    def get_usage_analytics(self):
        """Get usage analytics summary"""
        if not self.usage_stats:
            return {
                'total_configurations': 0,
                'total_uses': 0,
                'most_popular_crypto': 'N/A',
                'most_popular_methods': 'N/A',
                'average_uses_per_config': 0
            }
        
        total_configs = len(self.usage_stats)
        total_uses = sum(stats['count'] for stats in self.usage_stats.values())
        
        # Analyze crypto usage
        crypto_usage = {}
        method_usage = {'quantity': {}, 'positioning': {}}
        
        for config_key, stats in self.usage_stats.items():
            config = stats['config']
            
            # Track crypto usage
            crypto = config['crypto_symbol']
            crypto_usage[crypto] = crypto_usage.get(crypto, 0) + stats['count']
            
            # Track method usage
            qty_method = config['quantity_distribution']
            pos_method = config['rung_positioning']
            
            method_usage['quantity'][qty_method] = method_usage['quantity'].get(qty_method, 0) + stats['count']
            method_usage['positioning'][pos_method] = method_usage['positioning'].get(pos_method, 0) + stats['count']
        
        most_popular_crypto = max(crypto_usage.items(), key=lambda x: x[1])[0] if crypto_usage else 'N/A'
        most_popular_qty = max(method_usage['quantity'].items(), key=lambda x: x[1])[0] if method_usage['quantity'] else 'N/A'
        most_popular_pos = max(method_usage['positioning'].items(), key=lambda x: x[1])[0] if method_usage['positioning'] else 'N/A'
        
        return {
            'total_configurations': total_configs,
            'total_uses': total_uses,
            'most_popular_crypto': most_popular_crypto,
            'most_popular_qty_method': most_popular_qty,
            'most_popular_pos_method': most_popular_pos,
            'average_uses_per_config': total_uses / total_configs if total_configs > 0 else 0,
            'crypto_distribution': crypto_usage,
            'method_distribution': method_usage
        }
    
    def start_precalculations(self):
        """Start background precalculations for common parameter combinations"""
        if self.precalc_running:
            return
        
        self.precalc_running = True
        self.precalc_thread = threading.Thread(target=self._precalculate_common_configs, daemon=True)
        self.precalc_thread.start()
        print("Started background precalculations for improved responsiveness...")
    
    def _precalculate_common_configs(self):
        """Precalculate common parameter combinations in background"""
        try:
            # Define common parameter combinations to precalculate
            common_configs = self._get_common_configurations()
            self.precalc_total = len(common_configs)
            self.precalc_progress = 0
            
            print(f"Precalculating {self.precalc_total} common configurations...")
            
            for i, config in enumerate(common_configs):
                try:
                    # Check if paused for user priority
                    while self.precalc_paused:
                        time.sleep(0.1)  # Wait 100ms before checking again
                    
                    # Generate cache key
                    cache_key = self._generate_cache_key(config)
                    
                    # Skip if already cached
                    if cache_key in self.precalc_cache:
                        continue
                    
                    # Calculate ladder configuration
                    result = self.calculator.calculate_ladder_configuration(
                        config['aggression_level'], config['num_rungs'], config['timeframe_hours'],
                        config['budget'], config['quantity_distribution'], 
                        config['crypto_symbol'], config['rung_positioning']
                    )
                    
                    # Store in cache
                    self.precalc_cache[cache_key] = result
                    self.precalc_progress = i + 1
                    
                    # Progress update every 10 calculations
                    if (i + 1) % 10 == 0:
                        progress_pct = (i + 1) / self.precalc_total * 100
                        print(f"Precalculation progress: {i + 1}/{self.precalc_total} ({progress_pct:.1f}%)")
                    
                except Exception as e:
                    print(f"Error precalculating config {i + 1}: {e}")
                    continue
            
            print(f"Precalculation complete! Cached {len(self.precalc_cache)} configurations.")
            
        except Exception as e:
            print(f"Error in precalculation thread: {e}")
        finally:
            self.precalc_running = False
    
    def _get_common_configurations(self):
        """Generate list of configurations to precalculate based on usage patterns"""
        configs = []
        
        # Get most used configurations from usage stats
        most_used = self.get_most_used_configurations(limit=30)
        
        # Add most used configurations first (highest priority)
        for config_key, stats in most_used:
            configs.append(stats['config'])
        
        # If we don't have enough usage data, fall back to default configurations
        if len(configs) < 10:
            print("Insufficient usage data, using default configurations")
            configs.extend(self._get_default_configurations())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_configs = []
        for config in configs:
            config_key = self._generate_cache_key(config)
            if config_key not in seen:
                seen.add(config_key)
                unique_configs.append(config)
        
        print(f"Precalculating {len(unique_configs)} configurations (based on usage patterns)")
        return unique_configs
    
    def _get_default_configurations(self):
        """Get default configurations when usage data is insufficient"""
        configs = []
        
        # Default configurations for new users
        default_configs = [
            # Most common defaults
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'linear_increase', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'equal_notional', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            
            # Common variations
            {'aggression_level': 2, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 4, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 10, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 30, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            
            # Different timeframes
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 168, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 4320, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            
            # Different budgets
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 500, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 5000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'linear'},
            
            # Different positioning methods
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'exponential'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'fibonacci'},
            
            # Bitcoin configurations
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'BTCUSDT', 'rung_positioning': 'linear'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'linear_increase', 'crypto_symbol': 'BTCUSDT', 'rung_positioning': 'linear'},
        ]
        
        return default_configs
    
    def _generate_cache_key(self, config):
        """Generate a unique cache key for a configuration"""
        return f"{config['crypto_symbol']}_{config['aggression_level']}_{config['num_rungs']}_{config['timeframe_hours']}_{config['budget']}_{config['quantity_distribution']}_{config['rung_positioning']}"
    
    def get_precalculated_result(self, aggression_level, num_rungs, timeframe_hours, budget, quantity_distribution, crypto_symbol, rung_positioning):
        """Get precalculated result if available, otherwise return None"""
        config = {
            'aggression_level': aggression_level,
            'num_rungs': num_rungs,
            'timeframe_hours': timeframe_hours,
            'budget': budget,
            'quantity_distribution': quantity_distribution,
            'crypto_symbol': crypto_symbol,
            'rung_positioning': rung_positioning
        }
        
        cache_key = self._generate_cache_key(config)
        return self.precalc_cache.get(cache_key)
    
    def pause_precalculation(self):
        """Pause precalculation for user priority"""
        self.precalc_paused = True
        print("Precalculation paused for user request")
    
    def resume_precalculation(self):
        """Resume precalculation after user request"""
        self.precalc_paused = False
        print("Precalculation resumed")
    
    def get_precalc_status(self):
        """Get precalculation status for display"""
        if not self.precalc_running:
            analytics = self.get_usage_analytics()
            if analytics['total_uses'] > 0:
                return f"✓ {len(self.precalc_cache)} cached | {analytics['total_uses']} uses tracked | Most used: {analytics['most_popular_crypto']}"
            else:
                return f"✓ {len(self.precalc_cache)} configurations cached (learning usage patterns...)"
        else:
            progress_pct = (self.precalc_progress / self.precalc_total * 100) if self.precalc_total > 0 else 0
            status = "Precalculating" if not self.precalc_paused else "Paused for user"
            return f"{status}: {self.precalc_progress}/{self.precalc_total} ({progress_pct:.1f}%)"
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print(f"Starting Interactive Ladder GUI on http://localhost:{port}")
        print("Note: If you see cached data, please hard refresh your browser (Ctrl+F5)")
        self.app.run(debug=debug, port=port, dev_tools_hot_reload=True)

if __name__ == "__main__":
    gui = InteractiveLadderGUI()
    gui.run()
