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
        
        self.setup_layout()
        self.setup_callbacks()
    
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
                          style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '30px'})
                ], style={'padding': '20px', 'backgroundColor': '#1a1a1a', 'borderBottom': '2px solid #444444', 'marginBottom': '20px'}),
                
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
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
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
                        options=[
                            {'label': 'Price-Weighted (Current)', 'value': 'price_weighted'},
                            {'label': 'Equal Quantity', 'value': 'equal_quantity'},
                            {'label': 'Equal Notional', 'value': 'equal_notional'},
                            {'label': 'Linear Increase', 'value': 'linear_increase'},
                            {'label': 'Exponential Increase', 'value': 'exponential_increase'},
                            {'label': 'Risk-Parity', 'value': 'risk_parity'},
                            {'label': 'Kelly-Optimized', 'value': 'kelly_optimized'}
                        ],
                        value='price_weighted',
                        style={'width': '100%', 'marginBottom': '10px',
                               'backgroundColor': '#3d3d3d', 'color': '#000000', 'borderRadius': '4px'}
                    ),
                    html.Small("How quantities are distributed across ladder rungs",
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),

                # Rung Positioning Method
                html.Div([
                    html.Label("Rung Positioning Method", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='rung-positioning-dropdown',
                        options=[
                            {'label': 'Quantile-Based (Current)', 'value': 'quantile'},
                            {'label': 'Expected Value Optimization', 'value': 'expected_value'},
                            {'label': 'Linear Spacing', 'value': 'linear'},
                            {'label': 'Exponential Spacing', 'value': 'exponential'},
                            {'label': 'Logarithmic Spacing', 'value': 'logarithmic'},
                            {'label': 'Risk-Weighted', 'value': 'risk_weighted'}
                        ],
                        value='quantile',
                        style={'width': '100%', 'marginBottom': '10px',
                               'backgroundColor': '#3d3d3d', 'color': '#000000', 'borderRadius': '4px'}
                    ),
                    html.Small("How ladder rungs are positioned across price levels",
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),
                
                # Current Price Display
                html.Div([
                    html.Label("Current Price", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    html.Div(id='current-price-display',
                           style={'fontSize': '24px', 'color': '#28a745', 'marginBottom': '10px'})
                ], style={'marginBottom': '30px'}),

                # Cryptocurrency Selection
                html.Div([
                    html.Label("Cryptocurrency", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='crypto-dropdown',
                        options=[
                            {'label': 'Bitcoin (BTC)', 'value': 'BTCUSDT'},
                            {'label': 'Ethereum (ETH)', 'value': 'ETHUSDT'},
                            {'label': 'Solana (SOL)', 'value': 'SOLUSDT'},
                            {'label': 'Cardano (ADA)', 'value': 'ADAUSDT'},
                            {'label': 'Polygon (MATIC)', 'value': 'MATICUSDT'},
                            {'label': 'Chainlink (LINK)', 'value': 'LINKUSDT'},
                            {'label': 'Polkadot (DOT)', 'value': 'DOTUSDT'},
                            {'label': 'Avalanche (AVAX)', 'value': 'AVAXUSDT'},
                            {'label': 'Cosmos (ATOM)', 'value': 'ATOMUSDT'},
                            {'label': 'Algorand (ALGO)', 'value': 'ALGOUSDT'},
                            {'label': 'VeChain (VET)', 'value': 'VETUSDT'},
                            {'label': 'Hedera (HBAR)', 'value': 'HBARUSDT'},
                            {'label': 'Internet Computer (ICP)', 'value': 'ICPUSDT'},
                            {'label': 'Theta (THETA)', 'value': 'THETAUSDT'},
                            {'label': 'Fantom (FTM)', 'value': 'FTMUSDT'},
                            {'label': 'Harmony (ONE)', 'value': 'ONEUSDT'},
                            {'label': 'Near Protocol (NEAR)', 'value': 'NEARUSDT'},
                            {'label': 'Flow (FLOW)', 'value': 'FLOWUSDT'},
                            {'label': 'Helium (HNT)', 'value': 'HNTUSDT'},
                            {'label': 'Arweave (AR)', 'value': 'ARUSDT'},
                            {'label': 'The Graph (GRT)', 'value': 'GRTUSDT'},
                            {'label': '0x (ZRX)', 'value': 'ZRXUSDT'},
                            {'label': 'Basic Attention Token (BAT)', 'value': 'BATUSDT'},
                            {'label': 'Enjin Coin (ENJ)', 'value': 'ENJUSDT'},
                            {'label': 'Decentraland (MANA)', 'value': 'MANAUSDT'},
                            {'label': 'Sandbox (SAND)', 'value': 'SANDUSDT'},
                            {'label': 'ApeCoin (APE)', 'value': 'APEUSDT'},
                            {'label': 'Immutable X (IMX)', 'value': 'IMXUSDT'},
                            {'label': 'Loopring (LRC)', 'value': 'LRCUSDT'},
                            {'label': '1inch (1INCH)', 'value': '1INCHUSDT'},
                            {'label': 'SushiSwap (SUSHI)', 'value': 'SUSHIUSDT'},
                            {'label': 'Uniswap (UNI)', 'value': 'UNIUSDT'},
                            {'label': 'PancakeSwap (CAKE)', 'value': 'CAKEUSDT'},
                            {'label': 'Compound (COMP)', 'value': 'COMPUSDT'},
                            {'label': 'Maker (MKR)', 'value': 'MKRUSDT'},
                            {'label': 'Aave (AAVE)', 'value': 'AAVEUSDT'},
                            {'label': 'Yearn Finance (YFI)', 'value': 'YFIUSDT'},
                            {'label': 'Curve DAO Token (CRV)', 'value': 'CRVUSDT'},
                            {'label': 'Synthetix (SNX)', 'value': 'SNXUSDT'},
                            {'label': 'Balancer (BAL)', 'value': 'BALUSDT'},
                            {'label': 'Ren (REN)', 'value': 'RENUSDT'},
                            {'label': 'Ocean Protocol (OCEAN)', 'value': 'OCEANUSDT'},
                            {'label': 'Storj (STORJ)', 'value': 'STORJUSDT'},
                            {'label': 'Livepeer (LPT)', 'value': 'LPTUSDT'},
                            {'label': 'Ankr (ANKR)', 'value': 'ANKRUSDT'},
                            {'label': 'Fetch.ai (FET)', 'value': 'FETUSDT'},
                            {'label': 'SingularityNET (AGIX)', 'value': 'AGIXUSDT'},
                            {'label': 'OriginTrail (TRAC)', 'value': 'TRACUSDT'},
                            {'label': 'Numeraire (NMR)', 'value': 'NMRUSDT'},
                            {'label': 'PAX Gold (PAXG)', 'value': 'PAXGUSDT'},
                            {'label': 'Tether Gold (XAUT)', 'value': 'XAUTUSDT'}
                        ],
                        value='SOLUSDT',
                        style={'width': '100%', 'marginBottom': '10px',
                               'backgroundColor': '#3d3d3d', 'color': '#000000', 'borderRadius': '4px'}
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
             Output('calculation-cache', 'data')],
            [Input('aggression-slider', 'value'),
             Input('rungs-slider', 'value'),
             Input('timeframe-slider', 'value'),
             Input('budget-input', 'value'),
             Input('quantity-distribution-dropdown', 'value'),
             Input('crypto-dropdown', 'value'),
             Input('rung-positioning-dropdown', 'value'),
             Input('cache-buster', 'data')],
            [State('calculation-cache', 'data')]
        )
        def update_all_visualizations_callback(aggression_level, num_rungs, timeframe_slider,
                                             budget, quantity_distribution, crypto_symbol, rung_positioning,
                                             cache_buster, cache_data):
            # Map slider position to actual hours
            timeframe_map = {0: 24, 1: 168, 2: 720, 3: 4320, 4: 8760, 5: 26280, 6: 43800, 7: 87600}
            timeframe_hours = timeframe_map.get(timeframe_slider, 720)
            
            return self.update_all_visualizations(aggression_level, num_rungs, timeframe_hours,
                                                 budget, quantity_distribution, crypto_symbol, rung_positioning,
                                                 cache_data)
        
        # Current price callback
        @self.app.callback(
            Output('current-price-display', 'children'),
            [Input('interval-component', 'n_intervals'),
             Input('crypto-dropdown', 'value')]
        )
        def update_current_price_callback(interval_n, crypto_symbol):
            return self.update_current_price(interval_n, crypto_symbol)

    
    def update_all_visualizations(self, aggression_level, num_rungs, timeframe_hours,
                                 budget, quantity_distribution, crypto_symbol, rung_positioning,
                                 cache_data):
        """Main callback that updates all visualizations"""
        # Debounce updates
        current_time = time.time() * 1000
        if current_time - self.last_update_time < self.update_debounce_ms:
            return dash.no_update
        self.last_update_time = current_time
        
        try:
            # Validate inputs
            if not all([aggression_level, num_rungs, timeframe_hours, budget]):
                print("Warning: Missing input parameters")
                return dash.no_update
            
            # Calculate ladder configuration
            ladder_data = self.calculator.calculate_ladder_configuration(
                aggression_level, num_rungs, timeframe_hours, budget, quantity_distribution,
                crypto_symbol, rung_positioning
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
            
            # Update cache
            cache_data = {
                'timestamp': current_time,
                'ladder_data': ladder_data,
                'kpis': kpis
            }
            
            return (*figures, *kpis.values(), cache_data)
            
        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()
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
        return (*empty_figs, "N/A", "N/A", "N/A", "N/A", cache_data)
    
    def update_current_price(self, interval_n, crypto_symbol='SOLUSDT'):
        """Update current price display"""
        try:
            current_price = get_current_price(crypto_symbol)
            return f"${current_price:.2f}"
        except Exception as e:
            return f"Error: {e}"

    def run(self, debug=True, port=8050):
        """Run the application"""
        print(f"Starting Interactive Ladder GUI on http://localhost:{port}")
        print("Note: If you see cached data, please hard refresh your browser (Ctrl+F5)")
        self.app.run(debug=debug, port=port, dev_tools_hot_reload=True)

if __name__ == "__main__":
    gui = InteractiveLadderGUI()
    gui.run()
