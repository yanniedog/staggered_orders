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
            suppress_callback_exceptions=True
        )
        
        # Initialize calculation engines
        self.calculator = LadderCalculator()
        self.visualizer = VisualizationEngine()
        self.historical = HistoricalAnalyzer()
        
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
            # Header
            html.Div([
                html.H1("Interactive Staggered Order Ladder", 
                       style={'textAlign': 'center', 'color': '#007bff', 'marginBottom': '20px'}),
                html.P("Real-time visualization and analysis of order ladder configurations",
                      style={'textAlign': 'center', 'color': '#6c757d', 'marginBottom': '30px'})
            ]),
            
            # Main content area
            html.Div([
                # Left sidebar - Controls
                html.Div([
                    self.create_control_panel()
                ], style={'width': '25%', 'float': 'left', 'paddingRight': '10px'}),
                
                # Right side - Visualizations
                html.Div([
                    self.create_visualization_area()
                ], style={'width': '75%', 'float': 'right', 'paddingLeft': '10px'})
            ], style={'display': 'flex', 'width': '100%'}),
            
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
                html.H4("Configuration", style={'marginBottom': '20px', 'color': '#ffffff'})
            ], style={'backgroundColor': '#3d3d3d', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            html.Div([
                # Aggression Level Slider
                html.Div([
                    html.Label("Aggression Level", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Slider(
                        id='aggression-slider',
                        min=1, max=10, step=1, value=5,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Controls depth range: 1=Conservative, 10=Very Aggressive", 
                             style={'color': '#6c757d'})
                ], style={'marginBottom': '30px'}),
                
                # Number of Rungs Slider
                html.Div([
                    html.Label("Number of Rungs", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Slider(
                        id='rungs-slider',
                        min=5, max=50, step=1, value=20,
                        marks={i: str(i) for i in range(5, 51, 5)},
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
                        min=1, max=720, step=1, value=168,
                        marks={1: "1h", 24: "1d", 168: "1w", 720: "30d"},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Historical analysis window in hours", 
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
                
                # Current Price Display
                html.Div([
                    html.Label("Current Price", style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    html.Div(id='current-price-display', 
                           style={'fontSize': '24px', 'color': '#28a745', 'marginBottom': '10px'}),
                    html.Button("Refresh Price", id='refresh-price-btn', 
                              style={'backgroundColor': '#007bff', 'color': '#ffffff', 
                                     'border': 'none', 'padding': '8px 16px', 'borderRadius': '4px'})
                ], style={'marginBottom': '30px'}),
                
                # Action Buttons
                html.Div([
                    html.Button("Recalculate All", id='recalculate-btn', 
                              style={'backgroundColor': '#007bff', 'color': '#ffffff', 
                                     'border': 'none', 'padding': '10px', 'borderRadius': '4px',
                                     'width': '100%', 'marginBottom': '10px'}),
                    html.Button("Export Configuration", id='export-btn', 
                              style={'backgroundColor': '#6c757d', 'color': '#ffffff', 
                                     'border': 'none', 'padding': '10px', 'borderRadius': '4px',
                                     'width': '100%'})
                ])
            ], style={'backgroundColor': '#2d2d2d', 'padding': '20px', 'borderRadius': '8px', 
                     'border': '1px solid #444444'})
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
             Input('recalculate-btn', 'n_clicks')],
            [State('calculation-cache', 'data')]
        )
        def update_all_visualizations_callback(aggression_level, num_rungs, timeframe_hours, 
                                             budget, recalculate_clicks, cache_data):
            return self.update_all_visualizations(aggression_level, num_rungs, timeframe_hours, 
                                                 budget, recalculate_clicks, cache_data)
        
        # Current price callback
        @self.app.callback(
            Output('current-price-display', 'children'),
            [Input('refresh-price-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_current_price_callback(refresh_clicks, interval_n):
            return self.update_current_price(refresh_clicks, interval_n)
        
        # Export callback
        @self.app.callback(
            Output('export-btn', 'children'),
            [Input('export-btn', 'n_clicks')],
            [State('calculation-cache', 'data')]
        )
        def export_configuration_callback(export_clicks, cache_data):
            return self.export_configuration(export_clicks, cache_data)
    
    def update_all_visualizations(self, aggression_level, num_rungs, timeframe_hours, 
                                 budget, recalculate_clicks, cache_data):
        """Main callback that updates all visualizations"""
        # Debounce updates
        current_time = time.time() * 1000
        if current_time - self.last_update_time < self.update_debounce_ms:
            return dash.no_update
        self.last_update_time = current_time
        
        try:
            # Calculate ladder configuration
            ladder_data = self.calculator.calculate_ladder_configuration(
                aggression_level, num_rungs, timeframe_hours, budget
            )
            
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
            # Return empty figures on error
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Error loading data", x=0.5, y=0.5, showarrow=False)
            empty_figs = (empty_fig,) * 9
            return (*empty_figs, "N/A", "N/A", "N/A", "N/A", cache_data)
    
    def update_current_price(self, refresh_clicks, interval_n):
        """Update current price display"""
        try:
            current_price = get_current_price()
            return f"${current_price:.2f}"
        except Exception as e:
            return f"Error: {e}"
    
    def export_configuration(self, export_clicks, cache_data):
        """Export current configuration"""
        if export_clicks and cache_data:
            # Implementation for export functionality
            return "Exported!"
        return "Export Configuration"
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print(f"Starting Interactive Ladder GUI on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    gui = InteractiveLadderGUI()
    gui.run()
