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
import queue
import os
from functools import lru_cache
import warnings

# Import our custom modules
from gui_calculator import LadderCalculator
from gui_visualizations import VisualizationEngine
from gui_historical import HistoricalAnalyzer
from data_fetcher import get_current_price
from config import load_config
from usage_tracker import UsageTracker
from gui_utils import (
    generate_cache_key, map_timeframe_slider_to_hours, format_timeframe_hours,
    validate_configuration, get_timeframe_labels, get_aggression_labels,
    calculate_debounce_delay, create_error_response, create_loading_response,
    get_chart_priority, get_crypto_explanations, get_timeframe_explanations,
    get_quantity_distribution_explanations, get_rung_positioning_explanations,
    generate_smart_recommendations
)

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
        self.update_debounce_ms = 500  # Increased debounce time for better performance
        self.pending_update = None
        self.update_timer = None
        
        # Precalculation system
        self.precalc_cache = {}
        self.precalc_cache_file = 'precalc_cache.json'
        self.precalc_thread = None
        self.precalc_running = False
        self.precalc_progress = 0
        self.precalc_total = 0
        self.precalc_paused = False  # For user priority

        # Load persistent cache on startup
        self.load_persistent_cache()
        
        # Usage tracking system
        self.usage_tracker = UsageTracker('usage_stats.json')

        # Smart recalculation system
        self.last_settings = {}  # Track last used settings
        self.dependency_map = self._create_dependency_map()

        # Chart loading priority (critical charts first)
        self.chart_priority = {
            'ladder-configuration-chart': 1,  # Most important
            'touch-probability-curves': 1,
            'rung-touch-probabilities': 1,
            'historical-touch-frequency': 2,
            'profit-distribution': 2,
            'risk-return-profile': 2,
            'touch-vs-time': 3,
            'allocation-distribution': 3,
            'fit-quality-dashboard': 3
        }

        self.setup_layout()
        self.setup_callbacks()

        # Start precalculations in background (with error handling)
        try:
            self.start_precalculations()
        except Exception as e:
            print(f"Warning: Could not start precalculations: {e}")
            self.precalc_running = False
    
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

                # Status Indicators (Top Left Corner)
                html.Div([
                    # Precalculation Status with Progress Bar
                    html.Div([
                        html.Div([
                            html.Span("📊", style={'fontSize': '14px', 'marginRight': '8px'}),
                            html.Span(id='precalc-status', style={'fontSize': '12px', 'fontWeight': 'bold'})
                        ], style={'marginBottom': '5px'}),
                        # Progress Bar
                        html.Div([
                            html.Div(id='precalc-progress-bar', style={
                                'width': '0%', 'height': '4px', 'backgroundColor': '#28a745',
                                'borderRadius': '2px', 'transition': 'width 0.3s ease'
                            })
                        ], style={
                            'width': '100%', 'height': '4px', 'backgroundColor': '#444444',
                            'borderRadius': '2px', 'overflow': 'hidden'
                        })
                    ], style={
                        'padding': '10px', 'backgroundColor': '#2d2d2d', 'borderRadius': '8px',
                        'border': '1px solid #444444', 'marginBottom': '10px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.3)'
                    }),
                    
                    # User Request Status
                    html.Div([
                        html.Span("⚡", style={'fontSize': '14px', 'marginRight': '8px'}),
                        html.Span(id='user-request-status', style={'fontSize': '12px', 'fontWeight': 'bold'})
                    ], style={
                        'padding': '8px', 'backgroundColor': '#2d2d2d', 'borderRadius': '6px',
                        'border': '1px solid #444444', 'marginBottom': '10px',
                        'display': 'none'  # Hidden by default, shown during calculations
                    }, id='user-request-status-container')
                ], style={'marginBottom': '10px'}),
                
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
                    
                    # User request status indicator in top right
                    html.Div([
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
                # Recommendations Section
                html.Div([
                    html.Label("💡 Recommended Settings", style={'fontWeight': 'bold', 'color': '#ffc107', 'fontSize': '16px'}),
                    html.Div([
                        html.P([
                            html.Strong("For Optimal Performance:", style={'color': '#28a745'}),
                            html.Br(),
                            "• Quantity: ", html.Span("Kelly Optimized", style={'color': '#00bfff', 'fontWeight': 'bold'}),
                            " - Best risk-adjusted returns",
                            html.Br(),
                            "• Positioning: ", html.Span("Dynamic Density (Default)", style={'color': '#00bfff', 'fontWeight': 'bold'}),
                            " - Adapts to market volatility",
                            html.Br(),
                            "• Timeframe: ", html.Span("1-3 Years", style={'color': '#00bfff', 'fontWeight': 'bold'}),
                            " - Balanced historical data",
                            html.Br(),
                            "• Rungs: ", html.Span("15-25", style={'color': '#00bfff', 'fontWeight': 'bold'}),
                            " - Good granularity without fragmentation"
                        ], style={'color': '#e0e0e0', 'fontSize': '12px', 'lineHeight': '1.8', 'marginBottom': '10px'}),
                        html.P([
                            html.Strong("Conservative:", style={'color': '#6c757d'}), 
                            " Aggression 1-2, Equal Notional, Linear positioning",
                            html.Br(),
                            html.Strong("Aggressive:", style={'color': '#dc3545'}), 
                            " Aggression 4-5, Exponential Increase, Expected Value positioning"
                        ], style={'color': '#b0b0b0', 'fontSize': '11px', 'lineHeight': '1.6', 'fontStyle': 'italic'})
                    ], style={
                        'padding': '12px',
                        'backgroundColor': '#2d2d2d',
                        'borderRadius': '8px',
                        'border': '1px solid #444444',
                        'marginTop': '8px'
                    })
                ], style={'marginBottom': '25px', 'paddingBottom': '20px', 'borderBottom': '2px solid #444444'}),
                
                # Smart Recommendations Section
                html.Div([
                    html.Label("🎯 Smart Recommendations", style={'fontWeight': 'bold', 'color': '#ffc107', 'fontSize': '16px'}),
                    html.Div(id='smart-recommendations', children=[
                        html.P("Adjust settings above to see personalized recommendations", 
                              style={'color': '#6c757d', 'fontStyle': 'italic', 'textAlign': 'center'})
                    ], style={
                        'padding': '12px',
                        'backgroundColor': '#2d2d2d',
                        'borderRadius': '8px',
                        'border': '1px solid #444444',
                        'marginTop': '8px'
                    })
                ], style={'marginBottom': '25px', 'paddingBottom': '20px', 'borderBottom': '2px solid #444444'}),
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
                    html.Label([
                        "Analysis Timeframe ",
                        html.Span("ℹ️", style={'fontSize': '12px', 'marginLeft': '5px', 'cursor': 'help'})
                    ], style={'fontWeight': 'bold', 'color': '#ffffff'}),
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
                             style={'color': '#6c757d'}),
                    html.Div(id='timeframe-explanation', 
                            style={
                                'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 
                                'fontStyle': 'italic', 'backgroundColor': '#2d2d2d', 'padding': '8px',
                                'borderRadius': '4px', 'border': '1px solid #444444'
                            })
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
                    html.Label([
                        "Quantity Distribution Method ",
                        html.Span("ℹ️", style={'fontSize': '12px', 'marginLeft': '5px', 'cursor': 'help'})
                    ], style={'fontWeight': 'bold', 'color': '#ffffff'}),
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
                            style={
                                'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 
                                'fontStyle': 'italic', 'backgroundColor': '#2d2d2d', 'padding': '8px',
                                'borderRadius': '4px', 'border': '1px solid #444444'
                            })
                ], style={'marginBottom': '30px'}),
                
                # Rung Positioning Method
                html.Div([
                    html.Label([
                        "Rung Positioning Method ",
                        html.Span("ℹ️", style={'fontSize': '12px', 'marginLeft': '5px', 'cursor': 'help'})
                    ], style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='rung-positioning-dropdown',
                        options=self.get_sorted_positioning_options(),
                        value='dynamic_density',
                        style={'width': '100%', 'marginBottom': '10px', 'borderRadius': '4px'},
                        className='dark-dropdown'
                    ),
                    html.Small("How ladder rungs are positioned across price levels",
                             style={'color': '#6c757d'}),
                    html.Div(id='rung-positioning-explanation', 
                            style={
                                'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 
                                'fontStyle': 'italic', 'backgroundColor': '#2d2d2d', 'padding': '8px',
                                'borderRadius': '4px', 'border': '1px solid #444444'
                            })
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
                    html.Label([
                        "Cryptocurrency ",
                        html.Span("ℹ️", style={'fontSize': '12px', 'marginLeft': '5px', 'cursor': 'help'})
                    ], style={'fontWeight': 'bold', 'color': '#ffffff'}),
                    dcc.Dropdown(
                        id='crypto-dropdown',
                        className='dark-dropdown',
                        options=self.get_sorted_crypto_options(),
                        value='SOLUSDT',
                        style={'width': '100%', 'marginBottom': '10px', 'borderRadius': '4px'}
                    ),
                    html.Small("Select cryptocurrency for ladder analysis",
                             style={'color': '#6c757d'}),
                    html.Div(id='crypto-explanation', 
                            style={
                                'color': '#ffffff', 'marginTop': '8px', 'fontSize': '13px', 
                                'fontStyle': 'italic', 'backgroundColor': '#2d2d2d', 'padding': '8px',
                                'borderRadius': '4px', 'border': '1px solid #444444'
                            })
                ], style={'marginBottom': '30px'})
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
            timeframe_hours = map_timeframe_slider_to_hours(timeframe_slider)

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
            explanations = get_quantity_distribution_explanations()
            return explanations.get(method, "Method description not available.")
        
        # Rung Positioning Method Explanation Callback
        @self.app.callback(
            Output('rung-positioning-explanation', 'children'),
            [Input('rung-positioning-dropdown', 'value')]
        )
        def update_rung_positioning_explanation(method):
            explanations = get_rung_positioning_explanations()
            return explanations.get(method, "Method description not available.")
        
        # Cryptocurrency Explanation Callback
        @self.app.callback(
            Output('crypto-explanation', 'children'),
            [Input('crypto-dropdown', 'value')]
        )
        def update_crypto_explanation(crypto_symbol):
            explanations = get_crypto_explanations()
            return explanations.get(crypto_symbol, "Cryptocurrency description not available.")
        
        # Timeframe Explanation Callback
        @self.app.callback(
            Output('timeframe-explanation', 'children'),
            [Input('timeframe-slider', 'value')]
        )
        def update_timeframe_explanation(timeframe_value):
            explanations = get_timeframe_explanations()
            return explanations.get(timeframe_value, "Timeframe description not available.")
        
        @self.app.callback(
            [Output('precalc-status', 'children'),
             Output('precalc-progress-bar', 'style')],
            [Input('precalc-interval', 'n_intervals')]
        )
        def update_precalc_status(n_intervals):
            try:
                status_text = self.get_precalc_status()
                
                # Calculate progress bar width
                if self.precalc_running and self.precalc_total > 0:
                    progress_pct = (self.precalc_progress / self.precalc_total) * 100
                    progress_style = {
                        'width': f'{progress_pct:.1f}%', 'height': '4px', 'backgroundColor': '#28a745',
                        'borderRadius': '2px', 'transition': 'width 0.3s ease'
                    }
                else:
                    progress_style = {
                        'width': '0%', 'height': '4px', 'backgroundColor': '#28a745',
                        'borderRadius': '2px', 'transition': 'width 0.3s ease'
                    }
                
                return status_text, progress_style
            except Exception as e:
                print(f"Error in precalc status callback: {e}")
                return "[OK] Status unavailable", {'width': '0%', 'height': '4px', 'backgroundColor': '#28a745', 'borderRadius': '2px', 'transition': 'width 0.3s ease'}

        # User Request Status Callback
        @self.app.callback(
            [Output('user-request-status', 'children'),
             Output('user-request-status-container', 'style')],
            [Input('aggression-slider', 'value'),
             Input('rungs-slider', 'value'),
             Input('timeframe-slider', 'value'),
             Input('budget-input', 'value'),
             Input('quantity-distribution-dropdown', 'value'),
             Input('crypto-dropdown', 'value'),
             Input('rung-positioning-dropdown', 'value')]
        )
        def update_user_request_status(*inputs):
            try:
                # Show processing status when any parameter changes
                status_text = "🔄 Calculating..."
                container_style = {
                    'padding': '8px', 'backgroundColor': '#2d2d2d', 'borderRadius': '6px',
                    'border': '1px solid #444444', 'marginBottom': '10px',
                    'display': 'block', 'animation': 'pulse 1s infinite'
                }
                
                return status_text, container_style
                
            except Exception as e:
                print(f"Error in user request status callback: {e}")
                return "[OK] Ready", {'display': 'none'}

        # Hide user request status when calculations complete
        @self.app.callback(
            Output('user-request-status-container', 'style', allow_duplicate=True),
            [Input('calculation-cache', 'data')],
            prevent_initial_call=True
        )
        def hide_user_request_status(cache_data):
            """Hide user request status when calculations complete"""
            return {'display': 'none'}

        # Smart Recommendations Callback
        @self.app.callback(
            Output('smart-recommendations', 'children'),
            [Input('aggression-slider', 'value'),
             Input('rungs-slider', 'value'),
             Input('timeframe-slider', 'value'),
             Input('budget-input', 'value'),
             Input('quantity-distribution-dropdown', 'value'),
             Input('crypto-dropdown', 'value'),
             Input('rung-positioning-dropdown', 'value')]
        )
        def update_smart_recommendations(aggression, rungs, timeframe, budget, quantity_dist, crypto, positioning):
            """Generate smart recommendations based on current settings"""
            try:
                recommendations = generate_smart_recommendations(
                    aggression, rungs, timeframe, budget, quantity_dist, crypto, positioning
                )
                
                # Generate recommendation cards
                if not recommendations:
                    return html.P("✅ Your current settings look optimal! No specific recommendations at this time.", 
                                 style={'color': '#28a745', 'textAlign': 'center', 'fontWeight': 'bold'})
                
                cards = []
                for i, rec in enumerate(recommendations[:3]):  # Limit to 3 recommendations
                    color_map = {
                        'success': '#28a745',
                        'warning': '#ffc107', 
                        'info': '#17a2b8',
                        'error': '#dc3545'
                    }
                    color = color_map.get(rec['type'], '#6c757d')
                    
                    card = html.Div([
                        html.Div([
                            html.Strong(rec['title'], style={'color': color, 'fontSize': '14px'}),
                            html.Br(),
                            html.Span(rec['message'], style={'color': '#ffffff', 'fontSize': '12px'}),
                            html.Br(),
                            html.Span(f"💡 {rec['suggestion']}", style={'color': color, 'fontSize': '11px', 'fontStyle': 'italic'})
                        ], style={
                            'padding': '8px',
                            'backgroundColor': '#1a1a1a',
                            'borderRadius': '4px',
                            'border': f'1px solid {color}',
                            'marginBottom': '8px'
                        })
                    ])
                    cards.append(card)
                
                return cards
                
            except Exception as e:
                print(f"Error generating smart recommendations: {e}")
                return html.P("Unable to generate recommendations at this time.", 
                           style={'color': '#6c757d', 'textAlign': 'center'})

    
    def get_sorted_quantity_options(self):
        """Get quantity distribution options sorted by usage frequency"""
        analytics = self.usage_tracker.get_analytics()
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
        analytics = self.usage_tracker.get_analytics()
        method_usage = analytics.get('method_distribution', {}).get('positioning', {})
        
        # Default options with usage counts
        options = [
            {'label': 'Dynamic Density (Recommended)', 'value': 'dynamic_density'},
            {'label': 'Linear Spacing', 'value': 'linear'},
            {'label': 'Support/Resistance Clustering', 'value': 'support_resistance'},
            {'label': 'Volume Profile Weighted', 'value': 'volume_profile'},
            {'label': 'Touch Pattern Analysis', 'value': 'touch_pattern'},
            {'label': 'Adaptive Probability', 'value': 'adaptive_probability'},
            {'label': 'Expected Value Optimization', 'value': 'expected_value'},
            {'label': 'Quantile-Based', 'value': 'quantile'},
            {'label': 'Risk-Weighted', 'value': 'risk_weighted'},
            {'label': 'Exponential Spacing', 'value': 'exponential'},
            {'label': 'Logarithmic Spacing', 'value': 'logarithmic'},
            {'label': 'Fibonacci Levels', 'value': 'fibonacci'}
        ]
        
        # Sort by usage frequency (descending)
        def sort_key(option):
            usage_count = method_usage.get(option['value'], 0)
            # Put recommended first, then by usage
            if option['value'] == 'dynamic_density':
                return (1, -usage_count)  # Recommended first
            return (0, -usage_count)  # Then by usage
        
        return sorted(options, key=sort_key)
    
    def get_sorted_crypto_options(self):
        """Get cryptocurrency options sorted by usage frequency"""
        analytics = self.usage_tracker.get_analytics()
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

    def _create_dependency_map(self):
        """Create mapping of which settings affect which calculations"""
        return {
            'aggression_level': {
                'affects': ['depth_calculations', 'ladder_configuration', 'touch_probabilities', 'all_charts'],
                'cost': 'high'  # Expensive recalculation
            },
            'num_rungs': {
                'affects': ['ladder_configuration', 'touch_probabilities', 'all_charts'],
                'cost': 'high'
            },
            'timeframe_hours': {
                'affects': ['historical_data', 'weibull_params', 'touch_probabilities', 'historical_charts'],
                'cost': 'medium'
            },
            'budget': {
                'affects': ['quantity_distribution', 'allocations', 'kpis', 'order_tables'],
                'cost': 'low'
            },
            'quantity_distribution': {
                'affects': ['quantity_distribution', 'allocations', 'allocation_charts', 'kpis'],
                'cost': 'medium'
            },
            'crypto_symbol': {
                'affects': ['current_price', 'ladder_configuration', 'all_charts'],
                'cost': 'high'
            },
            'rung_positioning': {
                'affects': ['ladder_positions', 'touch_probabilities', 'configuration_charts'],
                'cost': 'medium'
            },
            'trading_fee': {
                'affects': ['kpis', 'order_calculations', 'order_tables'],
                'cost': 'low'
            },
            'min_notional': {
                'affects': ['order_filtering', 'order_tables'],
                'cost': 'low'
            }
        }

    def detect_setting_changes(self, new_settings):
        """Detect which settings have changed since last calculation"""
        if not self.last_settings:
            # First run - everything has "changed"
            self.last_settings = new_settings.copy()
            return list(new_settings.keys())

        changed_settings = []
        for setting, new_value in new_settings.items():
            if setting not in self.last_settings or self.last_settings[setting] != new_value:
                changed_settings.append(setting)

        # Update last settings
        self.last_settings = new_settings.copy()
        return changed_settings

    def get_affected_calculations(self, changed_settings):
        """Get all calculations affected by the changed settings"""
        affected = set()

        for setting in changed_settings:
            if setting in self.dependency_map:
                affected.update(self.dependency_map[setting]['affects'])

        return list(affected)

    def get_cached_component(self, component_type, params):
        """Get cached component if available"""
        cache_key = self._generate_component_cache_key(component_type, params)
        return self.component_cache.get(cache_key)

    def cache_component(self, component_type, params, result):
        """Cache a component for future use"""
        cache_key = self._generate_component_cache_key(component_type, params)
        self.component_cache[cache_key] = {
            'result': result,
            'timestamp': time.time() * 1000
        }

        # Cleanup old entries periodically
        if len(self.component_cache) > 100:
            self._cleanup_component_cache()

    def _generate_component_cache_key(self, component_type, params):
        """Generate cache key for component-level caching"""
        # Include relevant parameters based on component type
        if component_type == 'kpis':
            key_params = {
                'budget': params.get('budget'),
                'quantity_distribution': params.get('quantity_distribution'),
                'crypto_symbol': params.get('crypto_symbol'),
                'timeframe_hours': params.get('timeframe_hours')
            }
        elif component_type == 'order_tables':
            key_params = {
                'trading_fee': params.get('trading_fee'),
                'min_notional': params.get('min_notional'),
                'crypto_symbol': params.get('crypto_symbol')
            }
        else:
            key_params = params

        return f"{component_type}_{self.usage_tracker.generate_cache_key(key_params)}"

    def _cleanup_component_cache(self):
        """Remove old entries from component cache"""
        current_time = time.time() * 1000
        cutoff_time = current_time - (10 * 60 * 1000)  # 10 minutes for components

        keys_to_remove = []
        for key, cached_data in self.component_cache.items():
            if cached_data['timestamp'] < cutoff_time:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.component_cache[key]

        if keys_to_remove:
            print(f"Cleaned up {len(keys_to_remove)} old component cache entries")

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

                # Save to persistent cache for future use
                cache_key = self._generate_cache_key({
                    'aggression_level': aggression_level,
                    'num_rungs': num_rungs,
                    'timeframe_hours': timeframe_hours,
                    'budget': budget,
                    'quantity_distribution': quantity_distribution,
                    'crypto_symbol': crypto_symbol,
                    'rung_positioning': rung_positioning
                })
                self.precalc_cache[cache_key] = ladder_data
                self.save_persistent_cache()
            else:
                print(f"Using precalculated result: {crypto_symbol} {aggression_level} {num_rungs} {timeframe_hours}h {budget} {quantity_distribution} {rung_positioning}")
                # ladder_data is already set from precalculated result
            
            # Track usage of this configuration
            self.usage_tracker.track_usage(
                aggression_level, num_rungs, timeframe_hours, budget,
                quantity_distribution, crypto_symbol, rung_positioning
            )
            
            # Validate ladder data
            if not ladder_data or not isinstance(ladder_data, dict) or 'buy_depths' not in ladder_data:
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

    def update_all_visualizations(self, aggression_level, num_rungs, timeframe_hours,
                                budget, quantity_distribution, crypto_symbol, rung_positioning,
                                trading_fee, min_notional, cache_data):
        """Synchronous version of update_all_visualizations with caching"""
        try:
            # Pause precalculation for user priority
            self.pause_precalculation()

            # Simple debouncing
            current_time = time.time() * 1000
            if current_time - self.last_update_time < self.update_debounce_ms:
                self.resume_precalculation()
                return dash.no_update
            
            self.last_update_time = current_time

            # Validate inputs
            if not all([aggression_level, num_rungs, timeframe_hours, budget]):
                print("Warning: Missing input parameters")
                self.resume_precalculation()
                return dash.no_update

            # Try to get precalculated result first (fast path)
            ladder_data = self.get_precalculated_result(
                aggression_level, num_rungs, timeframe_hours, budget,
                quantity_distribution, crypto_symbol, rung_positioning
            )

            # If not precalculated, calculate now
            if ladder_data is None:
                print(f"Calculating: {crypto_symbol} {aggression_level} {num_rungs} {timeframe_hours}h {budget} {quantity_distribution} {rung_positioning}")
                ladder_data = self.calculator.calculate_ladder_configuration(
                    aggression_level, num_rungs, timeframe_hours, budget, quantity_distribution,
                    crypto_symbol, rung_positioning
                )

                # Save to persistent cache for future use
                cache_key = self._generate_cache_key({
                    'aggression_level': aggression_level,
                    'num_rungs': num_rungs,
                    'timeframe_hours': timeframe_hours,
                    'budget': budget,
                    'quantity_distribution': quantity_distribution,
                    'crypto_symbol': crypto_symbol,
                    'rung_positioning': rung_positioning
                })
                self.precalc_cache[cache_key] = ladder_data
                self.save_persistent_cache()
            else:
                print(f"Using precalculated result: {crypto_symbol} {aggression_level} {num_rungs} {timeframe_hours}h {budget} {quantity_distribution} {rung_positioning}")
            
            # Track usage of this configuration
            self.usage_tracker.track_usage(
                aggression_level, num_rungs, timeframe_hours, budget,
                quantity_distribution, crypto_symbol, rung_positioning
            )
            
            # Validate ladder data
            if not ladder_data or not isinstance(ladder_data, dict) or 'buy_depths' not in ladder_data:
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

            # Return all results
            return (*figures, *kpis.values(), buy_table, sell_table, cache_data)

        except Exception as e:
            print(f"Error in visualization update: {e}")
            import traceback
            traceback.print_exc()
            # Resume precalculation even on error
            self.resume_precalculation()
            return self._get_error_response(cache_data)



    def cache_charts(self, aggression_level, num_rungs, timeframe_hours, budget,
                    quantity_distribution, crypto_symbol, rung_positioning, figures, kpis):
        """Cache individual charts for future use"""
        cache_key = self._generate_chart_cache_key({
            'aggression_level': aggression_level,
            'num_rungs': num_rungs,
            'timeframe_hours': timeframe_hours,
            'budget': budget,
            'quantity_distribution': quantity_distribution,
            'crypto_symbol': crypto_symbol,
            'rung_positioning': rung_positioning
        })

        self.chart_cache[cache_key] = {
            'figures': figures,
            'kpis': kpis,
            'timestamp': time.time() * 1000
        }

        # Limit cache size to prevent memory issues
        if len(self.chart_cache) > 50:
            self._cleanup_chart_cache()

    def _generate_chart_cache_key(self, config):
        """Generate a unique cache key for chart configurations"""
        # Use a subset of parameters for chart caching (some parameters don't affect charts)
        key_params = {
            'aggression_level': config['aggression_level'],
            'num_rungs': config['num_rungs'],
            'timeframe_hours': config['timeframe_hours'],
            'quantity_distribution': config['quantity_distribution'],
            'crypto_symbol': config['crypto_symbol'],
            'rung_positioning': config['rung_positioning']
        }
        return self.usage_tracker.generate_cache_key(key_params)

    def _cleanup_chart_cache(self):
        """Remove old entries from chart cache"""
        current_time = time.time() * 1000
        cutoff_time = current_time - (24 * 60 * 60 * 1000)  # 24 hours

        keys_to_remove = []
        for key, cached_data in self.chart_cache.items():
            if cached_data['timestamp'] < cutoff_time:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.chart_cache[key]

    def _can_use_cached_results(self, cache_key):
        """Check if cached results can be used for the current configuration"""
        if cache_key in self.chart_cache:
            cached_data = self.chart_cache[cache_key]
            # Check if cache is recent (within 1 hour)
            current_time = time.time() * 1000
            cache_age = current_time - cached_data['timestamp']
            return cache_age < (60 * 60 * 1000)  # 1 hour
        return False

    def _get_cached_charts(self, cache_key):
        """Get cached charts if available"""
        if cache_key in self.chart_cache:
            cached_data = self.chart_cache[cache_key]
            return cached_data['figures'], cached_data['kpis']
        return None, None

    def _cache_charts(self, cache_key, figures, kpis):
        """Cache charts with timestamp"""
        self.chart_cache[cache_key] = {
            'figures': figures,
            'kpis': kpis,
            'timestamp': time.time() * 1000
        }
        
        # Limit cache size to prevent memory issues
        if len(self.chart_cache) > 50:
            # Remove oldest entries
            sorted_items = sorted(self.chart_cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_items[:10]:  # Remove 10 oldest entries
                del self.chart_cache[key]

    def _generate_charts_lazy(self, ladder_data, timeframe_hours, trading_fee, min_notional,
                              aggression_level, num_rungs, budget, quantity_distribution, 
                              crypto_symbol, rung_positioning):
        """Generate charts with lazy loading and smart caching"""
        try:
            # Generate cache key for this configuration
            cache_key = self._generate_chart_cache_key({
                'aggression_level': aggression_level,
                'num_rungs': num_rungs,
                'timeframe_hours': timeframe_hours,
                'budget': budget,
                'quantity_distribution': quantity_distribution,
                'crypto_symbol': crypto_symbol,
                'rung_positioning': rung_positioning
            })

            # Check if we can use cached results
            if self._can_use_cached_results(cache_key):
                print("Using cached charts")
                cached_figures, cached_kpis = self._get_cached_charts(cache_key)
                if cached_figures and cached_kpis:
                    # Create order tables (these are always fresh)
                    buy_table = self._create_buy_orders_table(ladder_data, trading_fee, min_notional)
                    sell_table = self._create_sell_orders_table(ladder_data, trading_fee, min_notional)
                    
                    # Update cache
                    cache_data = {
                        'timestamp': time.time() * 1000,
                        'ladder_data': ladder_data,
                        'kpis': cached_kpis,
                        'trading_fee': trading_fee,
                        'min_notional': min_notional,
                        'calculation_id': self.current_calculation_id
                    }
                    
                    return (*cached_figures, *cached_kpis.values(), buy_table, sell_table, cache_data)

            # Generate charts with priority-based loading
            print("Generating charts with lazy loading...")
            
            # Phase 1: Critical charts (load immediately)
            critical_charts = [
                'ladder-configuration-chart',
                'touch-probability-curves',
                'rung-touch-probabilities'
            ]
            
            figures = {}
            for chart_name in critical_charts:
                try:
                    fig = self._generate_single_chart(chart_name, ladder_data, timeframe_hours)
                    figures[chart_name] = fig
                except Exception as e:
                    print(f"Error generating {chart_name}: {e}")
                    figures[chart_name] = self._create_empty_chart(chart_name, "Error loading data")

            # Phase 2: Secondary charts (load after critical)
            secondary_charts = [
                'historical-touch-frequency',
                'profit-distribution',
                'risk-return-profile'
            ]
            
            for chart_name in secondary_charts:
                try:
                    fig = self._generate_single_chart(chart_name, ladder_data, timeframe_hours)
                    figures[chart_name] = fig
                except Exception as e:
                    print(f"Error generating {chart_name}: {e}")
                    figures[chart_name] = self._create_empty_chart(chart_name, "Error loading data")

            # Phase 3: Tertiary charts (load last)
            tertiary_charts = [
                'touch-vs-time',
                'allocation-distribution',
                'fit-quality-dashboard'
            ]
            
            for chart_name in tertiary_charts:
                try:
                    fig = self._generate_single_chart(chart_name, ladder_data, timeframe_hours)
                    figures[chart_name] = fig
                except Exception as e:
                    print(f"Error generating {chart_name}: {e}")
                    figures[chart_name] = self._create_empty_chart(chart_name, "Error loading data")

            # Calculate KPIs
            kpis = self.calculator.calculate_kpis(ladder_data)
            
            # Create order tables
            buy_table = self._create_buy_orders_table(ladder_data, trading_fee, min_notional)
            sell_table = self._create_sell_orders_table(ladder_data, trading_fee, min_notional)

            # Convert figures dict to ordered tuple for return
            ordered_figures = []
            chart_order = [
                'ladder-configuration-chart',
                'touch-probability-curves',
                'rung-touch-probabilities',
                'historical-touch-frequency',
                'profit-distribution',
                'risk-return-profile',
                'touch-vs-time',
                'allocation-distribution',
                'fit-quality-dashboard'
            ]

            for chart_name in chart_order:
                ordered_figures.append(figures.get(chart_name, self._create_empty_chart(chart_name, "Error loading data")))

            # Cache the results
            self._cache_charts(cache_key, tuple(ordered_figures), kpis)

            # Update cache
            cache_data = {
                'timestamp': time.time() * 1000,
                'ladder_data': ladder_data,
                'kpis': kpis,
                'trading_fee': trading_fee,
                'min_notional': min_notional,
                'calculation_id': self.current_calculation_id
            }

            return (*ordered_figures, *kpis.values(), buy_table, sell_table, cache_data)

        except Exception as e:
            print(f"Error in lazy chart generation: {e}")
            return self._get_error_response(cache_data)

        print(f"Cleaned up {len(keys_to_remove)} old chart cache entries")

    def _return_cached_charts(self, cached_data, trading_fee, min_notional, cache_data):
        """Return cached charts with updated tables and KPIs"""
        try:
            figures = cached_data['figures']
            kpis = cached_data['kpis']

            # For tables, we need fresh calculation since they depend on trading fees and min notional
            # We'll use placeholder for now and let the system recalculate if needed
            buy_table = html.Div("Loading tables...", style={'color': '#ffc107'})
            sell_table = html.Div("Loading tables...", style={'color': '#ffc107'})

            return (*figures, *kpis.values(), buy_table, sell_table, cache_data)

        except Exception as e:
            print(f"Error returning cached charts: {e}")
            return self._get_error_response(cache_data)

    def _get_cached_kpis(self, ladder_data):
        """Get KPIs from cache or return default values"""
        try:
            # For now, return default values - could be enhanced to cache KPIs separately
            return {
                'total_profit': "Cached",
                'monthly_fills': "Cached",
                'capital_efficiency': "Cached",
                'timeframe': "Cached"
            }
        except Exception as e:
            print(f"Error getting cached KPIs: {e}")
            return {
                'total_profit': "N/A",
                'monthly_fills': "N/A",
                'capital_efficiency': "N/A",
                'timeframe': "N/A"
            }

    def _get_cached_kpis(self, ladder_data):
        """Get KPIs from cache or return default values"""
        try:
            # For now, return default values - could be enhanced to cache KPIs separately
            return {
                'total_profit': "Cached",
                'monthly_fills': "Cached",
                'capital_efficiency': "Cached",
                'timeframe': "Cached"
            }
        except Exception as e:
            print(f"Error getting cached KPIs: {e}")
            return {
                'total_profit': "N/A",
                'monthly_fills': "N/A",
                'capital_efficiency': "N/A",
                'timeframe': "N/A"
            }

    def _get_cached_kpis(self, ladder_data):
        """Get KPIs from cache or return default values"""
        try:
            # For now, return default values - could be enhanced to cache KPIs separately
            return {
                'total_profit': "Cached",
                'monthly_fills': "Cached",
                'capital_efficiency': "Cached",
                'timeframe': "Cached"
            }
        except Exception as e:
            print(f"Error getting cached KPIs: {e}")
            return {
                'total_profit': "N/A",
                'monthly_fills': "N/A",
                'capital_efficiency': "N/A",
                'timeframe': "N/A"
            }

    def _get_cached_kpis(self, ladder_data):
        """Get KPIs from cache or return default values"""
        try:
            # For now, return default values - could be enhanced to cache KPIs separately
            return {
                'total_profit': "Cached",
                'monthly_fills': "Cached",
                'capital_efficiency': "Cached",
                'timeframe': "Cached"
            }
        except Exception as e:
            print(f"Error getting cached KPIs: {e}")
            return {
                'total_profit': "N/A",
                'monthly_fills': "N/A",
                'capital_efficiency': "N/A",
                'timeframe': "N/A"
            }

        loading_figs = (loading_fig,) * 9
        loading_table = html.Div([
            html.Span("⚙️", style={'marginRight': '8px'}),
            "Processing calculations..."
        ], style={'color': '#ffc107', 'fontSize': '14px'})

        # Add loading state to cache for progress tracking
        cache_data = cache_data.copy() if cache_data else {}
        cache_data['loading_state'] = {
            'status': 'loading',
            'timestamp': time.time() * 1000,
            'stage': 'initializing'
        }

        return (*loading_figs, "Loading...", "Loading...", "Loading...", "Loading...",
                loading_table, loading_table, cache_data)

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
    
    def start_precalculations(self):
        """Start background precalculations for common parameter combinations"""
        if self.precalc_running:
            return

        self.precalc_running = True
        try:
            self.precalc_thread = threading.Thread(target=self._precalculate_common_configs, daemon=True)
            self.precalc_thread.start()
            print("Started background precalculations for improved responsiveness...")
        except Exception as e:
            print(f"Error starting precalculation thread: {e}")
            self.precalc_running = False
    
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
                        self.precalc_progress = i + 1
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

                    # Save to persistent cache periodically
                    if (i + 1) % 5 == 0:
                        self.save_persistent_cache()

                    # Progress update every 10 calculations
                    if (i + 1) % 10 == 0:
                        progress_pct = (i + 1) / self.precalc_total * 100
                        print(f"Precalculation progress: {i + 1}/{self.precalc_total} ({progress_pct:.1f}%)")

                except Exception as e:
                    print(f"Error precalculating config {i + 1}: {e}")
                    self.precalc_progress = i + 1
                    continue

            print(f"Precalculation complete! Cached {len(self.precalc_cache)} configurations.")

            # Save final persistent cache
            self.save_persistent_cache()

        except Exception as e:
            print(f"Error in precalculation thread: {e}")
        finally:
            self.precalc_running = False
    
    def _get_common_configurations(self):
        """Generate list of configurations to precalculate based on usage patterns"""
        configs = []
        
        # Get most used configurations from usage stats
        most_used = self.usage_tracker.get_most_used_configurations(limit=30)
        
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
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'linear_increase', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'equal_notional', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            
            # Common variations
            {'aggression_level': 2, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 4, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 10, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 30, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            
            # Different timeframes
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 168, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 4320, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            
            # Different budgets
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 500, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 5000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'dynamic_density'},
            
            # Different positioning methods
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'exponential'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'SOLUSDT', 'rung_positioning': 'fibonacci'},
            
            # Bitcoin configurations
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'kelly_optimized', 'crypto_symbol': 'BTCUSDT', 'rung_positioning': 'dynamic_density'},
            {'aggression_level': 3, 'num_rungs': 20, 'timeframe_hours': 720, 'budget': 1000, 'quantity_distribution': 'linear_increase', 'crypto_symbol': 'BTCUSDT', 'rung_positioning': 'dynamic_density'},
        ]
        
        return default_configs
    
    def _generate_cache_key(self, config):
        """Generate a unique cache key for a configuration"""
        return generate_cache_key(config)
    
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

        # Try exact match first
        result = self.precalc_cache.get(cache_key)
        if result is not None:
            print(f"Using cached result (exact match): {cache_key}")
            return result

        # Try partial matching for better cache utilization
        result = self._find_closest_cache_match(config)
        if result is not None:
            print(f"Using cached result (closest match): {cache_key}")
            # Cache the result under the new key for future use
            self.precalc_cache[cache_key] = result
            return result

        return None

    def _find_closest_cache_match(self, config):
        """Find the closest cached configuration that can be reused"""
        target_key = self._generate_cache_key(config)

        # Look for configurations that differ only in minor parameters
        for cached_key, cached_result in self.precalc_cache.items():
            if self._configs_are_similar(config, self._parse_cache_key(cached_key)):
                print(f"Found similar configuration: {cached_key} -> {target_key}")
                return cached_result

        return None

    def _configs_are_similar(self, config1, config2):
        """Check if two configurations are similar enough to reuse results"""
        # Same crypto, timeframe, and positioning method are most important
        if (config1['crypto_symbol'] != config2['crypto_symbol'] or
            config1['timeframe_hours'] != config2['timeframe_hours'] or
            config1['rung_positioning'] != config2['rung_positioning']):
            return False

        # Allow some flexibility in other parameters
        # Aggression level can vary by 1, rungs by 5, budget by 20%
        aggression_diff = abs(config1['aggression_level'] - config2['aggression_level'])
        rungs_diff = abs(config1['num_rungs'] - config2['num_rungs'])
        budget_ratio = max(config1['budget'], config2['budget']) / min(config1['budget'], config2['budget'])

        return (aggression_diff <= 1 and rungs_diff <= 5 and budget_ratio <= 1.2)

    def _parse_cache_key(self, cache_key):
        """Parse a cache key back into configuration dict"""
        parts = cache_key.split('_')
        if len(parts) != 7:
            return None

        return {
            'crypto_symbol': parts[0],
            'aggression_level': int(parts[1]),
            'num_rungs': int(parts[2]),
            'timeframe_hours': int(parts[3]),
            'budget': float(parts[4]),
            'quantity_distribution': parts[5],
            'rung_positioning': parts[6]
        }

    def load_persistent_cache(self):
        """Load persistent cache from disk"""
        try:
            import json
            if os.path.exists(self.precalc_cache_file):
                try:
                    with open(self.precalc_cache_file, 'r') as f:
                        cache_data = json.load(f)
                        self.precalc_cache = cache_data
                        print(f"Loaded persistent cache: {len(self.precalc_cache)} configurations")
                    return True
                except json.JSONDecodeError as e:
                    print(f"Corrupted persistent cache file detected: {e}")
                    print("Removing corrupted cache file and starting fresh")
                    os.remove(self.precalc_cache_file)
                    self.precalc_cache = {}
                    return False
                except Exception as e:
                    print(f"Error reading persistent cache file: {e}")
                    # Try to backup the corrupted file
                    backup_file = self.precalc_cache_file + '.corrupted'
                    try:
                        os.rename(self.precalc_cache_file, backup_file)
                        print(f"Backed up corrupted cache to {backup_file}")
                    except:
                        pass
                    self.precalc_cache = {}
                    return False
            else:
                print("No persistent cache file found, starting fresh")
                return False
        except Exception as e:
            print(f"Error loading persistent cache: {e}")
            self.precalc_cache = {}
            return False

    def save_persistent_cache(self):
        """Save cache to persistent storage"""
        try:
            import json

            # Only save if we have significant cache
            if len(self.precalc_cache) >= 5:
                # Write to temporary file first, then rename (atomic operation)
                temp_file = self.precalc_cache_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(self.precalc_cache, f, indent=2)
                # Atomic rename
                import os
                os.replace(temp_file, self.precalc_cache_file)
                print(f"Saved persistent cache: {len(self.precalc_cache)} configurations")
                return True
            return False
        except Exception as e:
            print(f"Error saving persistent cache: {e}")
            return False
    
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
        try:
            # Check if precalc thread is still alive
            if hasattr(self, 'precalc_thread') and self.precalc_thread and not self.precalc_thread.is_alive():
                self.precalc_running = False

            if not self.precalc_running:
                try:
                    analytics = self.usage_tracker.get_analytics()
                    if analytics and analytics.get('total_uses', 0) > 0:
                        return f"[OK] {len(self.precalc_cache)} cached | {analytics['total_uses']} uses tracked | Most used: {analytics.get('most_popular_crypto', 'N/A')}"
                    else:
                        return f"[OK] {len(self.precalc_cache)} configurations cached (learning usage patterns...)"
                except Exception as e:
                    return f"[OK] {len(self.precalc_cache)} configurations cached"
            else:
                try:
                    progress_pct = (self.precalc_progress / self.precalc_total * 100) if self.precalc_total > 0 else 0
                    status = "Precalculating" if not self.precalc_paused else "Paused for user"
                    return f"{status}: {self.precalc_progress}/{self.precalc_total} ({progress_pct:.1f}%)"
                except Exception as e:
                    return "Precalculating..."
        except Exception as e:
            return "[OK] Status unavailable"
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print(f"Starting Interactive Ladder GUI on http://localhost:{port}")
        print("Note: If you see cached data, please hard refresh your browser (Ctrl+F5)")

        try:
            self.app.run(debug=debug, port=port, dev_tools_hot_reload=True)
        finally:
            # Clean shutdown of thread pool
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of resources"""
        try:
            print("GUI shutdown complete")
        except Exception as e:
            print(f"Error during shutdown: {e}")

if __name__ == "__main__":
    gui = InteractiveLadderGUI()
    gui.run()
