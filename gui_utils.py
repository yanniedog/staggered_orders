"""
Common utilities for the GUI application.
Consolidates shared functions to eliminate duplication and improve maintainability.
"""

import time
import hashlib
import json
from typing import Dict, Any, List, Optional


def generate_cache_key(config: Dict[str, Any]) -> str:
    """
    Generate a unique cache key for a configuration.
    Uses 7-part underscore-separated format for consistency with parsing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Unique cache key string in format: SYMBOL_AGG_RUNGS_HOURS_BUDGET_DISTRIB_POSITION
    """
    # Use 7-part format compatible with _parse_cache_key() in gui_app.py
    return (f"{config['crypto_symbol']}_{config['aggression_level']}_"
            f"{config['num_rungs']}_{config['timeframe_hours']}_"
            f"{config['budget']}_{config['quantity_distribution']}_"
            f"{config['rung_positioning']}")


def map_timeframe_slider_to_hours(slider_value: int) -> int:
    """
    Map timeframe slider value to actual hours.
    
    Args:
        slider_value: Slider value (0-7)
        
    Returns:
        Hours corresponding to slider value
    """
    timeframe_map = {
        0: 24,      # 1 day
        1: 168,     # 1 week
        2: 720,     # 1 month
        3: 4320,    # 6 months
        4: 8760,    # 1 year
        5: 26280,   # 3 years
        6: 43800,   # 5 years
        7: 87600    # max
    }
    return timeframe_map.get(slider_value, 720)


def format_timeframe_hours(timeframe_hours: int) -> str:
    """
    Format timeframe hours into display string.
    
    Args:
        timeframe_hours: Hours to format
        
    Returns:
        Formatted timeframe string
    """
    if timeframe_hours < 24:
        return f"{timeframe_hours}h"
    elif timeframe_hours < 168:
        return f"{timeframe_hours//24}d"
    elif timeframe_hours < 8760:
        return f"{timeframe_hours//168}w"
    else:
        return f"{timeframe_hours//8760}y"


def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration
    """
    validated = config.copy()
    
    # Validate aggression level
    if 'aggression_level' in validated:
        validated['aggression_level'] = max(1, min(5, int(validated['aggression_level'])))
    
    # Validate number of rungs
    if 'num_rungs' in validated:
        validated['num_rungs'] = max(5, min(50, int(validated['num_rungs'])))
    
    # Validate budget
    if 'budget' in validated:
        validated['budget'] = max(100, float(validated['budget']))
    
    # Validate timeframe hours
    if 'timeframe_hours' in validated:
        validated['timeframe_hours'] = max(24, int(validated['timeframe_hours']))
    
    # Validate trading fee
    if 'trading_fee' in validated:
        validated['trading_fee'] = max(0.0, min(1.0, float(validated['trading_fee'])))
    
    # Validate min notional
    if 'min_notional' in validated:
        validated['min_notional'] = max(1.0, float(validated['min_notional']))
    
    return validated


def get_timeframe_labels() -> Dict[int, str]:
    """
    Get timeframe labels for display.
    
    Returns:
        Dictionary mapping slider values to labels
    """
    return {
        0: "1 day",
        1: "1 week", 
        2: "1 month",
        3: "6 months",
        4: "1 year",
        5: "3 years",
        6: "5 years",
        7: "max"
    }


def get_aggression_labels() -> Dict[int, str]:
    """
    Get aggression level labels for display.
    
    Returns:
        Dictionary mapping aggression levels to descriptions
    """
    return {
        1: "Very Conservative",
        2: "Conservative", 
        3: "Moderate",
        4: "Aggressive",
        5: "Very Aggressive"
    }


def calculate_debounce_delay(last_update_time: float, debounce_ms: int = 300) -> bool:
    """
    Check if enough time has passed since last update for debouncing.
    
    Args:
        last_update_time: Timestamp of last update
        debounce_ms: Debounce delay in milliseconds
        
    Returns:
        True if update should proceed, False if should debounce
    """
    current_time = time.time() * 1000
    return current_time - last_update_time >= debounce_ms


def create_error_response(message: str, cache_data: Optional[Dict] = None) -> tuple:
    """
    Create a standardized error response for callbacks.
    
    Args:
        message: Error message to display
        cache_data: Optional cache data to preserve
        
    Returns:
        Tuple of error responses for all callback outputs
    """
    import plotly.graph_objects as go
    import dash_html_components as html
    
    # Create empty error figure
    error_fig = go.Figure()
    error_fig.add_annotation(
        text=f"Error: {message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#dc3545")
    )
    error_fig.update_layout(
        title="Error",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        height=400
    )
    
    # Return error state
    error_figures = (error_fig,) * 9  # 9 charts
    error_kpis = ("Error", "Error", "Error", "Error")  # 4 KPIs
    error_tables = [html.Div(f"Error: {message}", style={'color': '#dc3545'})] * 2  # 2 tables
    
    return (*error_figures, *error_kpis, *error_tables, cache_data or {})


def create_loading_response(cache_data: Optional[Dict] = None) -> tuple:
    """
    Create a standardized loading response for callbacks.
    
    Args:
        cache_data: Optional cache data to preserve
        
    Returns:
        Tuple of loading responses for all callback outputs
    """
    import plotly.graph_objects as go
    import dash_html_components as html
    
    # Create empty loading figure
    loading_fig = go.Figure()
    loading_fig.add_annotation(
        text="Loading...",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#ffc107")
    )
    loading_fig.update_layout(
        title="Loading...",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'size': 12},
        height=400
    )
    
    # Return loading state
    loading_figures = (loading_fig,) * 9  # 9 charts
    loading_kpis = ("Loading...", "Loading...", "Loading...", "Loading...")  # 4 KPIs
    loading_tables = [html.Div("Loading...", style={'color': '#ffc107'})] * 2  # 2 tables
    
    return (*loading_figures, *loading_kpis, *loading_tables, cache_data or {})


def get_chart_priority() -> Dict[str, int]:
    """
    Get chart loading priority mapping.
    
    Returns:
        Dictionary mapping chart IDs to priority levels (1=highest, 3=lowest)
    """
    return {
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


def get_crypto_explanations() -> Dict[str, str]:
    """
    Get cryptocurrency explanations for tooltips.
    
    Returns:
        Dictionary mapping crypto symbols to descriptions
    """
    return {
        'SOLUSDT': "Solana (SOL) - High-performance blockchain with fast transactions and low fees. Good for active trading strategies.",
        'BTCUSDT': "Bitcoin (BTC) - The original cryptocurrency with high liquidity and institutional adoption. Conservative choice.",
        'ETHUSDT': "Ethereum (ETH) - Smart contract platform with strong developer ecosystem. Moderate volatility.",
        'ADAUSDT': "Cardano (ADA) - Research-driven blockchain with focus on sustainability and scalability.",
        'DOTUSDT': "Polkadot (DOT) - Multi-chain protocol enabling different blockchains to transfer messages and value.",
        'LINKUSDT': "Chainlink (LINK) - Decentralized oracle network providing real-world data to smart contracts.",
        'UNIUSDT': "Uniswap (UNI) - Leading decentralized exchange protocol with automated market making.",
        'LTCUSDT': "Litecoin (LTC) - Bitcoin's 'silver' with faster block times and lower transaction fees.",
        'BNBUSDT': "Binance Coin (BNB) - Native token of Binance exchange with utility across the ecosystem.",
        'MATICUSDT': "Polygon (MATIC) - Layer 2 scaling solution for Ethereum with low fees and fast transactions."
    }


def get_timeframe_explanations() -> Dict[int, str]:
    """
    Get timeframe explanations for tooltips.
    
    Returns:
        Dictionary mapping timeframe slider values to descriptions
    """
    return {
        0: "1 Day - Very short-term analysis for day trading strategies. High volatility, quick fills expected.",
        1: "1 Week - Short-term analysis for swing trading. Moderate volatility with daily touch events.",
        2: "1 Month - Medium-term analysis balancing volatility and stability. Good for most trading strategies.",
        3: "6 Months - Longer-term analysis capturing seasonal patterns and market cycles.",
        4: "1 Year - Long-term analysis for position trading. Lower volatility, fewer but more significant moves.",
        5: "3 Years - Extended analysis including major market cycles and long-term trends.",
        6: "5 Years - Comprehensive analysis covering multiple market cycles and major events.",
        7: "Maximum - Uses all available historical data for the most comprehensive analysis possible."
    }


def get_quantity_distribution_explanations() -> Dict[str, str]:
    """
    Get quantity distribution method explanations.
    
    Returns:
        Dictionary mapping method names to descriptions
    """
    return {
        'kelly_optimized': "Kelly-optimized allocation maximizes long-term growth rate while managing risk through optimal position sizing.",
        'adaptive_kelly': "Adaptive Kelly adjusts position sizes based on changing market conditions and volatility patterns.",
        'volatility_weighted': "Volatility-weighted allocation assigns more capital to rungs with lower volatility for stable returns.",
        'sharpe_maximizing': "Sharpe-maximizing allocation optimizes the risk-adjusted return ratio across all ladder positions.",
        'fibonacci_weighted': "Fibonacci-weighted allocation uses Fibonacci ratios to distribute capital across ladder rungs.",
        'risk_parity': "Risk parity allocation equalizes risk contribution from each rung rather than equal capital allocation.",
        'price_weighted': "Price-weighted allocation assigns more capital to rungs closer to current market price.",
        'equal_notional': "Equal notional allocation distributes capital equally across all rungs for consistent position sizes.",
        'equal_quantity': "Equal quantity allocation places the same number of units at each rung regardless of price.",
        'linear_increase': "Linear increase allocation gradually increases position sizes from top to bottom rungs.",
        'exponential_increase': "Exponential increase allocation uses exponential scaling for position sizes across rungs.",
        'probability_weighted': "Probability-weighted allocation allocates more capital to rungs with higher touch probabilities."
    }


def get_rung_positioning_explanations() -> Dict[str, str]:
    """
    Get rung positioning method explanations.
    
    Returns:
        Dictionary mapping method names to descriptions
    """
    return {
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


def generate_smart_recommendations(aggression: int, rungs: int, timeframe: int, budget: float, 
                                 quantity_dist: str, crypto: str, positioning: str) -> List[Dict[str, str]]:
    """
    Generate smart recommendations based on current settings.
    
    Args:
        aggression: Aggression level (1-5)
        rungs: Number of rungs
        timeframe: Timeframe slider value
        budget: Budget amount
        quantity_dist: Quantity distribution method
        crypto: Cryptocurrency symbol
        positioning: Rung positioning method
        
    Returns:
        List of recommendation dictionaries
    """
    recommendations = []
    
    # Budget-based recommendations
    if budget and budget < 500:
        recommendations.append({
            'type': 'warning',
            'title': 'Low Budget Optimization',
            'message': 'Consider reducing rungs to 10-15 for better allocation per rung with this budget.',
            'suggestion': 'Try 15 rungs for optimal capital efficiency'
        })
    elif budget and budget > 10000:
        recommendations.append({
            'type': 'info',
            'title': 'High Budget Strategy',
            'message': 'With this budget, consider increasing rungs to 25-30 for better market coverage.',
            'suggestion': 'Try 25-30 rungs for comprehensive market coverage'
        })
    
    # Aggression-based recommendations
    if aggression == 1:
        recommendations.append({
            'type': 'success',
            'title': 'Conservative Strategy',
            'message': 'Perfect for risk-averse trading. Consider Equal Notional for stable returns.',
            'suggestion': 'Switch to Equal Notional quantity distribution'
        })
    elif aggression >= 4:
        recommendations.append({
            'type': 'warning',
            'title': 'High Risk Strategy',
            'message': 'High aggression requires careful risk management. Kelly Optimized is recommended.',
            'suggestion': 'Use Kelly Optimized for risk-adjusted returns'
        })
    
    # Timeframe-based recommendations
    timeframe_hours = map_timeframe_slider_to_hours(timeframe)
    
    if timeframe_hours <= 168:  # 1 week or less
        recommendations.append({
            'type': 'info',
            'title': 'Short-term Trading',
            'message': 'Short timeframe requires higher rung density. Consider Dynamic Density positioning.',
            'suggestion': 'Use Dynamic Density for short-term volatility adaptation'
        })
    elif timeframe_hours >= 8760:  # 1 year or more
        recommendations.append({
            'type': 'success',
            'title': 'Long-term Strategy',
            'message': 'Long timeframe allows for broader market coverage. Linear or Fibonacci positioning works well.',
            'suggestion': 'Consider Fibonacci levels for long-term technical analysis'
        })
    
    # Crypto-specific recommendations
    if crypto == 'BTCUSDT':
        recommendations.append({
            'type': 'info',
            'title': 'Bitcoin Strategy',
            'message': 'BTC has high liquidity and institutional support. Conservative aggression (2-3) recommended.',
            'suggestion': 'Consider aggression level 2-3 for Bitcoin'
        })
    elif crypto == 'SOLUSDT':
        recommendations.append({
            'type': 'success',
            'title': 'Solana Strategy',
            'message': 'SOL is great for active trading due to low fees. Higher aggression (3-4) can work well.',
            'suggestion': 'Try aggression level 3-4 for Solana'
        })
    
    # Rung count recommendations
    if rungs < 10:
        recommendations.append({
            'type': 'warning',
            'title': 'Low Rung Count',
            'message': 'Few rungs may miss market opportunities. Consider increasing to 15-20.',
            'suggestion': 'Increase to 15-20 rungs for better coverage'
        })
    elif rungs > 30:
        recommendations.append({
            'type': 'info',
            'title': 'High Rung Count',
            'message': 'Many rungs provide fine granularity but may fragment capital. Monitor allocation per rung.',
            'suggestion': 'Consider reducing to 20-25 rungs for better capital concentration'
        })
    
    return recommendations[:3]  # Limit to 3 recommendations
