"""
Unified touch probability analysis for both buy and sell directions.
Consolidates downward and upward wick analysis into a single bidirectional module.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import warnings


def compute_max_movements(df: pd.DataFrame, max_analysis_hours: int, 
                         direction: str = 'buy', candle_interval: str = "1h") -> np.ndarray:
    """
    Compute maximum price movements from each bar within the analysis window.
    Unified function for both buy (downward) and sell (upward) analysis.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        direction: 'buy' for downward movements, 'sell' for upward movements
        candle_interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
    
    Returns:
        Array of maximum movements (as percentages)
    """
    # Calculate interval duration in minutes
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
    }.get(candle_interval, 60)
    
    # Calculate how many bars represent the analysis window
    horizon_bars = max(1, int(max_analysis_hours * 60 / interval_minutes))
    
    max_movements = []
    movement_type = "drops" if direction == 'buy' else "pumps"
    
    print(f"Computing max {movement_type} for {len(df)} bars with {horizon_bars}-bar window ({max_analysis_hours} hours)...")
    print(f"Candle interval: {candle_interval} ({interval_minutes} minutes per bar)")
    
    for i in range(len(df) - horizon_bars):
        current_price = df.iloc[i]['close']
        
        # Look ahead within horizon
        future_bars = df.iloc[i+1:i+1+horizon_bars]
        
        if len(future_bars) == 0:
            continue
            
        if direction == 'buy':
            # Find minimum price (lowest low) for downward movements
            min_price = future_bars['low'].min()
            max_movement = (current_price - min_price) / current_price * 100
        else:
            # Find maximum price (highest high) for upward movements
            max_price = future_bars['high'].max()
            max_movement = (max_price - current_price) / current_price * 100
        
        max_movements.append(max_movement)
        
        # Progress indicator for large datasets
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} bars...")
    
    return np.array(max_movements)


def build_touch_probability_curve(max_movements: np.ndarray, 
                                 depth_range: Tuple[float, float] = (0.1, 10.0),
                                 num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build empirical touch probability curve for any direction.
    
    Args:
        max_movements: Array of maximum movements
        depth_range: (min_depth, max_depth) in percentage
        num_points: Number of depth points to evaluate
    
    Returns:
        Tuple of (depths, probabilities)
    """
    min_depth, max_depth = depth_range
    
    # Create depth grid
    depths = np.linspace(min_depth, max_depth, num_points)
    
    # Calculate empirical probabilities
    probabilities = []
    total_samples = len(max_movements)
    
    for depth in depths:
        # Fraction of samples where movement >= depth
        prob = np.sum(max_movements >= depth) / total_samples
        probabilities.append(prob)
    
    return depths, np.array(probabilities)


def filter_valid_movements(max_movements: np.ndarray, min_prob: float = 0.0001) -> np.ndarray:
    """
    Filter out movements that are too rare to be meaningful.
    
    Args:
        max_movements: Array of maximum movements
        min_prob: Minimum probability threshold
    
    Returns:
        Filtered array of movements
    """
    # Calculate empirical probabilities
    depths, probs = build_touch_probability_curve(max_movements, depth_range=(0.05, 20.0), num_points=200)
    
    # Find maximum depth where probability >= min_prob
    valid_mask = probs >= min_prob
    if not np.any(valid_mask):
        warnings.warn(f"No movements found with probability >= {min_prob}")
        return max_movements
    
    max_valid_depth = depths[valid_mask][-1]
    
    # Filter movements
    filtered_movements = max_movements[max_movements <= max_valid_depth]
    
    print(f"Filtered movements: {len(max_movements)} -> {len(filtered_movements)}")
    print(f"Max valid depth: {max_valid_depth:.2f}%")
    print(f"Flash movement threshold (>2%): {np.sum(max_movements > 2.0)} events")
    print(f"Extreme movement threshold (>5%): {np.sum(max_movements > 5.0)} events")
    print(f"Major movement threshold (>10%): {np.sum(max_movements > 10.0)} events")
    
    return filtered_movements


def calculate_touch_frequency(df: pd.DataFrame, max_analysis_hours: int, 
                            depths: np.ndarray, direction: str = 'buy', 
                            candle_interval: str = "1h") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate touch frequency and expected time-to-touch for each depth.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depths: Array of depth levels to analyze
        direction: 'buy' for downward, 'sell' for upward
        candle_interval: Candle interval
    
    Returns:
        Tuple of (touch_frequencies, expected_timeframes_hours)
    """
    # Compute max movements for the entire dataset
    max_movements = compute_max_movements(df, max_analysis_hours, direction, candle_interval)
    
    # Calculate frequency of touches at each depth
    total_samples = len(max_movements)
    touch_frequencies = []
    expected_timeframes = []
    
    for depth in depths:
        # Count touches at this depth or higher
        touches = np.sum(max_movements >= depth)
        frequency = touches / total_samples
        
        # Estimate expected time-to-touch
        if frequency > 0:
            expected_attempts = 1.0 / frequency
            expected_timeframe_hours = expected_attempts * max_analysis_hours
        else:
            expected_timeframe_hours = np.inf
            
        touch_frequencies.append(frequency)
        expected_timeframes.append(expected_timeframe_hours)
    
    return np.array(touch_frequencies), np.array(expected_timeframes)


def calculate_joint_probabilities(df: pd.DataFrame, max_analysis_hours: int,
                                depths: np.ndarray, direction: str = 'buy', 
                                candle_interval: str = "1h") -> np.ndarray:
    """
    Calculate joint probabilities for multiple rung fills.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depths: Array of depth levels
        direction: 'buy' for downward, 'sell' for upward
        candle_interval: Candle interval
    
    Returns:
        Array of joint probability matrices
    """
    max_movements = compute_max_movements(df, max_analysis_hours, direction, candle_interval)
    n_depths = len(depths)
    
    # Create joint probability matrix
    joint_probs = np.zeros((n_depths, n_depths))
    
    for i, depth_i in enumerate(depths):
        for j, depth_j in enumerate(depths):
            if i <= j:  # Only calculate upper triangle (symmetric)
                # Probability that both depths are touched
                touches_i = np.sum(max_movements >= depth_i)
                touches_j = np.sum(max_movements >= depth_j)
                touches_both = np.sum((max_movements >= depth_i) & (max_movements >= depth_j))
                
                # Joint probability (both touched in same event)
                joint_prob = touches_both / len(max_movements)
                joint_probs[i, j] = joint_prob
                joint_probs[j, i] = joint_prob  # Make symmetric
    
    return joint_probs


def create_timeframe_distribution(df: pd.DataFrame, max_analysis_hours: int,
                                depth: float, direction: str = 'buy', 
                                candle_interval: str = "1h", num_periods: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create empirical distribution of time-to-touch for a specific depth.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depth: Specific depth to analyze
        direction: 'buy' for downward, 'sell' for upward
        candle_interval: Candle interval
        num_periods: Number of time periods to analyze
    
    Returns:
        Tuple of (time_periods, cumulative_probabilities)
    """
    max_movements = compute_max_movements(df, max_analysis_hours, direction, candle_interval)
    
    # Calculate time periods (in hours)
    time_periods = np.arange(1, num_periods + 1) * max_analysis_hours / num_periods
    
    # For each time period, calculate probability of first touch
    cumulative_probs = []
    
    for period_hours in time_periods:
        # Calculate probability of touch within this time period
        touches_in_period = np.sum(max_movements >= depth)
        total_samples = len(max_movements)
        
        # Probability of touch within the analysis window
        single_period_prob = touches_in_period / total_samples
        
        # Scale probability based on time period relative to analysis window
        time_ratio = period_hours / max_analysis_hours
        cumulative_prob = min(1.0, single_period_prob * time_ratio)
            
        cumulative_probs.append(cumulative_prob)
    
    return time_periods, np.array(cumulative_probs)


def analyze_touch_probabilities(df: pd.DataFrame, max_analysis_hours: int, 
                               candle_interval: str = "1h", direction: str = 'buy') -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete touch probability analysis with enhanced timeframe analysis.
    Unified function for both buy and sell directions.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        candle_interval: Candle interval
        direction: 'buy' for downward movements, 'sell' for upward movements
    
    Returns:
        Tuple of (depths, empirical_probabilities)
    """
    movement_type = "drops" if direction == 'buy' else "pumps"
    print(f"Computing max {movement_type} for analysis window: {max_analysis_hours} hours")
    
    # Compute maximum movements
    max_movements = compute_max_movements(df, max_analysis_hours, direction, candle_interval)
    
    print(f"Computed {len(max_movements)} max {movement_type}")
    print(f"Movement statistics:")
    print(f"  Mean: {np.mean(max_movements):.3f}%")
    print(f"  Median: {np.median(max_movements):.3f}%")
    print(f"  95th percentile: {np.percentile(max_movements, 95):.3f}%")
    print(f"  99th percentile: {np.percentile(max_movements, 99):.3f}%")
    print(f"  99.9th percentile: {np.percentile(max_movements, 99.9):.3f}%")
    print(f"  Max: {np.max(max_movements):.3f}%")
    
    # Flash movement analysis
    flash_movements_2pct = np.sum(max_movements >= 2.0)
    flash_movements_5pct = np.sum(max_movements >= 5.0)
    flash_movements_10pct = np.sum(max_movements >= 10.0)
    
    print(f"Flash {movement_type} analysis:")
    print(f"  Movements >= 2%: {flash_movements_2pct} ({flash_movements_2pct/len(max_movements)*100:.2f}%)")
    print(f"  Movements >= 5%: {flash_movements_5pct} ({flash_movements_5pct/len(max_movements)*100:.2f}%)")
    print(f"  Movements >= 10%: {flash_movements_10pct} ({flash_movements_10pct/len(max_movements)*100:.2f}%)")
    
    # Filter valid movements
    filtered_movements = filter_valid_movements(max_movements)
    
    # Build touch probability curve
    depths, probabilities = build_touch_probability_curve(filtered_movements)
    
    print(f"Built {direction} touch probability curve with {len(depths)} points")
    print(f"Depth range: {depths[0]:.2f}% - {depths[-1]:.2f}%")
    print(f"Probability range: {probabilities[-1]:.4f} - {probabilities[0]:.4f}")
    
    return depths, probabilities


# Backward compatibility functions
def analyze_upward_touch_probabilities(df: pd.DataFrame, max_analysis_hours: int, candle_interval: str = "1h") -> Tuple[np.ndarray, np.ndarray]:
    """Backward compatibility wrapper for sell-side analysis."""
    return analyze_touch_probabilities(df, max_analysis_hours, candle_interval, direction='sell')


if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import fetch_solusdt_data
    
    df = fetch_solusdt_data()
    
    # Test buy-side analysis
    print("=== BUY-SIDE ANALYSIS ===")
    depths_buy, probs_buy = analyze_touch_probabilities(df, max_analysis_hours=24, direction='buy')
    
    print("\nSample buy touch probabilities:")
    for i in range(0, len(depths_buy), 10):
        print(f"  {depths_buy[i]:.2f}%: {probs_buy[i]:.4f}")
    
    # Test sell-side analysis
    print("\n=== SELL-SIDE ANALYSIS ===")
    depths_sell, probs_sell = analyze_touch_probabilities(df, max_analysis_hours=24, direction='sell')
    
    print("\nSample sell touch probabilities:")
    for i in range(0, len(depths_sell), 10):
        print(f"  {depths_sell[i]:.2f}%: {probs_sell[i]:.4f}")