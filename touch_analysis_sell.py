"""
Upward touch probability analysis for upward wicks.
Mirror of touch_analysis.py but for analyzing upward price movements.
"""
import pandas as pd
import numpy as np
from typing import Tuple
import warnings


def compute_max_pumps(df: pd.DataFrame, max_analysis_hours: int, candle_interval: str = "1h") -> np.ndarray:
    """
    Compute maximum pump from each bar within the analysis window.
    Optimized for flash pump detection with configurable candle intervals.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        candle_interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
    
    Returns:
        Array of maximum pumps (as percentages)
    """
    # Calculate interval duration in minutes
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
    }.get(candle_interval, 1)
    
    # Calculate how many bars represent the analysis window
    horizon_bars = max(1, int(max_analysis_hours * 60 / interval_minutes))
    
    max_pumps = []
    
    print(f"Computing max pumps for {len(df)} bars with {horizon_bars}-bar window ({max_analysis_hours} hours)...")
    print(f"Candle interval: {candle_interval} ({interval_minutes} minutes per bar)")
    
    for i in range(len(df) - horizon_bars):
        current_price = df.iloc[i]['close']
        
        # Look ahead within horizon
        future_bars = df.iloc[i+1:i+1+horizon_bars]
        
        if len(future_bars) == 0:
            continue
            
        # Find maximum price (highest high) in the horizon
        max_price = future_bars['high'].max()
        
        # Calculate maximum pump as percentage
        max_pump = (max_price - current_price) / current_price * 100
        
        max_pumps.append(max_pump)
        
        # Progress indicator for large datasets
        if i % 50000 == 0 and i > 0:
            print(f"  Processed {i:,} bars...")
    
    return np.array(max_pumps)


def build_upward_touch_probability_curve(max_pumps: np.ndarray, 
                                        depth_range: Tuple[float, float] = (0.1, 10.0),
                                        num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build empirical upward touch probability curve.
    
    Args:
        max_pumps: Array of maximum pumps
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
    total_samples = len(max_pumps)
    
    for depth in depths:
        # Fraction of samples where pump >= depth
        prob = np.sum(max_pumps >= depth) / total_samples
        probabilities.append(prob)
    
    return depths, np.array(probabilities)


def filter_valid_pumps(max_pumps: np.ndarray, min_prob: float = 0.0001) -> np.ndarray:
    """
    Filter out pumps that are too rare to be meaningful.
    Optimized for flash pump detection with more permissive filtering.
    
    Args:
        max_pumps: Array of maximum pumps
        min_prob: Minimum probability threshold (lower for flash pumps)
    
    Returns:
        Filtered array of pumps
    """
    # Calculate empirical probabilities
    depths, probs = build_upward_touch_probability_curve(max_pumps, depth_range=(0.05, 20.0), num_points=200)
    
    # Find maximum depth where probability >= min_prob
    valid_mask = probs >= min_prob
    if not np.any(valid_mask):
        warnings.warn(f"No pumps found with probability >= {min_prob}")
        return max_pumps
    
    max_valid_depth = depths[valid_mask][-1]
    
    # Filter pumps - keep more extreme events for flash pump analysis
    filtered_pumps = max_pumps[max_pumps <= max_valid_depth]
    
    print(f"Filtered pumps: {len(max_pumps)} -> {len(filtered_pumps)}")
    print(f"Max valid depth: {max_valid_depth:.2f}%")
    print(f"Flash pump threshold (>2%): {np.sum(max_pumps > 2.0)} events")
    print(f"Extreme pump threshold (>5%): {np.sum(max_pumps > 5.0)} events")
    print(f"Major pump threshold (>10%): {np.sum(max_pumps > 10.0)} events")
    
    return filtered_pumps


def calculate_upward_touch_frequency(df: pd.DataFrame, max_analysis_hours: int, 
                                   depths: np.ndarray, candle_interval: str = "1h") -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate upward touch frequency and expected time-to-touch for each depth.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depths: Array of depth levels to analyze
        candle_interval: Candle interval
    
    Returns:
        Tuple of (touch_frequencies, expected_timeframes_hours)
    """
    # Compute max pumps for the entire dataset
    max_pumps = compute_max_pumps(df, max_analysis_hours, candle_interval)
    
    # Calculate frequency of touches at each depth
    total_samples = len(max_pumps)
    touch_frequencies = []
    expected_timeframes = []
    
    for depth in depths:
        # Count touches at this depth or higher
        touches = np.sum(max_pumps >= depth)
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


def calculate_upward_joint_probabilities(df: pd.DataFrame, max_analysis_hours: int,
                                       depths: np.ndarray, candle_interval: str = "1h") -> np.ndarray:
    """
    Calculate joint probabilities for multiple upward rung fills.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depths: Array of depth levels
        candle_interval: Candle interval
    
    Returns:
        Array of joint probability matrices
    """
    max_pumps = compute_max_pumps(df, max_analysis_hours, candle_interval)
    n_depths = len(depths)
    
    # Create joint probability matrix
    joint_probs = np.zeros((n_depths, n_depths))
    
    for i, depth_i in enumerate(depths):
        for j, depth_j in enumerate(depths):
            if i <= j:  # Only calculate upper triangle (symmetric)
                # Probability that both depths are touched
                touches_i = np.sum(max_pumps >= depth_i)
                touches_j = np.sum(max_pumps >= depth_j)
                touches_both = np.sum((max_pumps >= depth_i) & (max_pumps >= depth_j))
                
                # Joint probability (both touched in same event)
                joint_prob = touches_both / len(max_pumps)
                joint_probs[i, j] = joint_prob
                joint_probs[j, i] = joint_prob  # Make symmetric
    
    return joint_probs


def create_upward_timeframe_distribution(df: pd.DataFrame, max_analysis_hours: int,
                                       depth: float, candle_interval: str = "1h", num_periods: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create empirical distribution of time-to-touch for a specific upward depth.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        depth: Specific depth to analyze
        candle_interval: Candle interval
        num_periods: Number of time periods to analyze
    
    Returns:
        Tuple of (time_periods, cumulative_probabilities)
    """
    max_pumps = compute_max_pumps(df, max_analysis_hours, candle_interval)
    
    # Calculate time periods (in hours)
    time_periods = np.arange(1, num_periods + 1) * max_analysis_hours / num_periods
    
    # For each time period, calculate probability of first touch
    cumulative_probs = []
    
    for period_hours in time_periods:
        # Calculate probability of touch within this time period
        # Using the empirical distribution from our analysis window
        touches_in_period = np.sum(max_pumps >= depth)
        total_samples = len(max_pumps)
        
        # Probability of touch within the analysis window
        single_period_prob = touches_in_period / total_samples
        
        # Scale probability based on time period relative to analysis window
        time_ratio = period_hours / max_analysis_hours
        cumulative_prob = min(1.0, single_period_prob * time_ratio)
            
        cumulative_probs.append(cumulative_prob)
    
    return time_periods, np.array(cumulative_probs)


def analyze_upward_touch_probabilities(df: pd.DataFrame, max_analysis_hours: int, candle_interval: str = "1h") -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete upward touch probability analysis with enhanced timeframe analysis.
    
    Args:
        df: DataFrame with OHLCV data
        max_analysis_hours: Maximum time window to analyze in hours
        candle_interval: Candle interval
    
    Returns:
        Tuple of (depths, empirical_probabilities)
    """
    print(f"Computing max pumps for analysis window: {max_analysis_hours} hours")
    
    # Compute maximum pumps
    max_pumps = compute_max_pumps(df, max_analysis_hours, candle_interval)
    
    print(f"Computed {len(max_pumps)} max pumps")
    print(f"Pump statistics:")
    print(f"  Mean: {np.mean(max_pumps):.3f}%")
    print(f"  Median: {np.median(max_pumps):.3f}%")
    print(f"  95th percentile: {np.percentile(max_pumps, 95):.3f}%")
    print(f"  99th percentile: {np.percentile(max_pumps, 99):.3f}%")
    print(f"  99.9th percentile: {np.percentile(max_pumps, 99.9):.3f}%")
    print(f"  Max: {np.max(max_pumps):.3f}%")
    
    # Flash pump analysis
    flash_pumps_2pct = np.sum(max_pumps >= 2.0)
    flash_pumps_5pct = np.sum(max_pumps >= 5.0)
    flash_pumps_10pct = np.sum(max_pumps >= 10.0)
    
    print(f"Flash pump analysis:")
    print(f"  Pumps >= 2%: {flash_pumps_2pct} ({flash_pumps_2pct/len(max_pumps)*100:.2f}%)")
    print(f"  Pumps >= 5%: {flash_pumps_5pct} ({flash_pumps_5pct/len(max_pumps)*100:.2f}%)")
    print(f"  Pumps >= 10%: {flash_pumps_10pct} ({flash_pumps_10pct/len(max_pumps)*100:.2f}%)")
    
    # Filter valid pumps
    filtered_pumps = filter_valid_pumps(max_pumps)
    
    # Build upward touch probability curve
    depths, probabilities = build_upward_touch_probability_curve(filtered_pumps)
    
    print(f"Built upward touch probability curve with {len(depths)} points")
    print(f"Depth range: {depths[0]:.2f}% - {depths[-1]:.2f}%")
    print(f"Probability range: {probabilities[-1]:.4f} - {probabilities[0]:.4f}")
    
    return depths, probabilities


if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import fetch_solusdt_data
    
    df = fetch_solusdt_data()
    depths, probs = analyze_upward_touch_probabilities(df, max_analysis_hours=24)
    
    print("\nSample upward touch probabilities:")
    for i in range(0, len(depths), 10):
        print(f"  {depths[i]:.2f}%: {probs[i]:.4f}")
