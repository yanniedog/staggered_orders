"""
Simplified analysis module consolidating core functions.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from typing import Tuple, Dict
import warnings

def analyze_touch_probabilities(df: pd.DataFrame, max_hours: int = 720) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze touch probabilities from historical data.
    
    Args:
        df: DataFrame with OHLCV data
        max_hours: Maximum analysis window in hours
    
    Returns:
        Tuple of (depths, probabilities)
    """
    print(f"Analyzing touch probabilities for {len(df)} bars...")
    
    # Calculate horizon in bars (assuming 1h candles)
    horizon_bars = max(1, int(max_hours))
    
    depths = []
    touches = []
    
    for i in range(len(df) - horizon_bars):
        current_price = df.iloc[i]['close']
        future_bars = df.iloc[i+1:i+1+horizon_bars]
        
        if len(future_bars) == 0:
            continue
            
        # Find minimum price (lowest low) in future window
        min_price = future_bars['low'].min()
        max_drop = (current_price - min_price) / current_price * 100
        
        if max_drop > 0:
            depths.append(max_drop)
            touches.append(1)
        else:
            depths.append(0)
            touches.append(0)
    
    # Convert to numpy arrays
    depths = np.array(depths)
    touches = np.array(touches)
    
    # Calculate empirical probabilities for depth bins
    depth_bins = np.linspace(0, 20, 50)  # 0% to 20% in 50 bins
    empirical_probs = []
    
    for i in range(len(depth_bins) - 1):
        bin_mask = (depths >= depth_bins[i]) & (depths < depth_bins[i + 1])
        if np.sum(bin_mask) > 0:
            prob = np.mean(touches[bin_mask])
            empirical_probs.append(prob)
        else:
            empirical_probs.append(0)
    
    # Use bin centers as depths
    bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
    
    return np.array(bin_centers), np.array(empirical_probs)

def weibull_tail(depth: np.ndarray, theta: float, p: float) -> np.ndarray:
    """Weibull tail distribution: q(d) = exp(-(depth/theta)^p)"""
    return np.exp(-(depth / theta) ** p)

def fit_weibull_tail(depths: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fit Weibull tail distribution to touch probabilities.
    
    Args:
        depths: Array of depths (percentages)
        probabilities: Array of empirical probabilities
    
    Returns:
        Tuple of (theta, p, fit_metrics)
    """
    print(f"Fitting Weibull distribution to {len(depths)} data points...")
    
    # Filter out zero probabilities for fitting
    valid_mask = probabilities > 0
    if np.sum(valid_mask) < 3:
        raise ValueError("Insufficient valid data points for fitting")
    
    valid_depths = depths[valid_mask]
    valid_probs = probabilities[valid_mask]
    
    # Initial parameter guesses
    theta_guess = np.mean(valid_depths)
    p_guess = 1.0
    
    try:
        # Fit the curve
        popt, pcov = curve_fit(
            weibull_tail, 
            valid_depths, 
            valid_probs,
            p0=[theta_guess, p_guess],
            bounds=([0.1, 0.1], [100, 10]),
            maxfev=10000
        )
        
        theta, p = popt
        
        # Calculate fit quality metrics
        fitted_probs = weibull_tail(valid_depths, theta, p)
        r_squared = pearsonr(valid_probs, fitted_probs)[0] ** 2
        
        metrics = {
            'r_squared': r_squared,
            'theta': theta,
            'p': p,
            'n_points': len(valid_depths)
        }
        
        print(f"Fit complete: θ={theta:.3f}, p={p:.3f}, R²={r_squared:.4f}")
        
        return theta, p, metrics
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        # Return default values
        return 2.0, 1.0, {'r_squared': 0.0, 'theta': 2.0, 'p': 1.0, 'n_points': 0}

def calculate_ladder_depths(theta: float, p: float, num_rungs: int = 30) -> np.ndarray:
    """
    Calculate ladder depths using Weibull quantiles.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        num_rungs: Number of ladder rungs
    
    Returns:
        Array of ladder depths
    """
    print(f"Calculating {num_rungs} ladder depths...")
    
    # Use quantile-based spacing
    quantiles = np.linspace(0.1, 0.9, num_rungs)
    
    # Convert quantiles to depths using Weibull inverse CDF
    depths = []
    for q in quantiles:
        if q <= 0:
            depth = 0.0
        elif q >= 1:
            depth = np.inf
        else:
            depth = theta * (-np.log(1 - q)) ** (1 / p)
        depths.append(depth)
    
    depths = np.array(depths)
    print(f"Depth range: {depths.min():.2f}% - {depths.max():.2f}%")
    
    return depths

def optimize_sizes(depths: np.ndarray, theta: float, p: float, budget: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Optimize position sizes using Kelly criterion.
    
    Args:
        depths: Array of ladder depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
        budget: Total budget in USD
    
    Returns:
        Tuple of (allocations, alpha, expected_returns)
    """
    print(f"Optimizing sizes for ${budget:.0f} budget...")
    
    # Calculate touch probabilities
    touch_probs = weibull_tail(depths, theta, p)
    
    # Calculate expected returns per dollar (depth * probability)
    expected_returns = depths * touch_probs
    
    # Use Kelly criterion: allocation proportional to expected return
    # Normalize to budget
    raw_allocations = expected_returns
    allocations = raw_allocations / np.sum(raw_allocations) * budget
    
    # Calculate alpha parameter (for monotonicity)
    alpha = 1.0  # Simplified
    
    print(f"Allocation range: ${allocations.min():.0f} - ${allocations.max():.0f}")
    
    return allocations, alpha, expected_returns

def weibull_touch_probability(depth: float, theta: float, p: float) -> float:
    """Calculate touch probability using Weibull distribution."""
    return np.exp(-(depth / theta) ** p)
