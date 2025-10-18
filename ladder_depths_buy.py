"""
Ladder depth calculation using Weibull distribution quantiles.
"""
import numpy as np
from typing import Tuple
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def weibull_cdf(depth: float, theta: float, p: float) -> float:
    """
    Weibull cumulative distribution function.
    
    Args:
        depth: Depth value (percentage)
        theta: Scale parameter
        p: Shape parameter
    
    Returns:
        Cumulative probability
    """
    return 1 - np.exp(-(depth / theta) ** p)


def weibull_quantile(quantile: float, theta: float, p: float) -> float:
    """
    Weibull quantile function (inverse CDF).
    
    Args:
        quantile: Probability value (0-1)
        theta: Scale parameter
        p: Shape parameter
    
    Returns:
        Depth value at given quantile
    """
    if quantile <= 0:
        return 0.0
    if quantile >= 1:
        return np.inf
    
    return theta * (-np.log(1 - quantile)) ** (1 / p)


def calculate_depth_range(theta: float, p: float, 
                        u_min: float, p_min: float) -> Tuple[float, float]:
    """
    Calculate minimum and maximum depths for the ladder.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        u_min: Top rung quantile (0-1)
        p_min: Bottom rung tail probability (0-1)
    
    Returns:
        Tuple of (d_min, d_max)
    """
    # Top rung: quantile of the fitted distribution
    d_min = weibull_quantile(u_min, theta, p)
    
    # Bottom rung: depth where probability = p_min
    # q(d_max) = p_min = exp(-(d_max/theta)^p)
    # d_max = theta * (-ln(p_min))^(1/p)
    d_max = theta * (-np.log(p_min)) ** (1 / p)
    
    print(f"Depth range calculation:")
    print(f"  d_min (top rung): {d_min:.3f}% (quantile {u_min:.2f})")
    print(f"  d_max (bottom rung): {d_max:.3f}% (tail prob {p_min:.3f})")
    print(f"  Range: {d_max - d_min:.3f}%")
    
    return d_min, d_max


def calculate_expected_value_depths(theta: float, p: float, d_min: float, d_max: float, 
                                   num_rungs: int, current_price: float, 
                                   profit_target_pct: float = 3.0) -> np.ndarray:
    """
    Calculate ladder depths using expected value optimization.
    
    Positions more rungs in high expected value zones where:
    E[profit] = touch_probability(d) × profit_potential(d) is maximized.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        d_min: Minimum depth
        d_max: Maximum depth
        num_rungs: Number of rungs
        current_price: Current market price
        profit_target_pct: Target profit percentage per pair
    
    Returns:
        Array of optimized ladder depths
    """
    # Create fine-grained depth range for analysis
    fine_depths = np.linspace(d_min, d_max, 1000)
    
    # Calculate expected value for each depth
    # E[profit] = touch_probability × profit_potential
    touch_probs = np.exp(-(fine_depths / theta) ** p)
    
    # Profit potential = profit_target_pct (simplified - could be more sophisticated)
    profit_potential = np.full_like(fine_depths, profit_target_pct)
    
    # Expected value = probability × profit
    expected_values = touch_probs * profit_potential
    
    # Find peaks and high-EV zones
    # Use density-based clustering to identify optimal zones
    from scipy.signal import find_peaks
    
    # Find peaks in expected value
    peaks, _ = find_peaks(expected_values, height=np.mean(expected_values))
    
    if len(peaks) == 0:
        # Fallback to quantile-based if no clear peaks
        print("No clear peaks found, using quantile-based positioning")
        return generate_ladder_depths(theta, p, d_min, d_max, num_rungs)
    
    # Create weighted sampling based on expected value
    # Higher expected value = higher probability of being selected
    weights = expected_values / np.sum(expected_values)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Sample depths with probability proportional to expected value
    selected_indices = np.random.choice(len(fine_depths), size=num_rungs, 
                                      replace=False, p=weights)
    
    # Sort selected depths
    selected_depths = np.sort(fine_depths[selected_indices])
    
    # Ensure we have coverage across the range
    # Add minimum and maximum if not already selected
    if selected_depths[0] > d_min + 0.1:
        selected_depths[0] = d_min
    if selected_depths[-1] < d_max - 0.1:
        selected_depths[-1] = d_max
    
    # Sort again after adjustments
    selected_depths = np.sort(selected_depths)
    
    print(f"Expected value depth optimization:")
    print(f"  Fine-grained analysis: {len(fine_depths)} points")
    print(f"  Expected value range: {np.min(expected_values):.4f} - {np.max(expected_values):.4f}")
    print(f"  Selected {num_rungs} depths with EV-weighted positioning")
    print(f"  Depth range: {selected_depths[0]:.3f}% - {selected_depths[-1]:.3f}%")
    
    return selected_depths


def generate_ladder_depths(theta: float, p: float, 
                          d_min: float, d_max: float, 
                          num_rungs: int) -> np.ndarray:
    """
    Generate ladder depths using quantile spacing.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        d_min: Minimum depth
        d_max: Maximum depth
        num_rungs: Number of rungs
    
    Returns:
        Array of ladder depths
    """
    # Convert depths to quantiles
    q_min = weibull_cdf(d_min, theta, p)
    q_max = weibull_cdf(d_max, theta, p)
    
    # Generate quantiles evenly spaced between q_min and q_max
    quantiles = np.linspace(q_min, q_max, num_rungs)
    
    # Convert quantiles back to depths
    depths = np.array([weibull_quantile(q, theta, p) for q in quantiles])
    
    print(f"Generated {num_rungs} ladder depths:")
    print(f"  Quantile range: {q_min:.3f} - {q_max:.3f}")
    print(f"  Depth range: {depths[0]:.3f}% - {depths[-1]:.3f}%")
    
    return depths


def calculate_ladder_depths(theta: float, p: float, num_rungs: int = None,
                          d_min: float = None, d_max: float = None,
                          method: str = 'quantile',
                          u_min: float = 0.75, p_min: float = 0.005,
                          current_price: float = None, profit_target_pct: float = 3.0) -> np.ndarray:
    """
    Calculate ladder depths using specified method.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        num_rungs: Number of rungs (if None, will use config default)
        d_min: Minimum depth (if None, will be calculated)
        d_max: Maximum depth (if None, will be calculated)
        method: Method to use ('quantile', 'expected_value', 'linear')
        u_min: Top rung quantile (for quantile method)
        p_min: Bottom rung tail probability (for quantile method)
        current_price: Current market price (for expected_value method)
        profit_target_pct: Target profit percentage (for expected_value method)
    
    Returns:
        Array of ladder depths
    """
    config = load_config()
    
    # Use provided num_rungs or default from config
    if num_rungs is None:
        num_rungs = config.get('num_rungs', 30)
    
    # Calculate depth range if not provided
    if d_min is None or d_max is None:
        d_min_calc, d_max_calc = calculate_depth_range(theta, p, u_min, p_min)
        d_min = d_min if d_min is not None else d_min_calc
        d_max = d_max if d_max is not None else d_max_calc
    
    print(f"Using depth range: {d_min:.3f}% - {d_max:.3f}%")
    
    # Generate depths using specified method
    if method == 'expected_value':
        if current_price is None:
            print("Warning: current_price required for expected_value method, falling back to quantile")
            method = 'quantile'
        else:
            depths = calculate_expected_value_depths(theta, p, d_min, d_max, num_rungs, 
                                                   current_price, profit_target_pct)
    elif method == 'quantile':
        depths = generate_ladder_depths(theta, p, d_min, d_max, num_rungs)
    elif method == 'linear':
        depths = np.linspace(d_min, d_max, num_rungs)
        print(f"Generated {num_rungs} linearly spaced depths")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile', 'expected_value', or 'linear'")
    
    return depths


def validate_ladder_depths(depths: np.ndarray) -> bool:
    """
    Validate that ladder depths are reasonable.
    
    Args:
        depths: Array of ladder depths
    
    Returns:
        True if depths are valid
    """
    # Check monotonicity
    if not np.all(np.diff(depths) > 0):
        raise ValueError("Ladder depths must be monotonically increasing")
    
    # Check reasonable range
    if depths[0] < 0.1:
        print(f"Warning: Top rung very shallow ({depths[0]:.3f}%)")
    
    if depths[-1] > 50.0:
        print(f"Warning: Bottom rung very deep ({depths[-1]:.3f}%)")
    
    # Check spacing
    min_spacing = np.min(np.diff(depths))
    if min_spacing < 0.05:
        print(f"Warning: Very tight spacing ({min_spacing:.3f}%)")
    
    print(f"Ladder validation passed:")
    print(f"  {len(depths)} rungs")
    print(f"  Spacing: {min_spacing:.3f}% - {np.max(np.diff(depths)):.3f}%")
    
    return True


if __name__ == "__main__":
    # Test with sample parameters
    theta = 2.5
    p = 1.2
    
    depths = calculate_ladder_depths(theta, p)
    validate_ladder_depths(depths)
    
    print("\nLadder depths:")
    for i, depth in enumerate(depths):
        print(f"  Rung {i+1}: {depth:.3f}%")
