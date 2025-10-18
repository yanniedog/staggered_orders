"""
Unified ladder depth calculation using Weibull distribution quantiles.
Consolidates buy and sell ladder depth calculations into a single bidirectional module.
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


def calculate_risk_adjusted_profit_targets(buy_depths: np.ndarray, 
                                         sell_depths: np.ndarray,
                                         target_total_profit: float,
                                         risk_adjustment_factor: float) -> np.ndarray:
    """
    Calculate risk-adjusted profit targets for each buy-sell pair.
    
    Args:
        buy_depths: Array of buy depths
        sell_depths: Array of sell depths
        target_total_profit: Total profit target (percentage)
        risk_adjustment_factor: Risk adjustment multiplier
    
    Returns:
        Array of profit targets for each pair
    """
    # Base profit per pair (equal distribution)
    base_profit = target_total_profit / len(buy_depths)
    
    # Risk adjustment: deeper buys get higher profit targets
    max_buy_depth = np.max(buy_depths)
    risk_factors = 1 + risk_adjustment_factor * (buy_depths / max_buy_depth)
    
    # Calculate individual profit targets
    profit_targets = base_profit * risk_factors
    
    # Normalize to ensure total equals target
    total_current = np.sum(profit_targets)
    profit_targets = profit_targets * (target_total_profit / total_current)
    
    print(f"Risk-adjusted profit targets:")
    print(f"  Base profit per pair: {base_profit:.2f}%")
    print(f"  Risk adjustment factor: {risk_adjustment_factor}")
    print(f"  Total target: {target_total_profit:.2f}%")
    print(f"  Actual total: {np.sum(profit_targets):.2f}%")
    print(f"  Profit range: {np.min(profit_targets):.2f}% - {np.max(profit_targets):.2f}%")
    
    return profit_targets


def calculate_probability_optimized_targets(buy_depths: np.ndarray, theta_sell: float, p_sell: float,
                                           current_price: float, mean_reversion_rate: float = 0.5,
                                           total_cost_pct: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate probability-optimized sell targets for each buy depth.
    
    For each buy depth, finds optimal sell depth that maximizes:
    P(buy_fill) × P(sell_fill|buy_filled) × profit_margin
    
    Args:
        buy_depths: Array of buy depths
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        current_price: Current market price
        mean_reversion_rate: Probability of price recovery after buy fill
        total_cost_pct: Total trading costs (buy + sell + slippage)
    
    Returns:
        Tuple of (sell_depths, profit_targets)
    """
    from scipy.optimize import minimize_scalar
    
    sell_depths = np.zeros_like(buy_depths)
    profit_targets = np.zeros_like(buy_depths)
    
    print(f"Calculating probability-optimized sell targets...")
    print(f"Mean reversion rate: {mean_reversion_rate:.3f}")
    print(f"Total cost: {total_cost_pct:.2f}%")
    
    for i, buy_depth in enumerate(buy_depths):
        buy_price = current_price * (1 - buy_depth / 100)
        
        # Calculate minimum profitable sell depth for this buy depth
        # sell_price >= buy_price * (1 + min_profit + costs)
        min_profit_pct = 1.0  # Minimum 1% profit per pair
        min_sell_price = buy_price * (1 + (min_profit_pct + total_cost_pct) / 100)
        min_sell_depth = (min_sell_price - current_price) / current_price * 100
        
        # Ensure minimum sell depth is positive
        min_sell_depth = max(0.1, min_sell_depth)
        
        # Calculate maximum sell depth (scale with buy depth, cap at 50%)
        max_sell_depth = min(50.0, buy_depth * 2.0)  # Scale with buy depth
        
        def expected_value(sell_depth):
            """Calculate expected value for given sell depth"""
            # Sell price
            sell_price = current_price * (1 + sell_depth / 100)
            
            # Profit percentage (before costs)
            profit_pct_before_costs = (sell_price - buy_price) / buy_price * 100
            
            # Profit percentage (after costs)
            profit_pct = profit_pct_before_costs - total_cost_pct
            
            # Skip if negative profit
            if profit_pct <= 0:
                return 0.0
            
            # Touch probabilities - NOTE: This is approximation within sell ladder calculation
            # In reality, buy-side would use buy-side theta/p params, but they're not available here
            # This is acceptable since we're just optimizing relative sell depths
            buy_touch_prob = np.exp(-(buy_depth / theta_sell) ** p_sell)  # Using sell params as proxy
            sell_touch_prob = np.exp(-(sell_depth / theta_sell) ** p_sell)
            
            # Joint probability = P(buy) × P(sell|buy)
            joint_prob = buy_touch_prob * sell_touch_prob * mean_reversion_rate
            
            # Expected value = probability × profit
            return joint_prob * profit_pct
        
        # Find optimal sell depth that maximizes expected value
        # Search range: minimum profitable depth to scaled maximum depth
        result = minimize_scalar(lambda x: -expected_value(x), 
                                bounds=(min_sell_depth, max_sell_depth), 
                                method='bounded')
        
        if result.success:
            optimal_sell_depth = result.x
            optimal_profit = -result.fun
        else:
            # Fallback to minimum profitable depth
            optimal_sell_depth = min_sell_depth
            optimal_profit = expected_value(optimal_sell_depth)
        
        # Ensure sell depths are monotonically increasing (deeper buys get deeper sells)
        # But allow more flexibility than the previous 1% minimum constraint
        if i > 0:
            min_sell_depth_monotonic = sell_depths[i-1] * 1.005  # 0.5% increase minimum
            optimal_sell_depth = max(optimal_sell_depth, min_sell_depth_monotonic)
        
        sell_depths[i] = optimal_sell_depth
        profit_targets[i] = optimal_profit
        
        print(f"  Rung {i+1}: Buy {buy_depth:.2f}% -> Sell {optimal_sell_depth:.2f}% "
              f"(profit: {optimal_profit:.2f}%, min_depth: {min_sell_depth:.2f}%)")
    
    print(f"Probability-optimized targets calculated:")
    print(f"  Sell depth range: {np.min(sell_depths):.2f}% - {np.max(sell_depths):.2f}%")
    print(f"  Profit target range: {np.min(profit_targets):.2f}% - {np.max(profit_targets):.2f}%")
    
    return sell_depths, profit_targets


def calculate_ladder_depths(theta: float, p: float, num_rungs: int = None,
                          d_min: float = None, d_max: float = None,
                          method: str = 'quantile',
                          u_min: float = 0.75, p_min: float = 0.005,
                          current_price: float = None, profit_target_pct: float = 3.0,
                          direction: str = 'buy') -> np.ndarray:
    """
    Calculate ladder depths using specified method for buy or sell direction.
    
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
        direction: 'buy' for buy-side depths, 'sell' for sell-side depths
    
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
    
    print(f"Using {direction} depth range: {d_min:.3f}% - {d_max:.3f}%")
    
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
        print(f"Generated {num_rungs} linearly spaced {direction} depths")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quantile', 'expected_value', or 'linear'")
    
    return depths


def calculate_sell_ladder_depths(theta_sell: float, p_sell: float,
                                buy_depths: np.ndarray,
                                target_total_profit: float,
                                risk_adjustment_factor: float,
                                d_min_sell: float = None,
                                d_max_sell: float = None,
                                method: str = 'risk_adjusted',
                                u_min_sell: float = 0.75,
                                p_min_sell: float = 0.005,
                                current_price: float = None,
                                mean_reversion_rate: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sell ladder depths using specified method.
    
    Args:
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        buy_depths: Array of buy depths
        target_total_profit: Total profit target (percentage)
        risk_adjustment_factor: Risk adjustment multiplier
        d_min_sell: Minimum sell depth (if None, will be calculated)
        d_max_sell: Maximum sell depth (if None, will be calculated)
        method: Method to use ('risk_adjusted', 'probability_optimized', 'quantile')
        u_min_sell: Top sell rung quantile (for quantile method)
        p_min_sell: Bottom sell rung tail probability (for quantile method)
        current_price: Current market price (for probability_optimized method)
        mean_reversion_rate: Mean reversion rate (for probability_optimized method)
    
    Returns:
        Tuple of (sell_depths, profit_targets)
    """
    num_rungs = len(buy_depths)  # Match number of buy rungs
    
    if method == 'probability_optimized':
        if current_price is None:
            print("Warning: current_price required for probability_optimized method, falling back to risk_adjusted")
            method = 'risk_adjusted'
        else:
            sell_depths, profit_targets = calculate_probability_optimized_targets(
                buy_depths, theta_sell, p_sell, current_price, mean_reversion_rate
            )
    elif method == 'quantile':
        # Use quantile-based calculation
        if d_min_sell is None or d_max_sell is None:
            d_min_sell_calc, d_max_sell_calc = calculate_depth_range(theta_sell, p_sell, u_min_sell, p_min_sell)
            d_min_sell = d_min_sell if d_min_sell is not None else d_min_sell_calc
            d_max_sell = d_max_sell if d_max_sell is not None else d_max_sell_calc
        print(f"Using quantile-based sell depth range: {d_min_sell:.3f}% - {d_max_sell:.3f}%")
        
        # Generate sell ladder depths using quantile spacing
        sell_depths = generate_ladder_depths(theta_sell, p_sell, d_min_sell, d_max_sell, num_rungs)
        
        # Calculate risk-adjusted profit targets
        profit_targets = calculate_risk_adjusted_profit_targets(
            buy_depths, sell_depths, target_total_profit, risk_adjustment_factor
        )
    elif method == 'risk_adjusted':
        # Use direct depth range - match buy side depth ranges
        if d_min_sell is None:
            d_min_sell = 0.5
        if d_max_sell is None:
            d_max_sell = 50.0  # Match buy side maximum depth
        print(f"Using direct sell depth range: {d_min_sell:.3f}% - {d_max_sell:.3f}%")
        
        # Generate evenly spaced sell depths
        sell_depths = np.linspace(d_min_sell, d_max_sell, num_rungs)
        
        # Calculate risk-adjusted profit targets
        profit_targets = calculate_risk_adjusted_profit_targets(
            buy_depths, sell_depths, target_total_profit, risk_adjustment_factor
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'risk_adjusted', 'probability_optimized', or 'quantile'")
    
    return sell_depths, profit_targets


def validate_ladder_depths(depths: np.ndarray, direction: str = 'buy') -> bool:
    """
    Validate that ladder depths are reasonable.
    
    Args:
        depths: Array of ladder depths
        direction: 'buy' or 'sell' for context
    
    Returns:
        True if depths are valid
    """
    # Check monotonicity
    if not np.all(np.diff(depths) > 0):
        raise ValueError(f"{direction.capitalize()} ladder depths must be monotonically increasing")
    
    # Check reasonable range
    if depths[0] < 0.1:
        print(f"Warning: Top {direction} rung very shallow ({depths[0]:.3f}%)")
    
    if depths[-1] > 50.0:
        print(f"Warning: Bottom {direction} rung very deep ({depths[-1]:.3f}%)")
    
    # Check spacing
    min_spacing = np.min(np.diff(depths))
    if min_spacing < 0.05:
        print(f"Warning: Very tight {direction} spacing ({min_spacing:.3f}%)")
    
    print(f"{direction.capitalize()} ladder validation passed:")
    print(f"  {len(depths)} rungs")
    print(f"  Spacing: {min_spacing:.3f}% - {np.max(np.diff(depths)):.3f}%")
    
    return True


def validate_sell_ladder_depths(sell_depths: np.ndarray, profit_targets: np.ndarray) -> bool:
    """
    Validate that sell ladder depths and profit targets are reasonable.
    
    Args:
        sell_depths: Array of sell depths
        profit_targets: Array of profit targets
    
    Returns:
        True if depths are valid
    """
    # Check monotonicity
    if not np.all(np.diff(sell_depths) > 0):
        raise ValueError("Sell ladder depths must be monotonically increasing")
    
    # Check reasonable range
    if sell_depths[0] < 0.1:
        print(f"Warning: Top sell rung very shallow ({sell_depths[0]:.3f}%)")
    
    if sell_depths[-1] > 50.0:
        print(f"Warning: Bottom sell rung very deep ({sell_depths[-1]:.3f}%)")
    
    # Check spacing
    min_spacing = np.min(np.diff(sell_depths))
    if min_spacing < 0.05:
        print(f"Warning: Very tight sell spacing ({min_spacing:.3f}%)")
    
    # Check profit targets
    if np.any(profit_targets <= 0):
        raise ValueError("All profit targets must be positive")
    
    if np.any(profit_targets > 50.0):
        print(f"Warning: Very high profit targets (max: {np.max(profit_targets):.2f}%)")
    
    print(f"Sell ladder validation passed:")
    print(f"  {len(sell_depths)} sell rungs")
    print(f"  Spacing: {min_spacing:.3f}% - {np.max(np.diff(sell_depths)):.3f}%")
    print(f"  Profit targets: {np.min(profit_targets):.2f}% - {np.max(profit_targets):.2f}%")
    
    return True


if __name__ == "__main__":
    # Test with sample parameters
    theta = 2.5
    p = 1.2
    
    # Test buy-side depths
    print("=== BUY-SIDE DEPTHS ===")
    depths_buy = calculate_ladder_depths(theta, p, direction='buy')
    validate_ladder_depths(depths_buy, direction='buy')
    
    print("\nBuy ladder depths:")
    for i, depth in enumerate(depths_buy):
        print(f"  Rung {i+1}: {depth:.3f}%")
    
    # Test sell-side depths
    print("\n=== SELL-SIDE DEPTHS ===")
    theta_sell = 2.5
    p_sell = 1.2
    target_total_profit = 100.0
    risk_adjustment_factor = 1.5
    
    sell_depths, profit_targets = calculate_sell_ladder_depths(
        theta_sell, p_sell, depths_buy, target_total_profit, risk_adjustment_factor
    )
    
    validate_sell_ladder_depths(sell_depths, profit_targets)
    
    print("\nSell ladder depths and profit targets:")
    for i, (buy_depth, sell_depth, profit) in enumerate(zip(depths_buy, sell_depths, profit_targets)):
        print(f"  Rung {i+1}: Buy {buy_depth:.3f}% -> Sell {sell_depth:.3f}% (Profit: {profit:.2f}%)")
