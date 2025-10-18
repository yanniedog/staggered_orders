"""
Sell ladder depth calculation using upward Weibull distribution quantiles.
Mirror of ladder_depths.py but for sell-side analysis with risk-adjusted profit targets.
"""
import numpy as np
from typing import Tuple
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def weibull_cdf_upward(depth: float, theta: float, p: float) -> float:
    """
    Weibull cumulative distribution function for upward movements.
    
    Args:
        depth: Depth value (percentage)
        theta: Scale parameter
        p: Shape parameter
    
    Returns:
        Cumulative probability
    """
    return 1 - np.exp(-(depth / theta) ** p)


def weibull_quantile_upward(quantile: float, theta: float, p: float) -> float:
    """
    Weibull quantile function (inverse CDF) for upward movements.
    
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


def calculate_sell_depth_range(theta_sell: float, p_sell: float, 
                             u_min: float, p_min: float) -> Tuple[float, float]:
    """
    Calculate minimum and maximum sell depths for the ladder.
    
    Args:
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        u_min: Top rung quantile (0-1)
        p_min: Bottom rung tail probability (0-1)
    
    Returns:
        Tuple of (d_min_sell, d_max_sell)
    """
    # Top rung: quantile of the fitted distribution
    d_min_sell = weibull_quantile_upward(u_min, theta_sell, p_sell)
    
    # Bottom rung: depth where probability = p_min
    # q(d_max) = p_min = exp(-(d_max/theta)^p)
    # d_max = theta * (-ln(p_min))^(1/p)
    d_max_sell = theta_sell * (-np.log(p_min)) ** (1 / p_sell)
    
    print(f"Sell depth range calculation:")
    print(f"  d_min_sell (top rung): {d_min_sell:.3f}% (quantile {u_min:.2f})")
    print(f"  d_max_sell (bottom rung): {d_max_sell:.3f}% (tail prob {p_min:.3f})")
    print(f"  Range: {d_max_sell - d_min_sell:.3f}%")
    
    return d_min_sell, d_max_sell


def generate_sell_ladder_depths(theta_sell: float, p_sell: float, 
                               d_min_sell: float, d_max_sell: float, 
                               num_rungs: int) -> np.ndarray:
    """
    Generate sell ladder depths using quantile spacing.
    
    Args:
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        d_min_sell: Minimum sell depth
        d_max_sell: Maximum sell depth
        num_rungs: Number of rungs
    
    Returns:
        Array of sell ladder depths
    """
    # Convert depths to quantiles
    q_min_sell = weibull_cdf_upward(d_min_sell, theta_sell, p_sell)
    q_max_sell = weibull_cdf_upward(d_max_sell, theta_sell, p_sell)
    
    # Generate quantiles evenly spaced between q_min and q_max
    quantiles = np.linspace(q_min_sell, q_max_sell, num_rungs)
    
    # Convert quantiles back to depths
    depths_sell = np.array([weibull_quantile_upward(q, theta_sell, p_sell) for q in quantiles])
    
    print(f"Generated {num_rungs} sell ladder depths:")
    print(f"  Quantile range: {q_min_sell:.3f} - {q_max_sell:.3f}")
    print(f"  Depth range: {depths_sell[0]:.3f}% - {depths_sell[-1]:.3f}%")
    
    return depths_sell


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
            d_min_sell_calc, d_max_sell_calc = calculate_sell_depth_range(theta_sell, p_sell, u_min_sell, p_min_sell)
            d_min_sell = d_min_sell if d_min_sell is not None else d_min_sell_calc
            d_max_sell = d_max_sell if d_max_sell is not None else d_max_sell_calc
        print(f"Using quantile-based sell depth range: {d_min_sell:.3f}% - {d_max_sell:.3f}%")
        
        # Generate sell ladder depths using quantile spacing
        sell_depths = generate_sell_ladder_depths(theta_sell, p_sell, d_min_sell, d_max_sell, num_rungs)
        
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
    theta_sell = 2.5
    p_sell = 1.2
    buy_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    target_total_profit = 100.0
    risk_adjustment_factor = 1.5
    
    sell_depths, profit_targets = calculate_sell_ladder_depths(
        theta_sell, p_sell, buy_depths, target_total_profit, risk_adjustment_factor
    )
    
    validate_sell_ladder_depths(sell_depths, profit_targets)
    
    print("\nSell ladder depths and profit targets:")
    for i, (buy_depth, sell_depth, profit) in enumerate(zip(buy_depths, sell_depths, profit_targets)):
        print(f"  Rung {i+1}: Buy {buy_depth:.3f}% -> Sell {sell_depth:.3f}% (Profit: {profit:.2f}%)")
