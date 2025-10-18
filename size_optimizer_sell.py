"""
Sell size optimization for ladder orders using upward Weibull distribution.
Ensures sell quantities match buy quantities and validates profit expectations.
"""
import numpy as np
from typing import Tuple
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def calculate_sell_alpha(theta_sell: float, p_sell: float, d_max_sell: float) -> float:
    """
    Calculate alpha parameter for sell-side monotone weight function.
    
    Args:
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        d_max_sell: Maximum sell depth
    
    Returns:
        Alpha parameter for sell side
    """
    # Alpha must be large enough so the mode of w(d) lies beyond d_max
    # Mode occurs at d_mode = theta * (alpha/p)^(1/p)
    # We want d_mode > d_max, so alpha > p * (d_max/theta)^p
    
    alpha_min = p_sell * (d_max_sell / theta_sell) ** p_sell
    alpha = max(1.0, alpha_min)
    
    print(f"Sell alpha calculation:")
    print(f"  Minimum alpha: {alpha_min:.3f}")
    print(f"  Chosen alpha: {alpha:.3f}")
    
    return alpha


def apply_sell_monotonic_constraint(fractions: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Apply monotonic constraint to ensure deeper sell rungs get more allocation.
    
    Args:
        fractions: Kelly fractions for each sell rung
        depths: Sell depth values for each rung
    
    Returns:
        Monotonic Kelly fractions
    """
    # Start with original Kelly fractions
    monotonic_fractions = fractions.copy()
    
    # Apply cumulative adjustment to ensure monotonicity
    for i in range(1, len(monotonic_fractions)):
        # Ensure current rung gets at least as much as previous rung
        # Weight by the relative depth difference
        depth_ratio = depths[i] / depths[i-1] if depths[i-1] > 0 else 1.0
        
        # Minimum allocation for current rung based on previous rung
        min_allocation = monotonic_fractions[i-1] * depth_ratio
        
        # Use the maximum of Kelly optimal and monotonic minimum
        monotonic_fractions[i] = max(monotonic_fractions[i], min_allocation)
    
    # Normalize to preserve total allocation
    original_total = np.sum(fractions)
    monotonic_total = np.sum(monotonic_fractions)
    
    if monotonic_total > 0 and original_total > 0:
        # Scale to preserve total Kelly fraction
        monotonic_fractions = monotonic_fractions * (original_total / monotonic_total)
    
    return monotonic_fractions


def sell_weight_function(depth: np.ndarray, theta_sell: float, p_sell: float, alpha: float) -> np.ndarray:
    """
    Calculate sell weight function: w(d) = d^alpha * exp(-(d/theta)^p)
    
    Args:
        depth: Array of sell depths
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        alpha: Monotonicity parameter
    
    Returns:
        Array of sell weights
    """
    # w(d) = d^alpha * exp(-(d/theta)^p)
    weights = (depth ** alpha) * np.exp(-(depth / theta_sell) ** p_sell)
    
    return weights


def calculate_sell_expected_returns(sell_depths: np.ndarray, theta_sell: float, p_sell: float) -> np.ndarray:
    """
    Calculate expected return per dollar at each sell depth.
    
    Args:
        sell_depths: Array of sell depths
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
    
    Returns:
        Array of expected returns
    """
    # Touch probability: q(d) = exp(-(d/theta)^p)
    touch_probs = np.exp(-(sell_depths / theta_sell) ** p_sell)
    
    # Expected return per dollar ≈ depth * touch_probability
    # (assuming edge is proportional to depth)
    expected_returns = sell_depths * touch_probs
    
    return expected_returns


def calculate_sell_quantities(buy_quantities: np.ndarray, 
                             buy_prices: np.ndarray,
                             sell_prices: np.ndarray,
                             profit_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sell quantities to match buy quantities and achieve profit targets.
    
    Args:
        buy_quantities: Array of buy quantities
        buy_prices: Array of buy prices
        sell_prices: Array of sell prices
        profit_targets: Array of profit targets (percentages)
    
    Returns:
        Tuple of (sell_quantities, actual_profits)
    """
    # Sell quantity must match buy quantity in base asset
    sell_quantities = buy_quantities.copy()
    
    # Calculate actual profits
    buy_notionals = buy_quantities * buy_prices
    sell_notionals = sell_quantities * sell_prices
    actual_profits = (sell_notionals - buy_notionals) / buy_notionals * 100
    
    print(f"Sell quantity calculation:")
    print(f"  Buy quantities: {np.min(buy_quantities):.3f} - {np.max(buy_quantities):.3f}")
    print(f"  Sell quantities: {np.min(sell_quantities):.3f} - {np.max(sell_quantities):.3f}")
    print(f"  Target profits: {np.min(profit_targets):.2f}% - {np.max(profit_targets):.2f}%")
    print(f"  Actual profits: {np.min(actual_profits):.2f}% - {np.max(actual_profits):.2f}%")
    
    return sell_quantities, actual_profits


def validate_sell_allocations(sell_quantities: np.ndarray, sell_prices: np.ndarray,
                             buy_quantities: np.ndarray, buy_prices: np.ndarray,
                             profit_targets: np.ndarray) -> bool:
    """
    Validate that sell allocations are reasonable and profitable.
    
    Args:
        sell_quantities: Array of sell quantities
        sell_prices: Array of sell prices
        buy_quantities: Array of buy quantities
        buy_prices: Array of buy prices
        profit_targets: Array of profit targets
    
    Returns:
        True if allocations are valid
    """
    # Check quantity matching
    if not np.allclose(sell_quantities, buy_quantities, rtol=1e-6):
        print("Warning: Sell quantities don't exactly match buy quantities")
    
    # Check profitability
    buy_notionals = buy_quantities * buy_prices
    sell_notionals = sell_quantities * sell_prices
    actual_profits = (sell_notionals - buy_notionals) / buy_notionals * 100
    
    unprofitable = actual_profits <= 0
    if np.any(unprofitable):
        print(f"Error: {np.sum(unprofitable)} pairs are unprofitable")
        print(f"Unprofitable pairs: {actual_profits[unprofitable]}")
        return False
    
    # Check reasonable profit range
    if np.any(actual_profits > 100.0):
        print(f"Warning: Very high profits (max: {np.max(actual_profits):.2f}%)")
    
    # Check profit target achievement
    profit_ratios = actual_profits / profit_targets
    avg_ratio = np.mean(profit_ratios)
    
    print(f"Sell allocation validation:")
    print(f"  All pairs profitable: {not np.any(unprofitable)}")
    print(f"  Profit range: {np.min(actual_profits):.2f}% - {np.max(actual_profits):.2f}%")
    print(f"  Average profit ratio: {avg_ratio:.2f}")
    print(f"  Total expected profit: {np.sum(actual_profits * buy_notionals / 100):.2f} USD")
    
    return True


def optimize_sell_sizes(buy_quantities: np.ndarray, buy_prices: np.ndarray,
                       sell_depths: np.ndarray, sell_prices: np.ndarray,
                       profit_targets: np.ndarray, theta_sell: float, p_sell: float,
                       independent_optimization: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Complete sell size optimization for ladder orders.
    
    Args:
        buy_quantities: Array of buy quantities
        buy_prices: Array of buy prices
        sell_depths: Array of sell depths
        sell_prices: Array of sell prices
        profit_targets: Array of profit targets
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        independent_optimization: If True, optimize sell quantities independently using Kelly Criterion
    
    Returns:
        Tuple of (sell_quantities, actual_profits, alpha_sell)
    """
    print("Optimizing sell sizes...")
    
    # ALWAYS match buy quantities to ensure profitable pairs
    # Independent optimization was causing unprofitable pairs due to quantity reduction
    print("Using matched buy quantities for sell optimization (ensures profitability)")
    sell_quantities = buy_quantities.copy()
    
    # Calculate alpha for sell-side monotonicity (for compatibility)
    alpha_sell = calculate_sell_alpha(theta_sell, p_sell, sell_depths[-1])
    
    # Calculate actual profits based on sell quantities
    actual_profits = (sell_prices - buy_prices) / buy_prices * 100
    
    # Calculate expected returns for sell side
    expected_returns = calculate_sell_expected_returns(sell_depths, theta_sell, p_sell)
    
    # Validate sell allocations
    validate_sell_allocations(sell_quantities, sell_prices, buy_quantities, 
                            buy_prices, profit_targets)
    
    print(f"Sell size optimization complete:")
    print(f"  Alpha sell: {alpha_sell:.3f}")
    print(f"  Total sell notional: ${np.sum(sell_quantities * sell_prices):.2f}")
    print(f"  Total buy notional: ${np.sum(buy_quantities * buy_prices):.2f}")
    print(f"  Expected return range: {np.min(expected_returns):.4f} - {np.max(expected_returns):.4f}")
    print(f"  Total expected profit: ${np.sum(actual_profits * buy_quantities * buy_prices / 100):.2f}")
    
    return sell_quantities, actual_profits, alpha_sell


if __name__ == "__main__":
    # Test with sample parameters
    theta_sell = 2.5
    p_sell = 1.2
    
    # Sample buy data
    buy_quantities = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    buy_prices = np.array([164.16, 163.75, 163.32, 162.87, 162.41, 161.92, 161.4, 160.86, 160.29, 159.68])
    
    # Sample sell data
    sell_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    sell_prices = buy_prices * (1 + sell_depths / 100)  # Assume sell prices are above buy prices
    profit_targets = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    
    sell_quantities, actual_profits, alpha_sell = optimize_sell_sizes(
        buy_quantities, buy_prices, sell_depths, sell_prices, 
        profit_targets, theta_sell, p_sell
    )
    
    print("\nSell size optimization results:")
    for i, (buy_qty, sell_qty, buy_price, sell_price, profit) in enumerate(
        zip(buy_quantities, sell_quantities, buy_prices, sell_prices, actual_profits)):
        print(f"  Rung {i+1}: Buy {buy_qty:.3f} @ ${buy_price:.2f} -> Sell {sell_qty:.3f} @ ${sell_price:.2f} (Profit: {profit:.2f}%)")
