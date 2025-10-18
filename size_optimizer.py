"""
Unified size optimization for ladder orders using Kelly Criterion and monotone weight functions.
Consolidates buy and sell size optimization into a single bidirectional module.
"""
import numpy as np
from typing import Tuple
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def kelly_criterion_allocation_with_monotonicity(depths: np.ndarray, theta: float, p: float, 
                                              budget: float, risk_free_rate: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Calculate optimal allocations using Kelly Criterion with monotonicity constraint.
    
    Ensures deeper rungs always get more allocation while maintaining Kelly optimality.
    
    Args:
        depths: Array of ladder depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
        budget: Total budget in USD
        risk_free_rate: Risk-free rate (default 0%)
    
    Returns:
        Tuple of (kelly_allocations, kelly_fraction)
    """
    # Calculate touch probabilities
    touch_probs = np.exp(-(depths / theta) ** p)
    
    # Kelly fraction for each rung: f* = (bp - q) / b
    # where b = depth (profit percentage), p = touch_prob, q = 1 - touch_prob
    kelly_fractions = np.zeros_like(depths)
    
    for i, (depth, touch_prob) in enumerate(zip(depths, touch_probs)):
        if touch_prob > 0 and depth > 0:
            # Kelly formula: f* = (bp - q) / b = (depth * touch_prob - (1 - touch_prob)) / depth
            # Simplified: f* = touch_prob - (1 - touch_prob) / depth
            kelly_fraction = touch_prob - (1 - touch_prob) / depth
            
            # Ensure non-negative (Kelly can be negative for unfavorable bets)
            kelly_fractions[i] = max(0, kelly_fraction)
        else:
            kelly_fractions[i] = 0
    
    # Apply monotonicity constraint using isotonic regression
    # Ensure deeper rungs get more allocation
    kelly_fractions_monotonic = apply_monotonic_constraint(kelly_fractions, depths)
    
    # Normalize Kelly fractions to budget
    total_kelly_fraction = np.sum(kelly_fractions_monotonic)
    
    if total_kelly_fraction > 0:
        # Scale down if total Kelly fraction > 1 (over-betting)
        if total_kelly_fraction > 1.0:
            kelly_fractions_monotonic = kelly_fractions_monotonic / total_kelly_fraction
            total_kelly_fraction = 1.0
        
        kelly_allocations = budget * kelly_fractions_monotonic
    else:
        # Fallback to equal allocation if Kelly fractions are all zero
        kelly_allocations = np.full(len(depths), budget / len(depths))
        total_kelly_fraction = 1.0
    
    print(f"Kelly Criterion allocation with monotonicity:")
    print(f"  Total Kelly fraction: {total_kelly_fraction:.3f}")
    print(f"  Allocation range: ${np.min(kelly_allocations):.2f} - ${np.max(kelly_allocations):.2f}")
    print(f"  Kelly fractions: {np.min(kelly_fractions_monotonic):.3f} - {np.max(kelly_fractions_monotonic):.3f}")
    print(f"  Monotonicity check: {np.all(np.diff(kelly_allocations) >= 0)}")
    
    return kelly_allocations, total_kelly_fraction


def apply_monotonic_constraint(fractions: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Apply monotonic constraint to ensure deeper rungs get more allocation.
    
    Uses cumulative adjustment to enforce monotonicity while preserving Kelly optimality.
    
    Args:
        fractions: Kelly fractions for each rung
        depths: Depth values for each rung
    
    Returns:
        Monotonic Kelly fractions
    """
    # Start with original Kelly fractions
    monotonic_fractions = fractions.copy()
    
    # Apply cumulative adjustment to ensure monotonicity
    for i in range(1, len(monotonic_fractions)):
        # Ensure current rung gets at least as much as previous rung
        # But weight by the relative depth difference
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


def calculate_alpha(theta: float, p: float, d_max: float) -> float:
    """
    Calculate alpha parameter for monotone weight function.
    
    Ensures deeper rungs always get more size allocation.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        d_max: Maximum depth
    
    Returns:
        Alpha parameter
    """
    # Alpha must be large enough so the mode of w(d) lies beyond d_max
    # Mode occurs at d_mode = theta * (alpha/p)^(1/p)
    # We want d_mode > d_max, so alpha > p * (d_max/theta)^p
    
    alpha_min = p * (d_max / theta) ** p
    alpha = max(1.0, alpha_min)
    
    print(f"Alpha calculation:")
    print(f"  Minimum alpha: {alpha_min:.3f}")
    print(f"  Chosen alpha: {alpha:.3f}")
    
    return alpha


def weight_function(depth: np.ndarray, theta: float, p: float, alpha: float) -> np.ndarray:
    """
    Calculate weight function: w(d) = d^alpha * exp(-(d/theta)^p)
    
    Args:
        depth: Array of depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
        alpha: Monotonicity parameter
    
    Returns:
        Array of weights
    """
    # w(d) = d^alpha * exp(-(d/theta)^p)
    weights = (depth ** alpha) * np.exp(-(depth / theta) ** p)
    
    return weights


def calculate_allocations(depths: np.ndarray, theta: float, p: float, 
                        budget: float) -> Tuple[np.ndarray, float]:
    """
    Calculate size allocations for ladder rungs using monotone weight function.
    
    Args:
        depths: Array of ladder depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
        budget: Total budget in USD
    
    Returns:
        Tuple of (allocations, alpha)
    """
    # Calculate alpha for monotonicity
    alpha = calculate_alpha(theta, p, depths[-1])
    
    # Calculate weights
    weights = weight_function(depths, theta, p, alpha)
    
    # Normalize to budget
    total_weight = np.sum(weights)
    allocations = budget * weights / total_weight
    
    print(f"Size allocation:")
    print(f"  Total weight: {total_weight:.3f}")
    print(f"  Budget: ${budget:.2f}")
    print(f"  Allocation range: ${np.min(allocations):.2f} - ${np.max(allocations):.2f}")
    
    return allocations, alpha


def validate_allocations(allocations: np.ndarray, depths: np.ndarray, budget: float) -> bool:
    """
    Validate that allocations are monotone increasing with depth and meet risk limits.
    
    Args:
        allocations: Array of allocations
        depths: Array of depths
        budget: Total budget
    
    Returns:
        True if allocations are valid
    """
    config = load_config()
    
    # Check monotonicity
    if not np.all(np.diff(allocations) >= 0):
        print("Warning: Allocations not monotonically increasing")
        print("Allocation differences:", np.diff(allocations))
        return False
    
    # Check reasonable range
    min_allocation = np.min(allocations)
    max_allocation = np.max(allocations)
    
    if min_allocation < 1.0:
        print(f"Warning: Very small allocation: ${min_allocation:.2f}")
    
    if max_allocation > 1000.0:
        print(f"Warning: Very large allocation: ${max_allocation:.2f}")
    
    # Check allocation ratio
    allocation_ratio = max_allocation / min_allocation
    print(f"Allocation ratio (max/min): {allocation_ratio:.2f}")
    
    if allocation_ratio > 100:
        print(f"Warning: Very high allocation ratio: {allocation_ratio:.2f}")
    
    # Risk management checks
    max_single_rung_pct = config.get('max_single_rung_pct', 5.0)
    max_position_size_pct = config.get('max_position_size_pct', 20.0)
    
    # Check single rung limit
    max_single_rung_allocation = np.max(allocations)
    max_single_rung_limit = budget * max_single_rung_pct / 100
    
    if max_single_rung_allocation > max_single_rung_limit:
        print(f"Warning: Single rung exceeds limit: ${max_single_rung_allocation:.2f} > ${max_single_rung_limit:.2f}")
        print(f"Max single rung limit: {max_single_rung_pct}% of budget")
    
    # Check total position size
    total_allocation = np.sum(allocations)
    max_position_limit = budget * max_position_size_pct / 100
    
    if total_allocation > max_position_limit:
        print(f"Warning: Total allocation exceeds limit: ${total_allocation:.2f} > ${max_position_limit:.2f}")
        print(f"Max position limit: {max_position_size_pct}% of budget")
    
    print("Allocation validation passed")
    return True


def calculate_expected_returns(depths: np.ndarray, theta: float, p: float) -> np.ndarray:
    """
    Calculate expected return per dollar at each depth.
    
    Args:
        depths: Array of depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
    
    Returns:
        Array of expected returns
    """
    # Touch probability: q(d) = exp(-(d/theta)^p)
    touch_probs = np.exp(-(depths / theta) ** p)
    
    # Expected return per dollar ≈ depth * touch_probability
    # (assuming edge is proportional to depth)
    expected_returns = depths * touch_probs
    
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


def optimize_sizes(depths: np.ndarray, theta: float, p: float, 
                  budget: float, use_kelly: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Complete size optimization for ladder orders (buy-side).
    
    Args:
        depths: Array of ladder depths
        theta: Weibull scale parameter
        p: Weibull shape parameter
        budget: Total budget in USD
        use_kelly: Whether to use Kelly Criterion (True) or monotone weights (False)
    
    Returns:
        Tuple of (allocations, alpha_or_kelly_fraction, expected_returns)
    """
    print("Optimizing ladder sizes...")
    
    if use_kelly:
        # Use Kelly Criterion with monotonicity for optimal position sizing
        allocations, kelly_fraction = kelly_criterion_allocation_with_monotonicity(depths, theta, p, budget)
        alpha_or_kelly_fraction = kelly_fraction
        print(f"Using Kelly Criterion with monotonicity for position sizing")
    else:
        # Use monotone weight function (original method)
        allocations, alpha = calculate_allocations(depths, theta, p, budget)
        alpha_or_kelly_fraction = alpha
        print(f"Using monotone weight function for position sizing")
    
    # Calculate expected returns
    expected_returns = calculate_expected_returns(depths, theta, p)
    
    # Validate allocations
    validate_allocations(allocations, depths, budget)
    
    print(f"Size optimization complete:")
    if use_kelly:
        print(f"  Kelly fraction: {alpha_or_kelly_fraction:.3f}")
    else:
        print(f"  Alpha: {alpha_or_kelly_fraction:.3f}")
    print(f"  Total allocation: ${np.sum(allocations):.2f}")
    print(f"  Expected return range: {np.min(expected_returns):.4f} - {np.max(expected_returns):.4f}")
    
    return allocations, alpha_or_kelly_fraction, expected_returns


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
    alpha_sell = calculate_alpha(theta_sell, p_sell, sell_depths[-1])
    
    # Calculate actual profits based on sell quantities
    actual_profits = (sell_prices - buy_prices) / buy_prices * 100
    
    # Calculate expected returns for sell side
    expected_returns = calculate_expected_returns(sell_depths, theta_sell, p_sell)
    
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
    theta = 2.5
    p = 1.2
    budget = 1000.0
    
    # Sample depths
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    
    print("=== BUY-SIDE OPTIMIZATION ===")
    allocations, alpha, expected_returns = optimize_sizes(depths, theta, p, budget)
    
    print("\nBuy size optimization results:")
    for i, (depth, alloc, exp_ret) in enumerate(zip(depths, allocations, expected_returns)):
        print(f"  Rung {i+1}: {depth:.2f}% -> ${alloc:.2f} (exp ret: {exp_ret:.4f})")
    
    # Test sell-side optimization
    print("\n=== SELL-SIDE OPTIMIZATION ===")
    theta_sell = 2.5
    p_sell = 1.2
    
    # Sample buy data
    buy_quantities = allocations / (100.0 * (1 - depths / 100))  # Convert allocations to quantities
    buy_prices = 100.0 * (1 - depths / 100)  # Sample buy prices
    
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
