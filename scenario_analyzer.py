"""
Scenario analyzer for profit optimization and ladder configuration.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from scipy.optimize import minimize_scalar
import warnings


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def weibull_touch_probability(depth: float, theta: float, p: float) -> float:
    """
    Calculate touch probability using Weibull distribution.
    
    Args:
        depth: Depth percentage
        theta: Weibull scale parameter
        p: Weibull shape parameter
    
    Returns:
        Touch probability (0-1)
    """
    return np.exp(-(depth / theta) ** p)


def calculate_optimal_rungs(theta: float, p: float, budget: float, 
                          min_notional: float, current_price: float,
                          min_rungs: int = 10, max_rungs: int = 50) -> int:
    """
    Calculate optimal number of rungs based on expected return optimization.
    
    Args:
        theta: Weibull scale parameter
        p: Weibull shape parameter
        budget: Total budget in USD
        min_notional: Minimum order value
        current_price: Current market price
        min_rungs: Minimum number of rungs
        max_rungs: Maximum number of rungs
    
    Returns:
        Optimal number of rungs
    """
    def expected_return_for_rungs(num_rungs: int) -> float:
        """Calculate expected return for given number of rungs"""
        if num_rungs < min_rungs or num_rungs > max_rungs:
            return -np.inf
            
        # Calculate depth range (from 0.5% to 15% typically)
        d_min = 0.5
        d_max = 15.0
        depths = np.linspace(d_min, d_max, num_rungs)
        
        # Calculate touch probabilities
        touch_probs = np.array([weibull_touch_probability(d, theta, p) for d in depths])
        
        # Calculate expected returns per dollar invested (depth * touch_probability)
        expected_returns_per_dollar = depths * touch_probs
        
        # Check if all rungs meet minimum notional
        min_allocation = budget / num_rungs
        min_qty = min_allocation / current_price
        min_price = current_price * (1 - d_max / 100)
        min_order_value = min_price * min_qty
        
        if min_order_value < min_notional:
            return -np.inf
        
        # Calculate total expected return per dollar (this is what we want to maximize)
        # Use the mean expected return per dollar as the metric
        total_expected_return = np.mean(expected_returns_per_dollar)
        
        # Add penalty for too many rungs (diminishing returns)
        # More rungs = smaller allocations = lower efficiency
        if num_rungs > 20:
            penalty = (num_rungs - 20) * 0.005  # Small penalty for over-granularity
            total_expected_return -= penalty
            
        return total_expected_return
    
    # Test discrete values instead of continuous optimization
    best_rungs = min_rungs
    best_return = -np.inf
    
    print(f"Evaluating rung counts from {min_rungs} to {max_rungs}...")
    
    for num_rungs in range(min_rungs, max_rungs + 1):
        expected_return = expected_return_for_rungs(num_rungs)
        
        if expected_return > best_return:
            best_return = expected_return
            best_rungs = num_rungs
        
        # Print first few and best so far
        if num_rungs <= min_rungs + 5 or num_rungs == best_rungs:
            print(f"  {num_rungs} rungs: {expected_return:.6f} expected return")
    
    optimal_rungs = best_rungs
    print(f"Optimal rungs calculation:")
    print(f"  Evaluated range: {min_rungs}-{max_rungs}")
    print(f"  Optimal rungs: {optimal_rungs}")
    print(f"  Best expected return: {best_return:.6f}")
    
    return optimal_rungs


def generate_profit_scenarios(min_profit: float, max_profit: float, 
                            num_scenarios: int) -> List[float]:
    """
    Generate realistic profit scenarios across logarithmic scale.
    
    Args:
        min_profit: Minimum profit percentage (per pair)
        max_profit: Maximum profit percentage (per pair)
        num_scenarios: Number of scenarios to generate
    
    Returns:
        List of profit percentages (per pair, not total)
    """
    # Ensure profit targets are realistic (per pair, not total)
    min_profit = max(0.5, min_profit)  # At least 0.5% per pair
    max_profit = min(200.0, max_profit)  # At most 200% per pair (aggressive - no risk management)
    
    # Generate logarithmic spacing
    log_min = np.log10(min_profit)
    log_max = np.log10(max_profit)
    log_scenarios = np.linspace(log_min, log_max, num_scenarios)
    
    scenarios = [10 ** log_val for log_val in log_scenarios]
    
    print(f"Generated {len(scenarios)} realistic profit scenarios (per pair):")
    for i, scenario in enumerate(scenarios):
        print(f"  Scenario {i+1}: {scenario:.1f}% per pair")
    
    return scenarios


def calculate_depth_range_for_profit(profit_pct: float, theta: float, p: float,
                                   risk_adjustment: float = 1.5) -> Tuple[float, float]:
    """
    Calculate appropriate depth range for target profit scenario.
    
    Args:
        profit_pct: Target profit percentage
        theta: Weibull scale parameter
        p: Weibull shape parameter
        risk_adjustment: Risk adjustment factor
    
    Returns:
        Tuple of (min_depth, max_depth) percentages
    """
    # Calculate optimal depth range based on Weibull parameters and profit target
    # Use the mode of the Weibull distribution as a reference point
    weibull_mode = theta * ((p - 1) / p) ** (1 / p) if p > 1 else theta * 0.5
    
    # For higher profits, need deeper buy depths
    # Scale depth range based on profit target and Weibull characteristics
    
    if profit_pct <= 25:
        # Conservative: shallow depths around the mode
        d_min = max(1.0, weibull_mode * 0.3)  # Start deeper
        d_max = min(weibull_mode * 1.0, 8.0)
    elif profit_pct <= 100:
        # Moderate: depths around the mode to 2x mode
        d_min = max(1.5, weibull_mode * 0.5)
        d_max = min(weibull_mode * 2.0, 15.0)
    elif profit_pct <= 500:
        # Aggressive: deeper depths up to 3x mode
        d_min = max(2.0, weibull_mode * 0.8)
        d_max = min(weibull_mode * 3.5, 30.0)
    else:
        # Very aggressive: very deep depths up to 5x mode
        d_min = max(2.0, weibull_mode * 1.0)
        d_max = min(weibull_mode * 5.0, 50.0)  # Allow up to 50%
    
    # Force much deeper ranges for extreme scenarios
    if profit_pct > 100:
        # Force very deep ranges for high-profit scenarios
        d_min = max(2.0, d_min)  # Start at least 2% deep
        d_max = max(30.0, d_max)  # Go at least 30% deep
    
    # Ensure reasonable bounds regardless of Weibull parameters
    d_min = max(0.5, min(d_min, 10.0))  # Allow deeper minimums
    d_max = max(d_min + 1.0, min(d_max, 50.0))  # Allow up to 50%
    
    print(f"Depth range for {profit_pct:.1f}% profit (Weibull mode: {weibull_mode:.2f}%):")
    print(f"  Min depth: {d_min:.2f}%")
    print(f"  Max depth: {d_max:.2f}%")
    
    return d_min, d_max


def calculate_empirical_mean_reversion_rate(df: pd.DataFrame, max_analysis_hours: int, 
                                         candle_interval: str = "1h") -> float:
    """
    Calculate empirical mean reversion rate from historical data.
    
    For each buy fill event, measure how often price recovers to sell levels.
    
    Args:
        df: Historical price data
        max_analysis_hours: Analysis window in hours
        candle_interval: Candle interval
    
    Returns:
        Mean reversion rate (0-1)
    """
    from touch_analysis import compute_max_drops
    
    # Calculate interval duration in minutes
    interval_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440
    }.get(candle_interval, 60)
    
    # Calculate how many bars represent the analysis window
    horizon_bars = max(1, int(max_analysis_hours * 60 / interval_minutes))
    
    mean_reversion_events = 0
    total_drop_events = 0
    
    print(f"Calculating empirical mean reversion rate...")
    print(f"Analysis window: {horizon_bars} bars ({max_analysis_hours} hours)")
    
    for i in range(len(df) - horizon_bars * 2):  # Need extra bars for recovery analysis
        current_price = df.iloc[i]['close']
        
        # Look ahead for drop event
        future_bars = df.iloc[i+1:i+1+horizon_bars]
        if len(future_bars) == 0:
            continue
            
        min_price = future_bars['low'].min()
        max_drop = (current_price - min_price) / current_price * 100
        
        # Only consider significant drops (>1%)
        if max_drop > 1.0:
            total_drop_events += 1
            
            # Look further ahead for recovery
            recovery_bars = df.iloc[i+1+horizon_bars:i+1+horizon_bars*2]
            if len(recovery_bars) > 0:
                max_recovery_price = recovery_bars['high'].max()
                recovery_pct = (max_recovery_price - min_price) / min_price * 100
                
                # Consider it mean reversion if price recovers at least 50% of the drop
                if recovery_pct >= max_drop * 0.5:
                    mean_reversion_events += 1
    
    if total_drop_events == 0:
        print("Warning: No significant drop events found for mean reversion calculation")
        return 0.5  # Conservative default
    
    mean_reversion_rate = mean_reversion_events / total_drop_events
    
    print(f"Mean reversion analysis:")
    print(f"  Total drop events (>1%): {total_drop_events}")
    print(f"  Mean reversion events: {mean_reversion_events}")
    print(f"  Empirical mean reversion rate: {mean_reversion_rate:.3f}")
    
    return mean_reversion_rate


def calculate_scenario_metrics(buy_depths: np.ndarray, sell_depths: np.ndarray,
                             allocations: np.ndarray, theta: float, p: float,
                             theta_sell: float, p_sell: float, 
                             current_price: float, profit_targets: np.ndarray,
                             max_analysis_hours: float = 720.0,
                             total_cost_pct: float = 0.25,
                             mean_reversion_rate: float = None) -> Dict:
    """
    Calculate comprehensive metrics for a scenario with corrected calculations.
    
    Args:
        buy_depths: Buy ladder depths
        sell_depths: Sell ladder depths  
        allocations: Size allocations
        theta: Buy-side Weibull scale
        p: Buy-side Weibull shape
        theta_sell: Sell-side Weibull scale
        p_sell: Sell-side Weibull shape
        current_price: Current market price
        profit_targets: Target profit percentages (per pair, not total)
    
    Returns:
        Dictionary of scenario metrics
    """
    # Validate inputs
    assert np.all(buy_depths > 0), "Buy depths must be positive"
    assert np.all(sell_depths > 0), "Sell depths must be positive"
    assert np.all(allocations > 0), "Allocations must be positive"
    assert current_price > 0, "Current price must be positive"
    
    # Validate and handle mean reversion rate
    if mean_reversion_rate is None:
        print("[WARNING] Mean reversion rate not provided, using default 0.5")
        mean_reversion_rate = 0.5
    elif np.isnan(mean_reversion_rate) or np.isinf(mean_reversion_rate):
        print(f"[ERROR] Mean reversion rate is invalid: {mean_reversion_rate}")
        print("[ERROR] Using default 0.5")
        mean_reversion_rate = 0.5
    elif not (0.1 <= mean_reversion_rate <= 0.9):
        print(f"[WARNING] Mean reversion rate {mean_reversion_rate:.3f} outside reasonable range (0.1-0.9)")
        print("[WARNING] Clamping to reasonable range")
        mean_reversion_rate = max(0.1, min(0.9, mean_reversion_rate))
    
    # Calculate touch probabilities
    buy_touch_probs = np.array([weibull_touch_probability(d, theta, p) for d in buy_depths])
    sell_touch_probs = np.array([weibull_touch_probability(d, theta_sell, p_sell) for d in sell_depths])
    
    # Validate probabilities are in [0, 1] and handle edge cases
    if np.any(np.isnan(buy_touch_probs)) or np.any(np.isnan(sell_touch_probs)):
        raise ValueError("NaN values in touch probabilities")
    
    if np.any(np.isinf(buy_touch_probs)) or np.any(np.isinf(sell_touch_probs)):
        raise ValueError("Inf values in touch probabilities")
    
    # Clip probabilities to valid range
    buy_touch_probs = np.clip(buy_touch_probs, 1e-10, 1.0)
    sell_touch_probs = np.clip(sell_touch_probs, 1e-10, 1.0)
    
    # Validate probabilities are reasonable
    if np.any(buy_touch_probs < 1e-6):
        print(f"Warning: Very low buy touch probabilities detected (min: {np.min(buy_touch_probs):.2e})")
    
    if np.any(sell_touch_probs < 1e-6):
        print(f"Warning: Very low sell touch probabilities detected (min: {np.min(sell_touch_probs):.2e})")
    
    # Calculate expected values
    buy_prices = current_price * (1 - buy_depths / 100)
    sell_prices = current_price * (1 + sell_depths / 100)
    
    # Calculate profit per pair (before costs) - this is the actual profit percentage per trade
    profit_per_pair_before_costs = (sell_prices - buy_prices) / buy_prices * 100
    
    # Calculate profit per pair (after trading costs)
    profit_per_pair = profit_per_pair_before_costs - total_cost_pct
    
    # Validate profit targets are realistic (per pair, not total)
    max_profit_per_pair = np.max(profit_per_pair_before_costs)
    if max_profit_per_pair > 200:  # More than 200% profit per pair is unrealistic
        warnings.warn(f"Unrealistic profit target detected: {max_profit_per_pair:.1f}% per pair")
    
    # CORRECTED: Joint probability calculation
    # P(Buy AND Sell) = P(Buy) × P(Sell | Buy filled)
    # Where P(Sell | Buy filled) accounts for mean reversion probability
    
    # Step 1: Probability that buy order fills
    buy_fill_probs = buy_touch_probs
    
    # Step 2: Conditional probability that sell order fills GIVEN buy filled
    # This is NOT simply sell_touch_probs * mean_reversion_rate
    # Instead, it's the probability that price recovers enough to hit sell levels
    # from the buy price level, which depends on the price gap and market behavior
    
    # For each buy-sell pair, calculate the probability of recovery
    conditional_sell_probs = np.zeros_like(sell_touch_probs)
    for i in range(len(buy_depths)):
        # The sell depth from the buy price perspective
        # If we bought at depth d1, we need price to rise by d2 from that level
        # This is approximately: P(recovery to sell level | bought at buy level)
        # Simplified: use sell probability scaled by mean reversion rate
        conditional_sell_probs[i] = sell_touch_probs[i] * mean_reversion_rate
    
    # Step 3: Joint probability = P(Buy) × P(Sell | Buy)
    joint_probs = buy_fill_probs * conditional_sell_probs
    
    # Expected profit per dollar invested
    # Weight by allocation size and joint probability
    total_allocation = np.sum(allocations)
    expected_profit_per_dollar = np.sum(profit_per_pair * joint_probs * allocations) / total_allocation
    
    # CORRECTED: Expected timeframe calculation
    # Instead of "time to first fill", calculate "expected fills per month"
    # This is more intuitive and realistic
    
    # Expected number of buy fills per analysis window
    expected_buy_fills_per_window = np.sum(buy_touch_probs)
    
    # Convert to fills per month (assuming analysis window represents typical market behavior)
    # If max_analysis_hours = 720 (30 days), then fills per month = expected_buy_fills_per_window
    fills_per_month = expected_buy_fills_per_window * (30 * 24) / max_analysis_hours
    
    # Expected timeframe for first fill (in hours) - more realistic calculation
    if np.sum(buy_touch_probs) > 0:
        # Expected time to first fill using geometric distribution
        # E[T] = 1/p where p is the probability of success per trial
        # Each "trial" is one analysis window
        expected_timeframe_hours = max_analysis_hours / np.sum(buy_touch_probs)
    else:
        expected_timeframe_hours = np.inf
    
    # Validate timeframe is reasonable
    if expected_timeframe_hours > 365 * 24:  # More than 1 year
        warnings.warn(f"Unrealistic timeframe: {expected_timeframe_hours:.0f} hours ({expected_timeframe_hours/24:.0f} days)")
    
    # Calculate risk metrics
    max_single_loss = np.max(allocations)  # Worst case: lose largest position
    
    # CORRECTED: Proper Sharpe ratio calculation
    # Sharpe = (E[R] - Rf) / σ where Rf is risk-free rate, σ is volatility
    # For this analysis, we'll use a simplified version
    
    # Calculate volatility of returns (standard deviation of profit_per_pair)
    profit_volatility = np.std(profit_per_pair)
    
    # Risk-free rate (assume 0% for simplicity, or use treasury rate)
    risk_free_rate = 0.0
    
    # Proper Sharpe ratio
    if profit_volatility > 0:
        sharpe_ratio = (expected_profit_per_dollar - risk_free_rate) / profit_volatility
    else:
        sharpe_ratio = 0.0
    
    # Validate Sharpe ratio is reasonable
    if sharpe_ratio > 5.0:  # Sharpe > 5 is extremely rare
        warnings.warn(f"Unrealistic Sharpe ratio: {sharpe_ratio:.2f}")
    
    # Calculate cost impact analysis
    profitable_pairs = np.sum(profit_per_pair > 0)
    total_pairs = len(profit_per_pair)
    profitability_ratio = profitable_pairs / total_pairs if total_pairs > 0 else 0
    
    # Calculate expected profit after costs
    expected_profit_after_costs = np.sum(profit_per_pair * joint_probs * allocations) / total_allocation
    
    # Additional metrics for validation
    expected_monthly_profit = expected_profit_per_dollar * total_allocation * fills_per_month
    expected_monthly_fills = fills_per_month
    
    metrics = {
        'expected_profit_per_dollar': expected_profit_per_dollar,
        'expected_profit_after_costs': expected_profit_after_costs,
        'expected_timeframe_hours': expected_timeframe_hours,
        'expected_monthly_fills': expected_monthly_fills,
        'expected_monthly_profit': expected_monthly_profit,
        'total_allocation': total_allocation,
        'max_single_loss': max_single_loss,
        'sharpe_ratio': sharpe_ratio,
        'profit_volatility': profit_volatility,
        'avg_buy_touch_prob': np.mean(buy_touch_probs),
        'avg_sell_touch_prob': np.mean(sell_touch_probs),
        'joint_touch_prob': np.mean(joint_probs),
        'profit_range': (np.min(profit_per_pair), np.max(profit_per_pair)),
        'profit_range_before_costs': (np.min(profit_per_pair_before_costs), np.max(profit_per_pair_before_costs)),
        'depth_range': (np.min(buy_depths), np.max(buy_depths)),
        'profitable_pairs': profitable_pairs,
        'total_pairs': total_pairs,
        'profitability_ratio': profitability_ratio,
        'total_cost_pct': total_cost_pct,
        'mean_reversion_rate': mean_reversion_rate
    }
    
    return metrics


def analyze_profit_scenarios(theta: float, p: float, theta_sell: float, p_sell: float,
                           budget: float, current_price: float, 
                           min_notional: float, risk_adjustment: float = 1.5,
                           total_cost_pct: float = 0.25, df: pd.DataFrame = None,
                           max_analysis_hours: float = 720.0, candle_interval: str = "1h") -> pd.DataFrame:
    """
    Analyze multiple profit scenarios across different rung counts and return comprehensive comparison.
    
    Args:
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        budget: Total budget in USD
        current_price: Current market price
        min_notional: Minimum order value
        risk_adjustment: Risk adjustment factor
    
    Returns:
        DataFrame with scenario analysis results
    """
    config = load_config()
    
    # Calculate empirical mean reversion rate if data is provided
    mean_reversion_rate = None
    if df is not None:
        mean_reversion_rate = calculate_empirical_mean_reversion_rate(
            df, max_analysis_hours, candle_interval
        )
        print(f"[OK] Calculated empirical mean reversion rate: {mean_reversion_rate:.3f}")
        
        # Validate and clamp mean reversion rate to reasonable range
        if not (0.1 <= mean_reversion_rate <= 0.9):
            print(f"[WARNING] Mean reversion rate {mean_reversion_rate:.3f} is outside reasonable range (0.1-0.9)")
            print("[WARNING] Clamping to reasonable range")
            mean_reversion_rate = max(0.1, min(0.9, mean_reversion_rate))
    else:
        print("[WARNING] No historical data provided for mean reversion calculation")
        print("[WARNING] Using conservative default mean reversion rate of 0.5")
        print("[WARNING] This may not reflect actual market behavior - results should be interpreted cautiously")
        mean_reversion_rate = 0.5
    
    # Generate profit scenarios
    profit_scenarios = generate_profit_scenarios(
        config['min_profit_pct'], 
        config['max_profit_pct'], 
        config['num_profit_scenarios']
    )
    
    # Determine rung counts based on config
    if config.get('num_rungs') is not None:
        # Use fixed number of rungs
        rung_counts = [config['num_rungs']]
        print(f"Using fixed number of rungs: {config['num_rungs']}")
    else:
        # Use optimization range
        rung_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        # Filter rung counts to be within min/max bounds
        rung_counts = [r for r in rung_counts if config['min_rungs'] <= r <= config['max_rungs']]
        print(f"Using optimization range: {config['min_rungs']}-{config['max_rungs']} rungs")
    
    results = []
    
    for profit_target in profit_scenarios:
        print(f"\nAnalyzing {profit_target:.1f}% profit scenario...")
        
        # Test each rung count for this profit scenario
        for num_rungs in rung_counts:
            print(f"  Testing {num_rungs} rungs...")
            
            # Calculate optimal rungs for this scenario (but use the specified rung count)
            optimal_rungs = num_rungs
            
            # Calculate depth ranges
            d_min, d_max = calculate_depth_range_for_profit(profit_target, theta, p, risk_adjustment)
            
            # Generate buy ladder depths
            buy_depths = np.linspace(d_min, d_max, optimal_rungs)
            
            # Generate sell depths (match buy depths for equal depth coverage)
            sell_d_min = d_min  # Sell depths match buy depths
            sell_d_max = d_max
            sell_depths = np.linspace(sell_d_min, sell_d_max, optimal_rungs)
            
            # Calculate intelligent allocations using Kelly Criterion with monotonicity
            from size_optimizer_buy import optimize_sizes
            allocations, _, _ = optimize_sizes(buy_depths, theta, p, budget, use_kelly=True)
            
            # Calculate profit targets for each pair
            # profit_target is already per pair, not total
            profit_targets = np.full(optimal_rungs, profit_target)
            
            # Calculate scenario metrics
            metrics = calculate_scenario_metrics(
                buy_depths, sell_depths, allocations, theta, p,
                theta_sell, p_sell, current_price, profit_targets, 
                max_analysis_hours, total_cost_pct, mean_reversion_rate
            )
            
            # Store results
            result = {
                'profit_target_pct': profit_target,
                'num_rungs': optimal_rungs,
                'buy_depth_min': d_min,
                'buy_depth_max': d_max,
                'sell_depth_min': sell_d_min,
                'sell_depth_max': sell_d_max,
                'expected_profit_per_dollar': metrics['expected_profit_per_dollar'],
                'expected_profit_after_costs': metrics['expected_profit_after_costs'],
                'expected_timeframe_hours': metrics['expected_timeframe_hours'],
                'expected_monthly_profit': metrics['expected_monthly_profit'],
                'expected_monthly_fills': metrics['expected_monthly_fills'],
                'total_allocation': metrics['total_allocation'],
                'max_single_loss': metrics['max_single_loss'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
                'avg_sell_touch_prob': metrics['avg_sell_touch_prob'],
                'joint_touch_prob': metrics['joint_touch_prob'],
                'profit_range_min': metrics['profit_range'][0],
                'profit_range_max': metrics['profit_range'][1],
                'profit_range_before_costs_min': metrics['profit_range_before_costs'][0],
                'profit_range_before_costs_max': metrics['profit_range_before_costs'][1],
                'profitable_pairs': metrics['profitable_pairs'],
                'total_pairs': metrics['total_pairs'],
                'profitability_ratio': metrics['profitability_ratio'],
                'total_cost_pct': metrics['total_cost_pct'],
                'mean_reversion_rate': mean_reversion_rate
            }
            
            results.append(result)
    
    # Convert to DataFrame and calculate expected profit per dollar per month
    df = pd.DataFrame(results)
    
    # Calculate expected profit per dollar per month (capital efficiency × time efficiency)
    df['expected_profit_per_dollar_per_month'] = df['expected_profit_per_dollar'] * (30 * 24) / df['expected_timeframe_hours']
    
    # Rank by expected profit per dollar per month
    df = df.sort_values('expected_profit_per_dollar_per_month', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    
    print(f"\nScenario analysis complete. Generated {len(df)} scenarios.")
    print("Top 10 scenarios by expected profit per dollar per month:")
    for _, row in df.head(10).iterrows():
        print(f"  Rank {row['rank']}: {row['profit_target_pct']:.1f}% profit, "
              f"{row['num_rungs']} rungs, "
              f"{row['expected_profit_per_dollar_per_month']:.4f} monthly return, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe")
    
    return df


def get_optimal_scenario(scenarios_df: pd.DataFrame, 
                        criteria: str = 'expected_profit_per_dollar_per_month') -> Dict:
    """
    Get the optimal scenario based on specified criteria.
    
    Args:
        scenarios_df: DataFrame from analyze_profit_scenarios
        criteria: Ranking criteria ('expected_profit_per_dollar_per_month', 'expected_profit_per_dollar', 'sharpe_ratio', 'expected_timeframe_hours')
    
    Returns:
        Dictionary with optimal scenario details
    """
    if criteria == 'expected_timeframe_hours':
        # For timeframe, prefer shorter (ascending order)
        optimal_row = scenarios_df.loc[scenarios_df[criteria].idxmin()]
    else:
        # For profit and sharpe, prefer higher (descending order)
        optimal_row = scenarios_df.loc[scenarios_df[criteria].idxmax()]
    
    return optimal_row.to_dict()


if __name__ == "__main__":
    # Test with sample parameters
    theta = 2.5
    p = 1.2
    theta_sell = 3.0
    p_sell = 1.1
    budget = 10000.0
    current_price = 181.5
    min_notional = 10.0
    
    scenarios = analyze_profit_scenarios(
        theta, p, theta_sell, p_sell, budget, current_price, min_notional
    )
    
    print("\nFull scenario comparison:")
    print(scenarios[['rank', 'profit_target_pct', 'expected_profit_per_dollar', 
                    'expected_timeframe_hours', 'sharpe_ratio']].to_string(index=False))
