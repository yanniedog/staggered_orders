"""
Unified analysis module consolidating scenario analysis, sensitivity analysis, and visualization.
Combines scenario_analyzer.py, sensitivity_analyzer.py, and scenario_visualizer.py into a single module.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
from scipy.optimize import minimize_scalar
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import seaborn as sns


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')


# ============================================================================
# SCENARIO ANALYSIS FUNCTIONS
# ============================================================================

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
        profit_pct: Target profit percentage per pair
        theta: Weibull scale parameter
        p: Weibull shape parameter
        risk_adjustment: Risk adjustment factor
    
    Returns:
        Tuple of (d_min, d_max)
    """
    # Conservative depth range for low profits
    if profit_pct <= 2.0:
        d_min = 0.3
        d_max = 3.0
    elif profit_pct <= 5.0:
        d_min = 0.5
        d_max = 5.0
    elif profit_pct <= 10.0:
        d_min = 1.0
        d_max = 8.0
    elif profit_pct <= 25.0:
        d_min = 1.5
        d_max = 12.0
    elif profit_pct <= 50.0:
        d_min = 2.0
        d_max = 15.0
    else:
        # Very aggressive for high profits
        d_min = 3.0
        d_max = 20.0
    
    # Apply risk adjustment
    d_max = d_max * risk_adjustment
    
    print(f"Depth range for {profit_pct:.1f}% profit: {d_min:.1f}% - {d_max:.1f}%")
    
    return d_min, d_max


def calculate_scenario_metrics(buy_depths: np.ndarray, sell_depths: np.ndarray,
                             allocations: np.ndarray, theta: float, p: float,
                             theta_sell: float, p_sell: float, current_price: float,
                             profit_targets: np.ndarray) -> Dict:
    """
    Calculate comprehensive metrics for a scenario.
    
    Args:
        buy_depths: Array of buy depths
        sell_depths: Array of sell depths
        allocations: Array of allocations
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        current_price: Current market price
        profit_targets: Array of profit targets
    
    Returns:
        Dictionary of scenario metrics
    """
    # Calculate touch probabilities
    buy_touch_probs = np.array([weibull_touch_probability(d, theta, p) for d in buy_depths])
    sell_touch_probs = np.array([weibull_touch_probability(d, theta_sell, p_sell) for d in sell_depths])
    
    # Calculate joint probabilities (simplified independence assumption)
    joint_probs = buy_touch_probs * sell_touch_probs
    
    # Calculate expected profits
    buy_prices = current_price * (1 - buy_depths / 100)
    sell_prices = current_price * (1 + sell_depths / 100)
    
    # Calculate quantities
    quantities = allocations / buy_prices
    
    # Calculate profit per pair
    profit_per_pair = (sell_prices - buy_prices) / buy_prices * 100
    
    # Calculate expected profit per dollar
    expected_profit_per_dollar = np.sum(joint_probs * profit_per_pair * quantities) / np.sum(allocations)
    
    # Calculate expected timeframe
    avg_joint_prob = np.mean(joint_probs)
    expected_timeframe_hours = 1.0 / avg_joint_prob if avg_joint_prob > 0 else np.inf
    
    # Calculate monthly metrics
    hours_per_month = 24 * 30
    expected_monthly_fills = hours_per_month / expected_timeframe_hours if expected_timeframe_hours < np.inf else 0
    expected_monthly_profit = expected_monthly_fills * np.sum(allocations) * expected_profit_per_dollar / 100
    
    # Calculate risk metrics
    max_single_loss = np.max(allocations)
    total_allocation = np.sum(allocations)
    
    # Calculate Sharpe ratio (simplified)
    profit_volatility = np.std(profit_per_pair) / np.mean(profit_per_pair) if np.mean(profit_per_pair) > 0 else 0
    sharpe_ratio = expected_profit_per_dollar / profit_volatility if profit_volatility > 0 else 0
    
    return {
        'expected_profit_per_dollar': expected_profit_per_dollar,
        'expected_timeframe_hours': expected_timeframe_hours,
        'expected_monthly_fills': expected_monthly_fills,
        'expected_monthly_profit': expected_monthly_profit,
        'max_single_loss': max_single_loss,
        'total_allocation': total_allocation,
        'sharpe_ratio': sharpe_ratio,
        'profit_volatility': profit_volatility,
        'joint_touch_prob': avg_joint_prob,
        'avg_buy_touch_prob': np.mean(buy_touch_probs),
        'avg_sell_touch_prob': np.mean(sell_touch_probs)
    }


def analyze_profit_scenarios(theta: float, p: float, theta_sell: float, p_sell: float,
                           budget: float, current_price: float, min_notional: float,
                           risk_adjustment_factor: float, total_cost_pct: float,
                           df: pd.DataFrame, max_analysis_hours: int, candle_interval: str) -> pd.DataFrame:
    """
    Analyze multiple profit scenarios to find optimal configuration.
    
    Args:
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        budget: Total budget in USD
        current_price: Current market price
        min_notional: Minimum order value
        risk_adjustment_factor: Risk adjustment factor
        total_cost_pct: Total trading costs
        df: Historical data DataFrame
        max_analysis_hours: Maximum analysis window
        candle_interval: Candle interval
    
    Returns:
        DataFrame with scenario analysis results
    """
    config = load_config()
    
    # Generate profit scenarios
    profit_scenarios = generate_profit_scenarios(
        config['min_profit_pct'], config['max_profit_pct'], config['num_profit_scenarios']
    )
    
    results = []
    
    print(f"Analyzing {len(profit_scenarios)} profit scenarios...")
    
    for i, profit_target in enumerate(profit_scenarios):
        print(f"\nScenario {i+1}/{len(profit_scenarios)}: {profit_target:.1f}% profit target")
        
        # Calculate optimal rungs for this profit target
        optimal_rungs = calculate_optimal_rungs(
            theta, p, budget, min_notional, current_price,
            config['min_rungs'], config['max_rungs']
        )
        
        # Calculate depth ranges
        d_min, d_max = calculate_depth_range_for_profit(profit_target, theta, p, risk_adjustment_factor)
        
        # Generate buy ladder depths
        buy_depths = np.linspace(d_min, d_max, optimal_rungs)
        
        # Generate sell depths (simplified)
        sell_d_min = d_min * 0.3
        sell_d_max = d_max * 0.8
        sell_depths = np.linspace(sell_d_min, sell_d_max, optimal_rungs)
        
        # Calculate allocations (equal for simplicity)
        allocations = np.full(optimal_rungs, budget / optimal_rungs)
        
        # Calculate profit targets
        profit_targets = np.full(optimal_rungs, profit_target)
        
        # Calculate metrics
        metrics = calculate_scenario_metrics(
            buy_depths, sell_depths, allocations, theta, p,
            theta_sell, p_sell, current_price, profit_targets
        )
        
        # Create result record
        result = {
            'profit_target_pct': profit_target,
            'num_rungs': optimal_rungs,
            'buy_depth_min': d_min,
            'buy_depth_max': d_max,
            'sell_depth_min': sell_d_min,
            'sell_depth_max': sell_d_max,
            'budget_usd': budget,
            'current_price': current_price,
            **metrics
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('expected_profit_per_dollar', ascending=False).reset_index(drop=True)
    
    print(f"\nScenario analysis complete. Top 3 scenarios:")
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        print(f"  {i+1}. {row['profit_target_pct']:.1f}% profit: "
              f"{row['expected_profit_per_dollar']:.4f} monthly return, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe")
    
    return df


def get_optimal_scenario(scenarios_df: pd.DataFrame, metric: str = 'expected_profit_per_dollar') -> Dict:
    """
    Get the optimal scenario based on specified metric.
    
    Args:
        scenarios_df: DataFrame with scenario results
        metric: Metric to optimize for
    
    Returns:
        Dictionary with optimal scenario details
    """
    optimal_idx = scenarios_df[metric].idxmax()
    optimal_scenario = scenarios_df.loc[optimal_idx].to_dict()
    
    print(f"Optimal scenario selected:")
    print(f"  Profit target: {optimal_scenario['profit_target_pct']:.1f}%")
    print(f"  Rungs: {optimal_scenario['num_rungs']}")
    print(f"  Expected monthly return: {optimal_scenario.get('expected_profit_per_dollar', 0):.4f}")
    print(f"  Expected timeframe: {optimal_scenario['expected_timeframe_hours']:.1f} hours")
    
    return optimal_scenario


# ============================================================================
# SENSITIVITY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_rung_sensitivity(theta: float, p: float, theta_sell: float, p_sell: float,
                           budget: float, current_price: float, 
                           profit_target: float = 100.0) -> pd.DataFrame:
    """
    Analyze sensitivity to number of rungs.
    
    Args:
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        budget: Total budget in USD
        current_price: Current market price
        profit_target: Target profit percentage
    
    Returns:
        DataFrame with rung sensitivity analysis
    """
    config = load_config()
    
    # Determine rung counts based on config
    if config.get('num_rungs') is not None:
        # Use fixed number of rungs
        rung_counts = [config['num_rungs']]
        print(f"Using fixed number of rungs: {config['num_rungs']}")
    else:
        # Use optimization range
        rung_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50]
        rung_counts = [r for r in rung_counts if config['min_rungs'] <= r <= config['max_rungs']]
        print(f"Using optimization range: {config['min_rungs']}-{config['max_rungs']} rungs")
    
    results = []
    
    print(f"Analyzing rung sensitivity for {profit_target}% profit target...")
    
    for num_rungs in rung_counts:
        print(f"  Testing {num_rungs} rungs...")
        
        # Calculate depth ranges for this profit target
        d_min, d_max = calculate_depth_range_for_profit(profit_target, theta, p)
        
        # Generate buy ladder depths
        buy_depths = np.linspace(d_min, d_max, num_rungs)
        
        # Generate sell depths
        sell_d_min = d_min * 0.3
        sell_d_max = d_max * 0.8
        sell_depths = np.linspace(sell_d_min, sell_d_max, num_rungs)
        
        # Calculate allocations (equal for simplicity)
        allocations = np.full(num_rungs, budget / num_rungs)
        
        # Calculate profit targets
        profit_targets = np.full(num_rungs, profit_target)
        
        # Calculate metrics
        metrics = calculate_scenario_metrics(
            buy_depths, sell_depths, allocations, theta, p,
            theta_sell, p_sell, current_price, profit_targets
        )
        
        # Calculate additional metrics specific to rung analysis
        min_allocation = np.min(allocations)
        max_allocation = np.max(allocations)
        allocation_ratio = max_allocation / min_allocation
        
        # Calculate depth range metrics
        depth_range = d_max - d_min
        avg_depth = (d_min + d_max) / 2
        
        # Capital efficiency calculation
        capital_efficiency = metrics['expected_monthly_profit'] / budget
        
        # Risk-adjusted return (proper Sharpe ratio)
        risk_adjusted_return = metrics['sharpe_ratio']
        
        # Additional validation metrics
        expected_monthly_fills = metrics.get('expected_monthly_fills', 0)
        profit_volatility = metrics.get('profit_volatility', 0)
        
        result = {
            'num_rungs': num_rungs,
            'profit_target_pct': profit_target,
            'buy_depth_min': d_min,
            'buy_depth_max': d_max,
            'depth_range': depth_range,
            'avg_depth': avg_depth,
            'sell_depth_min': sell_d_min,
            'sell_depth_max': sell_d_max,
            'expected_profit_per_dollar': metrics['expected_profit_per_dollar'],
            'expected_timeframe_hours': metrics['expected_timeframe_hours'],
            'expected_monthly_fills': expected_monthly_fills,
            'expected_monthly_profit': metrics.get('expected_monthly_profit', 0),
            'capital_efficiency': capital_efficiency,
            'risk_adjusted_return': risk_adjusted_return,
            'profit_volatility': profit_volatility,
            'allocation_ratio': allocation_ratio,
            'min_allocation': min_allocation,
            'max_allocation': max_allocation,
            'joint_touch_prob': metrics['joint_touch_prob'],
            'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
            'avg_sell_touch_prob': metrics['avg_sell_touch_prob']
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('capital_efficiency', ascending=False).reset_index(drop=True)
    
    print(f"\nRung sensitivity analysis complete.")
    print("Top configurations by capital efficiency:")
    for _, row in df.head(3).iterrows():
        print(f"  {row['num_rungs']} rungs: {row['capital_efficiency']:.4f} efficiency, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe")
    
    return df


def analyze_depth_sensitivity(theta: float, p: float, theta_sell: float, p_sell: float,
                            budget: float, current_price: float, num_rungs: int = 30,
                            profit_target: float = 100.0) -> pd.DataFrame:
    """
    Analyze sensitivity to depth range configurations.
    
    Args:
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        budget: Total budget in USD
        current_price: Current market price
        num_rungs: Number of rungs to test
        profit_target: Target profit percentage
    
    Returns:
        DataFrame with depth sensitivity analysis
    """
    print(f"Analyzing depth sensitivity for {num_rungs} rungs, {profit_target}% profit target...")
    
    # Test different depth ranges
    depth_ranges = [
        (0.5, 5.0),   # Conservative
        (1.0, 8.0),   # Moderate
        (1.5, 12.0),  # Aggressive
        (2.0, 15.0),  # Very aggressive
        (3.0, 20.0),  # Extreme
    ]
    
    results = []
    
    for d_min, d_max in depth_ranges:
        print(f"  Testing depth range: {d_min:.1f}% - {d_max:.1f}%")
        
        # Generate buy ladder depths
        buy_depths = np.linspace(d_min, d_max, num_rungs)
        
        # Generate sell depths
        sell_d_min = d_min * 0.3
        sell_d_max = d_max * 0.8
        sell_depths = np.linspace(sell_d_min, sell_d_max, num_rungs)
        
        # Calculate allocations (equal for simplicity)
        allocations = np.full(num_rungs, budget / num_rungs)
        
        # Calculate profit targets
        profit_targets = np.full(num_rungs, profit_target)
        
        # Calculate metrics
        metrics = calculate_scenario_metrics(
            buy_depths, sell_depths, allocations, theta, p,
            theta_sell, p_sell, current_price, profit_targets
        )
        
        # Calculate additional metrics
        depth_range = d_max - d_min
        avg_depth = (d_min + d_max) / 2
        
        # Capital efficiency calculation
        capital_efficiency = metrics['expected_monthly_profit'] / budget
        
        result = {
            'depth_range_name': f"{d_min:.1f}-{d_max:.1f}%",
            'buy_depth_min': d_min,
            'buy_depth_max': d_max,
            'depth_range': depth_range,
            'avg_depth': avg_depth,
            'sell_depth_min': sell_d_min,
            'sell_depth_max': sell_d_max,
            'expected_profit_per_dollar': metrics['expected_profit_per_dollar'],
            'expected_timeframe_hours': metrics['expected_timeframe_hours'],
            'expected_monthly_fills': metrics.get('expected_monthly_fills', 0),
            'expected_monthly_profit': metrics.get('expected_monthly_profit', 0),
            'capital_efficiency': capital_efficiency,
            'risk_adjusted_return': metrics['sharpe_ratio'],
            'profit_volatility': metrics.get('profit_volatility', 0),
            'joint_touch_prob': metrics['joint_touch_prob'],
            'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
            'avg_sell_touch_prob': metrics['avg_sell_touch_prob']
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('capital_efficiency', ascending=False).reset_index(drop=True)
    
    print(f"\nDepth sensitivity analysis complete.")
    print("Top configurations by capital efficiency:")
    for _, row in df.head(3).iterrows():
        print(f"  {row['depth_range_name']}: {row['capital_efficiency']:.4f} efficiency, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe")
    
    return df


def analyze_combined_sensitivity(theta: float, p: float, theta_sell: float, p_sell: float,
                               budget: float, current_price: float,
                               profit_target: float = 100.0) -> pd.DataFrame:
    """
    Analyze combined sensitivity to both rungs and depth ranges.
    
    Args:
        theta: Buy-side Weibull scale parameter
        p: Buy-side Weibull shape parameter
        theta_sell: Sell-side Weibull scale parameter
        p_sell: Sell-side Weibull shape parameter
        budget: Total budget in USD
        current_price: Current market price
        profit_target: Target profit percentage
    
    Returns:
        DataFrame with combined sensitivity analysis
    """
    print(f"Analyzing combined sensitivity for {profit_target}% profit target...")
    
    # Test combinations of rungs and depth ranges
    rung_counts = [15, 20, 25, 30, 35]
    depth_ranges = [
        (0.5, 5.0),   # Conservative
        (1.0, 8.0),   # Moderate
        (1.5, 12.0),  # Aggressive
        (2.0, 15.0),  # Very aggressive
    ]
    
    results = []
    
    for num_rungs in rung_counts:
        for d_min, d_max in depth_ranges:
            print(f"  Testing {num_rungs} rungs, depth {d_min:.1f}%-{d_max:.1f}%")
            
            # Generate buy ladder depths
            buy_depths = np.linspace(d_min, d_max, num_rungs)
            
            # Generate sell depths
            sell_d_min = d_min * 0.3
            sell_d_max = d_max * 0.8
            sell_depths = np.linspace(sell_d_min, sell_d_max, num_rungs)
            
            # Calculate allocations (equal for simplicity)
            allocations = np.full(num_rungs, budget / num_rungs)
            
            # Calculate profit targets
            profit_targets = np.full(num_rungs, profit_target)
            
            # Calculate metrics
            metrics = calculate_scenario_metrics(
                buy_depths, sell_depths, allocations, theta, p,
                theta_sell, p_sell, current_price, profit_targets
            )
            
            # Calculate additional metrics
            depth_range = d_max - d_min
            avg_depth = (d_min + d_max) / 2
            
            # Capital efficiency calculation
            capital_efficiency = metrics['expected_monthly_profit'] / budget
            
            result = {
                'num_rungs': num_rungs,
                'depth_range_name': f"{d_min:.1f}-{d_max:.1f}%",
                'buy_depth_min': d_min,
                'buy_depth_max': d_max,
                'depth_range': depth_range,
                'avg_depth': avg_depth,
                'sell_depth_min': sell_d_min,
                'sell_depth_max': sell_d_max,
                'expected_profit_per_dollar': metrics['expected_profit_per_dollar'],
                'expected_timeframe_hours': metrics['expected_timeframe_hours'],
                'expected_monthly_fills': metrics.get('expected_monthly_fills', 0),
                'expected_monthly_profit': metrics.get('expected_monthly_profit', 0),
                'capital_efficiency': capital_efficiency,
                'risk_adjusted_return': metrics['sharpe_ratio'],
                'profit_volatility': metrics.get('profit_volatility', 0),
                'joint_touch_prob': metrics['joint_touch_prob'],
                'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
                'avg_sell_touch_prob': metrics['avg_sell_touch_prob']
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('capital_efficiency', ascending=False).reset_index(drop=True)
    
    print(f"\nCombined sensitivity analysis complete.")
    print("Top configurations by capital efficiency:")
    for _, row in df.head(5).iterrows():
        print(f"  {row['num_rungs']} rungs, {row['depth_range_name']}: "
              f"{row['capital_efficiency']:.4f} efficiency, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe")
    
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_scenario_comparison_matrix(scenarios_df: pd.DataFrame) -> None:
    """
    Create scenario comparison matrix heatmap.
    
    Rows: number of rungs
    Columns: depth strategy (inferred from depth ranges)
    Cell color: expected monthly profit
    Cell annotation: fills per month
    """
    create_output_dir()
    
    # Create depth strategy categories based on depth ranges
    def categorize_strategy(row):
        min_depth = row['buy_depth_min']
        max_depth = row['buy_depth_max']
        
        if min_depth <= 1.0 and max_depth <= 5.0:
            return 'Conservative'
        elif min_depth <= 2.0 and max_depth <= 10.0:
            return 'Moderate'
        elif min_depth <= 3.0 and max_depth <= 15.0:
            return 'Aggressive'
        else:
            return 'Very Aggressive'
    
    scenarios_df['strategy'] = scenarios_df.apply(categorize_strategy, axis=1)
    
    # Create pivot table for heatmap
    pivot_table = scenarios_df.pivot_table(
        values='expected_monthly_profit',
        index='num_rungs',
        columns='strategy',
        aggfunc='mean'
    )
    
    # Create fills per month pivot for annotations
    fills_pivot = scenarios_df.pivot_table(
        values='expected_monthly_fills',
        index='num_rungs',
        columns='strategy',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap with annotations
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Expected Monthly Profit ($)'})
    
    # Add fills per month annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            if not pd.isna(pivot_table.iloc[i, j]):
                fills_value = fills_pivot.iloc[i, j]
                if not pd.isna(fills_value):
                    ax.text(j + 0.5, i + 0.7, f'({fills_value:.1f} fills)', 
                           ha='center', va='center', fontsize=8, color='blue')
    
    ax.set_xlabel('Depth Strategy')
    ax.set_ylabel('Number of Rungs')
    ax.set_title('Scenario Comparison Matrix\n(Cell color = Monthly Profit, Annotation = Fills per Month)')
    
    plt.tight_layout()
    plt.savefig('output/scenario_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Scenario comparison matrix saved to output/scenario_comparison_matrix.png")


def plot_risk_return_tradeoff(scenarios_df: pd.DataFrame) -> None:
    """
    Create risk-return tradeoff scatter plot.
    
    X: Risk (max drawdown or volatility)
    Y: Expected monthly return
    Size: number of rungs
    Color: depth strategy
    Add Pareto frontier line
    """
    create_output_dir()
    
    # Calculate risk metrics
    scenarios_df['max_drawdown'] = scenarios_df['max_single_loss'] / scenarios_df['total_allocation'] * 100
    scenarios_df['volatility'] = scenarios_df.get('profit_volatility', 0)
    
    # Use volatility as risk metric (or max drawdown if volatility is not available)
    risk_metric = 'volatility' if 'profit_volatility' in scenarios_df.columns else 'max_drawdown'
    
    # Create depth strategy categories
    def categorize_strategy(row):
        min_depth = row['buy_depth_min']
        max_depth = row['buy_depth_max']
        
        if min_depth <= 1.0 and max_depth <= 5.0:
            return 'Conservative'
        elif min_depth <= 2.0 and max_depth <= 10.0:
            return 'Moderate'
        elif min_depth <= 3.0 and max_depth <= 15.0:
            return 'Aggressive'
        else:
            return 'Very Aggressive'
    
    scenarios_df['strategy'] = scenarios_df.apply(categorize_strategy, axis=1)
    
    # Create color mapping
    strategy_colors = {'Conservative': 'green', 'Moderate': 'blue', 'Aggressive': 'orange', 'Very Aggressive': 'red'}
    colors = [strategy_colors.get(s, 'gray') for s in scenarios_df['strategy']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(scenarios_df[risk_metric], scenarios_df['expected_monthly_profit'],
                        s=scenarios_df['num_rungs']*10, alpha=0.7, c=colors,
                        edgecolors='black', linewidth=1)
    
    # Add Pareto frontier
    # Sort by risk and find Pareto optimal points
    sorted_scenarios = scenarios_df.sort_values(risk_metric)
    pareto_points = []
    
    max_return_so_far = -np.inf
    for _, row in sorted_scenarios.iterrows():
        if row['expected_monthly_profit'] > max_return_so_far:
            pareto_points.append((row[risk_metric], row['expected_monthly_profit']))
            max_return_so_far = row['expected_monthly_profit']
    
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel(f'Risk ({risk_metric.replace("_", " ").title()})')
    ax.set_ylabel('Expected Monthly Profit ($)')
    ax.set_title('Risk-Return Tradeoff\n(Size = Rungs, Color = Strategy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Risk-return tradeoff plot saved to output/risk_return_tradeoff.png")


def create_all_scenario_visualizations(scenarios_df: pd.DataFrame, rung_sensitivity_df: pd.DataFrame, 
                                     depth_sensitivity_df: pd.DataFrame, combined_sensitivity_df: pd.DataFrame) -> None:
    """
    Create all scenario analysis visualizations.
    
    Args:
        scenarios_df: Scenario analysis results
        rung_sensitivity_df: Rung sensitivity analysis results
        depth_sensitivity_df: Depth sensitivity analysis results
        combined_sensitivity_df: Combined sensitivity analysis results
    """
    print("Creating scenario analysis visualizations...")
    
    # Create individual plots
    plot_scenario_comparison_matrix(scenarios_df)
    plot_risk_return_tradeoff(scenarios_df)
    
    # Create sensitivity analysis plots
    if not rung_sensitivity_df.empty:
        plot_rung_sensitivity(rung_sensitivity_df)
    
    if not depth_sensitivity_df.empty:
        plot_depth_sensitivity(depth_sensitivity_df)
    
    if not combined_sensitivity_df.empty:
        plot_combined_sensitivity(combined_sensitivity_df)
    
    print("All scenario visualizations created successfully")


def plot_rung_sensitivity(rung_sensitivity_df: pd.DataFrame) -> None:
    """Create rung sensitivity visualization."""
    create_output_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Capital efficiency vs number of rungs
    ax1.plot(rung_sensitivity_df['num_rungs'], rung_sensitivity_df['capital_efficiency'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Rungs')
    ax1.set_ylabel('Capital Efficiency')
    ax1.set_title('Capital Efficiency vs Number of Rungs')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected timeframe vs number of rungs
    ax2.plot(rung_sensitivity_df['num_rungs'], rung_sensitivity_df['expected_timeframe_hours'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Rungs')
    ax2.set_ylabel('Expected Timeframe (hours)')
    ax2.set_title('Expected Timeframe vs Number of Rungs')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/rung_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Rung sensitivity plot saved to output/rung_sensitivity.png")


def plot_depth_sensitivity(depth_sensitivity_df: pd.DataFrame) -> None:
    """Create depth sensitivity visualization."""
    create_output_dir()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Capital efficiency vs depth range
    ax1.bar(range(len(depth_sensitivity_df)), depth_sensitivity_df['capital_efficiency'], 
            color='skyblue', edgecolor='black')
    ax1.set_xlabel('Depth Range')
    ax1.set_ylabel('Capital Efficiency')
    ax1.set_title('Capital Efficiency vs Depth Range')
    ax1.set_xticks(range(len(depth_sensitivity_df)))
    ax1.set_xticklabels(depth_sensitivity_df['depth_range_name'], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected timeframe vs depth range
    ax2.bar(range(len(depth_sensitivity_df)), depth_sensitivity_df['expected_timeframe_hours'], 
            color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Depth Range')
    ax2.set_ylabel('Expected Timeframe (hours)')
    ax2.set_title('Expected Timeframe vs Depth Range')
    ax2.set_xticks(range(len(depth_sensitivity_df)))
    ax2.set_xticklabels(depth_sensitivity_df['depth_range_name'], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/depth_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Depth sensitivity plot saved to output/depth_sensitivity.png")


def plot_combined_sensitivity(combined_sensitivity_df: pd.DataFrame) -> None:
    """Create combined sensitivity visualization."""
    create_output_dir()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap of capital efficiency
    pivot_table = combined_sensitivity_df.pivot_table(
        values='capital_efficiency',
        index='num_rungs',
        columns='depth_range_name',
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Capital Efficiency'})
    
    ax.set_xlabel('Depth Range')
    ax.set_ylabel('Number of Rungs')
    ax.set_title('Combined Sensitivity Analysis\n(Capital Efficiency Heatmap)')
    
    plt.tight_layout()
    plt.savefig('output/combined_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Combined sensitivity plot saved to output/combined_sensitivity.png")


if __name__ == "__main__":
    # Test with sample parameters
    theta = 2.5
    p = 1.2
    theta_sell = 2.5
    p_sell = 1.2
    budget = 10000.0
    current_price = 100.0
    min_notional = 10.0
    
    print("=== SCENARIO ANALYSIS TEST ===")
    scenarios_df = analyze_profit_scenarios(
        theta, p, theta_sell, p_sell, budget, current_price, min_notional,
        1.5, 0.25, None, 720, '1h'
    )
    
    print("\n=== SENSITIVITY ANALYSIS TEST ===")
    rung_sensitivity_df = analyze_rung_sensitivity(theta, p, theta_sell, p_sell, budget, current_price)
    depth_sensitivity_df = analyze_depth_sensitivity(theta, p, theta_sell, p_sell, budget, current_price)
    combined_sensitivity_df = analyze_combined_sensitivity(theta, p, theta_sell, p_sell, budget, current_price)
    
    print("\n=== VISUALIZATION TEST ===")
    create_all_scenario_visualizations(scenarios_df, rung_sensitivity_df, depth_sensitivity_df, combined_sensitivity_df)
