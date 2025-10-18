"""
Sensitivity analysis for ladder configuration parameters.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import yaml
from scenario_analyzer import (
    weibull_touch_probability, calculate_scenario_metrics,
    calculate_depth_range_for_profit
)


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


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
        # profit_target is already per pair, not total - do NOT divide by num_rungs
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
        
        # CORRECTED: Capital efficiency calculation
        # Use expected monthly profit per dollar invested
        capital_efficiency = metrics['expected_monthly_profit'] / budget
        
        # CORRECTED: Risk-adjusted return (proper Sharpe ratio)
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
        num_rungs: Number of rungs to use
        profit_target: Target profit percentage
    
    Returns:
        DataFrame with depth sensitivity analysis
    """
    # Define different depth range strategies
    depth_strategies = {
        'Conservative': (0.5, 3.0),    # Shallow depths
        'Moderate': (1.0, 8.0),        # Medium depths
        'Aggressive': (2.0, 15.0),     # Deep depths
        'Very Aggressive': (3.0, 25.0) # Very deep depths
    }
    
    results = []
    
    print(f"Analyzing depth sensitivity for {num_rungs} rungs, {profit_target}% profit...")
    
    for strategy_name, (d_min, d_max) in depth_strategies.items():
        print(f"  Testing {strategy_name} strategy: {d_min}%-{d_max}%")
        
        # Generate buy ladder depths
        buy_depths = np.linspace(d_min, d_max, num_rungs)
        
        # Generate sell depths (match buy depths for equal depth coverage)
        sell_d_min = d_min
        sell_d_max = d_max
        sell_depths = np.linspace(sell_d_min, sell_d_max, num_rungs)
        
        # Calculate allocations (equal for simplicity)
        allocations = np.full(num_rungs, budget / num_rungs)
        
        # Calculate profit targets
        # profit_target is already per pair, not total - do NOT divide by num_rungs
        profit_targets = np.full(num_rungs, profit_target)
        
        # Calculate metrics
        metrics = calculate_scenario_metrics(
            buy_depths, sell_depths, allocations, theta, p,
            theta_sell, p_sell, current_price, profit_targets
        )
        
        # Calculate depth-specific metrics
        depth_range = d_max - d_min
        avg_depth = (d_min + d_max) / 2
        
        # Probability of any fill (at least one rung touches)
        any_fill_prob = 1 - np.prod(1 - np.array([weibull_touch_probability(d, theta, p) for d in buy_depths]))
        
        # Expected number of fills
        expected_fills = np.sum(np.array([weibull_touch_probability(d, theta, p) for d in buy_depths]))
        
        # CORRECTED: Use proper metrics from scenario_analyzer
        result = {
            'strategy': strategy_name,
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
            'expected_monthly_fills': metrics.get('expected_monthly_fills', 0),
            'expected_monthly_profit': metrics.get('expected_monthly_profit', 0),
            'capital_efficiency': metrics.get('expected_monthly_profit', 0) / budget,
            'risk_adjusted_return': metrics['sharpe_ratio'],
            'profit_volatility': metrics.get('profit_volatility', 0),
            'joint_touch_prob': metrics['joint_touch_prob'],
            'any_fill_probability': any_fill_prob,
            'expected_fills': expected_fills,
            'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
            'avg_sell_touch_prob': metrics['avg_sell_touch_prob']
        }
        
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('capital_efficiency', ascending=False).reset_index(drop=True)
    
    print(f"\nDepth sensitivity analysis complete.")
    print("Strategy rankings by capital efficiency:")
    for _, row in df.iterrows():
        print(f"  {row['strategy']}: {row['capital_efficiency']:.4f} efficiency, "
              f"{row['expected_timeframe_hours']:.1f}h timeframe, "
              f"{row['expected_fills']:.1f} expected fills")
    
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
    config = load_config()
    
    # Determine rung counts based on config
    if config.get('num_rungs') is not None:
        # Use fixed number of rungs
        rung_counts = [config['num_rungs']]
        print(f"Using fixed number of rungs: {config['num_rungs']}")
    else:
        # Use optimization range
        rung_counts = [15, 25, 35, 45]
        rung_counts = [r for r in rung_counts if config['min_rungs'] <= r <= config['max_rungs']]
        print(f"Using optimization range: {config['min_rungs']}-{config['max_rungs']} rungs")
    
    depth_strategies = {
        'Conservative': (0.5, 3.0),
        'Moderate': (1.0, 8.0),
        'Aggressive': (2.0, 15.0)
    }
    
    results = []
    
    print(f"Analyzing combined sensitivity for {profit_target}% profit...")
    
    for num_rungs in rung_counts:
        for strategy_name, (d_min, d_max) in depth_strategies.items():
            print(f"  Testing {num_rungs} rungs, {strategy_name} strategy...")
            
            # Generate depths
            buy_depths = np.linspace(d_min, d_max, num_rungs)
            sell_d_min = d_min * 0.3
            sell_d_max = d_max * 0.8
            sell_depths = np.linspace(sell_d_min, sell_d_max, num_rungs)
            
            # Calculate allocations
            allocations = np.full(num_rungs, budget / num_rungs)
            profit_targets = np.full(num_rungs, profit_target)
            
            # Calculate metrics
            metrics = calculate_scenario_metrics(
                buy_depths, sell_depths, allocations, theta, p,
                theta_sell, p_sell, current_price, profit_targets
            )
            
            # Calculate combined metrics
            depth_range = d_max - d_min
            avg_depth = (d_min + d_max) / 2
            
            # Risk metrics
            allocation_ratio = np.max(allocations) / np.min(allocations)
            max_single_loss = np.max(allocations)
            
            # CORRECTED: Efficiency metrics
            capital_efficiency = metrics.get('expected_monthly_profit', 0) / budget
            risk_efficiency = metrics['sharpe_ratio']
            
            # CORRECTED: Combined score with realistic weightings
            # Weight by: 40% monthly profit, 30% Sharpe ratio, 20% fills per month, 10% timeframe
            monthly_profit_score = min(1.0, capital_efficiency * 100)  # Normalize to 0-1
            sharpe_score = min(1.0, risk_efficiency / 3.0)  # Normalize Sharpe to 0-1 (3.0 is excellent)
            fills_score = min(1.0, metrics.get('expected_monthly_fills', 0) / 10.0)  # Normalize fills to 0-1
            timeframe_score = 1.0 / (1.0 + metrics['expected_timeframe_hours'] / (30 * 24))  # Normalize to 30 days
            
            combined_score = (0.4 * monthly_profit_score + 
                            0.3 * sharpe_score + 
                            0.2 * fills_score + 
                            0.1 * timeframe_score)
            
            result = {
                'num_rungs': num_rungs,
                'strategy': strategy_name,
                'profit_target_pct': profit_target,
                'buy_depth_min': d_min,
                'buy_depth_max': d_max,
                'depth_range': depth_range,
                'avg_depth': avg_depth,
                'expected_profit_per_dollar': metrics['expected_profit_per_dollar'],
                'expected_timeframe_hours': metrics['expected_timeframe_hours'],
                'expected_monthly_fills': metrics.get('expected_monthly_fills', 0),
                'expected_monthly_profit': metrics.get('expected_monthly_profit', 0),
                'capital_efficiency': capital_efficiency,
                'risk_efficiency': risk_efficiency,
                'profit_volatility': metrics.get('profit_volatility', 0),
                'combined_score': combined_score,
                'monthly_profit_score': monthly_profit_score,
                'sharpe_score': sharpe_score,
                'fills_score': fills_score,
                'timeframe_score': timeframe_score,
                'allocation_ratio': allocation_ratio,
                'max_single_loss': max_single_loss,
                'joint_touch_prob': metrics['joint_touch_prob'],
                'avg_buy_touch_prob': metrics['avg_buy_touch_prob'],
                'avg_sell_touch_prob': metrics['avg_sell_touch_prob']
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    print(f"\nCombined sensitivity analysis complete.")
    print("Top 5 configurations by combined score:")
    for _, row in df.head().iterrows():
        print(f"  {row['num_rungs']} rungs, {row['strategy']}: "
              f"score={row['combined_score']:.4f}, "
              f"efficiency={row['capital_efficiency']:.4f}, "
              f"timeframe={row['expected_timeframe_hours']:.1f}h")
    
    return df


def create_sensitivity_matrix(scenarios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a sensitivity matrix from scenario analysis results.
    
    Args:
        scenarios_df: DataFrame from scenario analysis
    
    Returns:
        Pivot table showing sensitivity matrix
    """
    # Create pivot table for visualization
    matrix = scenarios_df.pivot_table(
        values='expected_profit_per_dollar',
        index='num_rungs',
        columns='buy_depth_min',  # Use min depth as proxy for strategy
        aggfunc='mean'
    )
    
    return matrix


if __name__ == "__main__":
    # Test with sample parameters
    theta = 2.5
    p = 1.2
    theta_sell = 3.0
    p_sell = 1.1
    budget = 10000.0
    current_price = 181.5
    
    print("=== Rung Sensitivity Analysis ===")
    rung_sensitivity = analyze_rung_sensitivity(
        theta, p, theta_sell, p_sell, budget, current_price
    )
    
    print("\n=== Depth Sensitivity Analysis ===")
    depth_sensitivity = analyze_depth_sensitivity(
        theta, p, theta_sell, p_sell, budget, current_price
    )
    
    print("\n=== Combined Sensitivity Analysis ===")
    combined_sensitivity = analyze_combined_sensitivity(
        theta, p, theta_sell, p_sell, budget, current_price
    )
    
    print("\n=== Sensitivity Matrix ===")
    matrix = create_sensitivity_matrix(combined_sensitivity)
    print(matrix)
