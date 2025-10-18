"""
Consolidated validation for staggered order analysis results.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def validate_analysis_results(scenarios_df: pd.DataFrame, optimal_scenario: Dict,
                             fit_metrics: Dict, fit_metrics_sell: Dict,
                             paired_orders_df: pd.DataFrame, 
                             logger=None) -> bool:
    """
    Consolidated validation of analysis results with critical checks only.
    
    Args:
        scenarios_df: Scenario analysis results
        optimal_scenario: Optimal scenario dictionary
        fit_metrics: Buy-side fit metrics
        fit_metrics_sell: Sell-side fit metrics
        paired_orders_df: Paired orders DataFrame
        logger: Optional logger instance for problem logging
    
    Returns:
        True if all validations pass, False otherwise
    """
    print("\n" + "="*50)
    print("VALIDATION CHECKS")
    print("="*50)
    
    validation_passed = True
    
    # 0. Check for NaN/Inf values in critical data
    print("\n0. NaN/Inf VALUE CHECKS")
    print("-" * 30)
    
    nan_checks = []
    if scenarios_df is not None and not scenarios_df.empty:
        numeric_cols = scenarios_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if scenarios_df[col].isna().any():
                nan_checks.append(f"Scenarios: {col} has NaN values")
            if np.isinf(scenarios_df[col]).any():
                nan_checks.append(f"Scenarios: {col} has Inf values")
    
    if not paired_orders_df.empty:
        numeric_cols = paired_orders_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if paired_orders_df[col].isna().any():
                nan_checks.append(f"Paired orders: {col} has NaN values")
            if np.isinf(paired_orders_df[col]).any():
                nan_checks.append(f"Paired orders: {col} has Inf values")
    
    if nan_checks:
        print("[FAIL] Found NaN/Inf values:")
        for check in nan_checks:
            print(f"  - {check}")
            if logger:
                logger.log_problem(check, "ERROR", {"validation_check": "NaN/Inf values"})
        validation_passed = False
    else:
        print("[PASS] No NaN/Inf values found")
    
    # 0b. Validate Weibull parameters
    print("\n0b. WEIBULL PARAMETER BOUNDS")
    print("-" * 30)
    
    theta = fit_metrics.get('theta', None)
    p = fit_metrics.get('p', None)
    theta_sell = fit_metrics_sell.get('theta', None)
    p_sell = fit_metrics_sell.get('p', None)
    
    weibull_checks = []
    
    # Validate buy-side parameters
    if theta is not None and p is not None:
        if theta <= 0 or theta > 50:
            weibull_checks.append(f"Buy-side theta out of bounds: {theta:.3f} (should be 0 < theta <= 50)")
        if p <= 0 or p > 5:
            weibull_checks.append(f"Buy-side p out of bounds: {p:.3f} (should be 0 < p <= 5)")
        if theta > 20:
            weibull_checks.append(f"Buy-side theta very high: {theta:.3f}% (may indicate unrealistic depths)")
        if p > 3:
            weibull_checks.append(f"Buy-side p very high: {p:.3f} (may indicate extreme tail behavior)")
    else:
        weibull_checks.append("Buy-side Weibull parameters not found in fit_metrics")
    
    # Validate sell-side parameters
    if theta_sell is not None and p_sell is not None:
        if theta_sell <= 0 or theta_sell > 50:
            weibull_checks.append(f"Sell-side theta out of bounds: {theta_sell:.3f} (should be 0 < theta <= 50)")
        if p_sell <= 0 or p_sell > 5:
            weibull_checks.append(f"Sell-side p out of bounds: {p_sell:.3f} (should be 0 < p <= 5)")
        if theta_sell > 20:
            weibull_checks.append(f"Sell-side theta very high: {theta_sell:.3f}% (may indicate unrealistic depths)")
        if p_sell > 3:
            weibull_checks.append(f"Sell-side p very high: {p_sell:.3f} (may indicate extreme tail behavior)")
    else:
        weibull_checks.append("Sell-side Weibull parameters not found in fit_metrics")
    
    if weibull_checks:
        print("[WARN] Weibull parameter validation issues:")
        for check in weibull_checks:
            print(f"  - {check}")
            if logger:
                logger.log_problem(check, "WARNING", {"validation_check": "Weibull parameters"})
    else:
        print("[PASS] All Weibull parameters within reasonable bounds")
    
    # 1. Validate probability bounds
    print("\n1. PROBABILITY BOUNDS")
    print("-" * 30)
    
    if 'avg_buy_touch_prob' in optimal_scenario:
        buy_prob = optimal_scenario['avg_buy_touch_prob']
        if not (0 <= buy_prob <= 1):
            print(f"[FAIL] Buy touch probability out of bounds: {buy_prob:.3f}")
            validation_passed = False
        else:
            print(f"[PASS] Buy touch probability: {buy_prob:.3f}")
    
    if 'avg_sell_touch_prob' in optimal_scenario:
        sell_prob = optimal_scenario['avg_sell_touch_prob']
        if not (0 <= sell_prob <= 1):
            print(f"[FAIL] Sell touch probability out of bounds: {sell_prob:.3f}")
            validation_passed = False
        else:
            print(f"[PASS] Sell touch probability: {sell_prob:.3f}")
    
    # 2. Validate Weibull fit quality
    print("\n2. WEIBULL FIT QUALITY")
    print("-" * 30)
    
    buy_quality = fit_metrics.get('fit_quality', 'unknown')
    sell_quality = fit_metrics_sell.get('fit_quality', 'unknown')
    
    if buy_quality in ['poor', 'unknown']:
        print(f"[FAIL] Buy-side fit quality: {buy_quality}")
        if logger:
            logger.log_problem(f"Buy-side fit quality is {buy_quality}", "ERROR", 
                             {"r_squared": fit_metrics.get('r_squared', 0)})
        validation_passed = False
    else:
        print(f"[PASS] Buy-side fit quality: {buy_quality}")
    
    if sell_quality in ['poor', 'unknown']:
        print(f"[FAIL] Sell-side fit quality: {sell_quality}")
        if logger:
            logger.log_problem(f"Sell-side fit quality is {sell_quality}", "ERROR",
                             {"r_squared": fit_metrics_sell.get('r_squared', 0)})
        validation_passed = False
    else:
        print(f"[PASS] Sell-side fit quality: {sell_quality}")
    
    # 3. Validate paired orders profitability
    print("\n3. PAIRED ORDERS PROFITABILITY")
    print("-" * 30)
    
    if not paired_orders_df.empty:
        unprofitable_pairs = paired_orders_df[paired_orders_df['profit_pct'] <= 0]
        if len(unprofitable_pairs) > 0:
            print(f"[FAIL] {len(unprofitable_pairs)} unprofitable pairs found")
            if logger:
                logger.log_problem(f"Found {len(unprofitable_pairs)} unprofitable pairs", "ERROR",
                                 {"unprofitable_count": len(unprofitable_pairs),
                                  "total_pairs": len(paired_orders_df)})
            validation_passed = False
        else:
            print(f"[PASS] All {len(paired_orders_df)} pairs are profitable")
    else:
        print(f"[FAIL] No paired orders generated")
        if logger:
            logger.log_problem("No paired orders generated", "CRITICAL")
        validation_passed = False
    
    # 3b. Validate time horizon reasonableness
    print("\n3b. TIME HORIZON VALIDATION")
    print("-" * 30)
    
    if optimal_scenario and 'expected_timeframe_hours' in optimal_scenario:
        timeframe = optimal_scenario['expected_timeframe_hours']
        if timeframe > 365 * 24:  # More than 1 year
            print(f"[WARN] Very long timeframe: {timeframe:.0f} hours ({timeframe/24/30:.1f} months)")
            print("       Strategy may not be practical for such long horizons")
        elif timeframe < 1:  # Less than 1 hour
            print(f"[WARN] Very short timeframe: {timeframe:.2f} hours")
            print("       Strategy may be too aggressive")
        else:
            print(f"[PASS] Timeframe is reasonable: {timeframe:.1f} hours ({timeframe/24:.1f} days)")
    else:
        print("[INFO] Timeframe data not available")
    
    # 4. Validate profit targets are reasonable
    print("\n4. PROFIT TARGET VALIDATION")
    print("-" * 30)
    
    if not paired_orders_df.empty:
        max_profit_pct = paired_orders_df['profit_pct'].max()
        min_profit_pct = paired_orders_df['profit_pct'].min()
        avg_profit_pct = paired_orders_df['profit_pct'].mean()
        
        # Check profit range
        if max_profit_pct > 50:  # More than 50% profit per pair is unrealistic
            print(f"[WARN] Maximum profit per pair is very high: {max_profit_pct:.1f}%")
            print("        This may indicate unrealistic profit targets")
            if logger:
                logger.log_problem(f"Unrealistic maximum profit: {max_profit_pct:.1f}%", "WARNING")
        else:
            print(f"[PASS] Maximum profit per pair: {max_profit_pct:.1f}%")
        
        if min_profit_pct < 0.5:  # Less than 0.5% profit per pair is too low
            print(f"[WARN] Minimum profit per pair is very low: {min_profit_pct:.1f}%")
            print("        This may not cover trading costs")
        else:
            print(f"[PASS] Minimum profit per pair: {min_profit_pct:.1f}%")
        
        print(f"[INFO] Average profit per pair: {avg_profit_pct:.1f}%")
        
        # Check for expected profit column
        if 'expected_profit' in paired_orders_df.columns:
            total_expected = paired_orders_df['expected_profit'].sum()
            total_potential = paired_orders_df['profit_usd'].sum()
            if total_expected > 0:
                efficiency = total_expected / total_potential
                print(f"[PASS] Expected profit efficiency: {efficiency:.3f}")
                
                if efficiency < 0.1:  # Less than 10% efficiency is poor
                    print(f"[WARN] Low profit efficiency: {efficiency:.3f}")
                    print("        Strategy may not be profitable after costs")
            else:
                print(f"[WARN] Expected profit is zero or negative")
        else:
            print(f"[FAIL] Expected profit column missing from paired orders")
            validation_passed = False
        
        # Check order sizes are reasonable
        if 'buy_notional' in paired_orders_df.columns:
            max_order_size = paired_orders_df['buy_notional'].max()
            min_order_size = paired_orders_df['buy_notional'].min()
            
            if max_order_size > 10000:  # More than $10k per order
                print(f"[WARN] Very large order size: ${max_order_size:.0f}")
                print("        May exceed practical trading limits")
            
            if min_order_size < 10:  # Less than $10 per order
                print(f"[WARN] Very small order size: ${min_order_size:.2f}")
                print("        May not meet exchange minimums")
            
            print(f"[INFO] Order size range: ${min_order_size:.2f} - ${max_order_size:.0f}")
    
    # 5. Validate scenario analysis consistency
    print("\n5. SCENARIO ANALYSIS CONSISTENCY")
    print("-" * 30)
    
    if scenarios_df is not None and not scenarios_df.empty:
        # Check for reasonable profit targets
        max_scenario_profit = scenarios_df['profit_target_pct'].max()
        if max_scenario_profit > 50:
            print(f"[WARN] Maximum scenario profit target is very high: {max_scenario_profit:.1f}%")
        else:
            print(f"[PASS] Maximum scenario profit target: {max_scenario_profit:.1f}%")
        
        # Check for positive expected returns
        negative_returns = scenarios_df[scenarios_df['expected_profit_per_dollar'] <= 0]
        if len(negative_returns) > 0:
            print(f"[WARN] {len(negative_returns)} scenarios have negative expected returns")
        else:
            print(f"[PASS] All scenarios have positive expected returns")
    else:
        print(f"[FAIL] No scenario analysis data available")
        validation_passed = False
    
    # 6. Validate data consistency
    print("\n6. DATA CONSISTENCY")
    print("-" * 30)
    
    if not paired_orders_df.empty and scenarios_df is not None:
        # Check that paired orders match scenario expectations
        paired_profit_range = (paired_orders_df['profit_pct'].min(), paired_orders_df['profit_pct'].max())
        scenario_profit_range = (scenarios_df['profit_target_pct'].min(), scenarios_df['profit_target_pct'].max())
        
        print(f"[INFO] Paired orders profit range: {paired_profit_range[0]:.1f}% - {paired_profit_range[1]:.1f}%")
        print(f"[INFO] Scenario profit range: {scenario_profit_range[0]:.1f}% - {scenario_profit_range[1]:.1f}%")
        
        # Check for reasonable overlap
        if paired_profit_range[1] < scenario_profit_range[0] or paired_profit_range[0] > scenario_profit_range[1]:
            print(f"[WARN] Paired orders profit range doesn't overlap with scenario range")
        else:
            print(f"[PASS] Paired orders profit range overlaps with scenario range")
    
    # 7. Validate market conditions and practical constraints
    print("\n7. MARKET CONDITIONS & PRACTICAL CONSTRAINTS")
    print("-" * 30)
    
    # Check if current price is reasonable for SOLUSDT
    if 'current_price' in optimal_scenario:
        current_price = optimal_scenario['current_price']
        if current_price < 1.0 or current_price > 1000:
            print(f"[WARN] Unusual current price: ${current_price:.2f}")
            print("        May indicate data quality issues")
        else:
            print(f"[PASS] Current price is reasonable: ${current_price:.2f}")
    
    # Check if budget allocation is reasonable
    if 'total_allocation' in optimal_scenario:
        total_allocation = optimal_scenario['total_allocation']
        budget = optimal_scenario.get('budget_usd', 10000)
        allocation_ratio = total_allocation / budget if budget > 0 else 0
        
        if allocation_ratio > 1.0:
            print(f"[WARN] Total allocation exceeds budget: {allocation_ratio:.2f}")
            print("        This should not be possible")
            validation_passed = False
        elif allocation_ratio < 0.1:
            print(f"[WARN] Very low budget utilization: {allocation_ratio:.2f}")
            print("        Strategy may be too conservative")
        else:
            print(f"[PASS] Budget utilization: {allocation_ratio:.2f}")
    
    # Check if touch probabilities are realistic
    if 'avg_buy_touch_prob' in optimal_scenario and 'avg_sell_touch_prob' in optimal_scenario:
        buy_prob = optimal_scenario['avg_buy_touch_prob']
        sell_prob = optimal_scenario['avg_sell_touch_prob']
        
        if buy_prob < 0.001:  # Less than 0.1% chance
            print(f"[WARN] Very low buy touch probability: {buy_prob:.4f}")
            print("        Orders may rarely fill")
        
        if sell_prob < 0.001:  # Less than 0.1% chance
            print(f"[WARN] Very low sell touch probability: {sell_prob:.4f}")
            print("        Sell orders may rarely fill")
        
        if buy_prob > 0.5:  # More than 50% chance
            print(f"[WARN] Very high buy touch probability: {buy_prob:.4f}")
            print("        May indicate shallow depths")
        
        print(f"[INFO] Buy touch probability: {buy_prob:.4f}")
        print(f"[INFO] Sell touch probability: {sell_prob:.4f}")
    
    # Check Sharpe ratio is realistic
    if 'sharpe_ratio' in optimal_scenario:
        sharpe = optimal_scenario['sharpe_ratio']
        if sharpe > 5.0:
            print(f"[WARN] Unrealistic Sharpe ratio: {sharpe:.2f}")
            print("        Values > 5 are extremely rare in practice")
        elif sharpe < 0:
            print(f"[WARN] Negative Sharpe ratio: {sharpe:.2f}")
            print("        Strategy may not be profitable")
        else:
            print(f"[PASS] Sharpe ratio is reasonable: {sharpe:.2f}")
    
    # Summary
    print("\n" + "="*50)
    if validation_passed:
        print("[SUCCESS] All critical validations passed")
    else:
        print("[FAILURE] Critical validation failures found")
    print("="*50)
    
    return validation_passed


def validate_weibull_fit(fit_metrics: Dict, min_quality: float = 0.90) -> bool:
    """
    Validate Weibull fit quality.
    
    Args:
        fit_metrics: Dictionary with fit metrics
        min_quality: Minimum R² threshold
    
    Returns:
        True if fit quality is acceptable
    """
    r_squared = fit_metrics.get('r_squared', 0)
    return r_squared >= min_quality


def validate_paired_orders(paired_orders_df: pd.DataFrame) -> bool:
    """
    Validate paired orders specifications.
    
    Args:
        paired_orders_df: DataFrame with paired order specifications
    
    Returns:
        True if all orders are valid
    """
    if paired_orders_df.empty:
        return False
    
    # Check profitability
    unprofitable = paired_orders_df['profit_pct'] <= 0
    if unprofitable.any():
        return False
    
    # Check quantity matching (should be close)
    qty_mismatch = ~np.isclose(paired_orders_df['buy_qty'], paired_orders_df['sell_qty'], rtol=1e-6)
    if qty_mismatch.any():
        return False
    
    return True

