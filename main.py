"""
Simplified main orchestrator for the staggered order ladder system.
Consolidated imports and streamlined orchestration logic.
"""
from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Import consolidated modules
from config import load_config
from data_fetcher import fetch_solusdt_data, get_current_price
from touch_analysis import analyze_touch_probabilities, analyze_upward_touch_probabilities
from weibull_fit import fit_weibull_tail, validate_fit_quality
from ladder_depths import calculate_ladder_depths, validate_ladder_depths, calculate_sell_ladder_depths, validate_sell_ladder_depths
from size_optimizer import optimize_sizes, optimize_sell_sizes
from order_builder import build_paired_orders, export_paired_orders_csv, export_orders_csv
from output import create_all_visualizations, create_excel_workbook
from analysis import analyze_profit_scenarios, get_optimal_scenario, analyze_rung_sensitivity, analyze_depth_sensitivity, analyze_combined_sensitivity, create_all_scenario_visualizations
from validation import validate_analysis_results
from logger import LoggingContext


def cleanup_output_directory(logger=None):
    """Clean up all files in output directory at startup"""
    import shutil
    import glob
    
    output_dir = "output"
    
    if os.path.exists(output_dir):
        if logger:
            logger.logger.info("Cleaning up previous output files...")
        
        # Get all files in output directory
        files_to_delete = glob.glob(os.path.join(output_dir, "*"))
        
        for file_path in files_to_delete:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    if logger:
                        logger.logger.info(f"  Deleted: {os.path.basename(file_path)}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    if logger:
                        logger.logger.info(f"  Deleted directory: {os.path.basename(file_path)}")
            except Exception as e:
                if logger:
                    logger.logger.warning(f"  Warning: Could not delete {file_path}: {e}")
        
        if logger:
            logger.logger.info(f"Cleaned up {len(files_to_delete)} files/directories")
    else:
        if logger:
            logger.logger.info("Output directory does not exist, will be created")
    
    if logger:
        logger.logger.info("")


def run_analysis_step(step_name: str, func, logger=None, *args, **kwargs):
    """Run analysis step with error handling"""
    if logger:
        logger.logger.info(f"{step_name}...")
    try:
        result = func(*args, **kwargs)
        if logger:
            logger.logger.info(f"[OK] {step_name}")
        return result
    except Exception as e:
        if logger:
            logger.logger.error(f"[ERROR] {step_name}: {e}")
        raise


def main():
    """Main execution function"""
    symbol = "SOLUSDT"
    
    with LoggingContext(output_dir="output", symbol=symbol) as logger:
        try:
            logger.logger.info("=" * 60)
            logger.logger.info("    STAGGERED ORDER LADDER SYSTEM")
            logger.logger.info("=" * 60)
            cleanup_output_directory(logger)
            logger.log_analysis_step("Starting analysis", "SUCCESS")
            
            # Load configuration
            config = run_analysis_step("Loading configuration", load_config, logger)
            logger.log_configuration(config)
            
            # Create output directory
            if not os.path.exists('output'):
                os.makedirs('output')
                logger.logger.info("Created output directory")
                logger.log_analysis_step("Created output directory", "SUCCESS")
            
            logger.logger.info(f"Lookback: {config['lookback_days']} days")
            logger.logger.info(f"Max analysis window: {config.get('max_analysis_hours', 720)} hours")
            logger.log_data_summary({
                "lookback_days": config['lookback_days'],
                "max_analysis_hours": config.get('max_analysis_hours', 720),
                "timeframe": "1h"
            })
            
            # Fetch data
            df = run_analysis_step("Fetching 1h data", fetch_solusdt_data, logger)
            logger.logger.info(f"Loaded {len(df)} candles from {df['open_time'].min()} to {df['open_time'].max()}")
            
            if logger:
                logger.log_data_summary({
                    "timeframe": "1h",
                    "candles": len(df),
                    "start_date": str(df['open_time'].min()),
                    "end_date": str(df['open_time'].max())
                })
            
            # Get current price
            current_price = run_analysis_step("Getting current price", get_current_price, logger)
            logger.logger.info(f"Current {config['symbol']} price: ${current_price:.2f}")
            
            # Analyze touch probabilities
            depths, empirical_probs = run_analysis_step("Analyzing buy-side touch probabilities", 
                                                       analyze_touch_probabilities, logger, df, config['max_analysis_hours'], '1h')
            
            depths_upward, empirical_probs_upward = run_analysis_step("Analyzing sell-side touch probabilities", 
                                                                     analyze_upward_touch_probabilities, logger, df, config['max_analysis_hours'], '1h')
            
            # Fit Weibull distributions
            theta, p, fit_metrics = run_analysis_step("Fitting buy-side Weibull distribution", 
                                                     fit_weibull_tail, logger, depths, empirical_probs)
            
            theta_sell, p_sell, fit_metrics_sell = run_analysis_step("Fitting sell-side Weibull distribution", 
                                                                     fit_weibull_tail, logger, depths_upward, empirical_probs_upward)
            
            if logger:
                logger.log_weibull_fit("buy", fit_metrics)
                logger.log_weibull_fit("sell", fit_metrics_sell)
            
            # Validate fit quality
            min_quality = config.get('min_fit_quality', 0.90)
            if not validate_fit_quality(fit_metrics, min_quality):
                warnings.warn(f"Buy-side fit quality below threshold ({fit_metrics['r_squared']:.3f} < {min_quality:.2f})")
                if logger:
                    logger.log_problem(f"Buy-side fit quality below threshold ({fit_metrics['r_squared']:.3f} < {min_quality:.2f})", "WARNING")
            
            if not validate_fit_quality(fit_metrics_sell, min_quality):
                warnings.warn(f"Sell-side fit quality below threshold ({fit_metrics_sell['r_squared']:.3f} < {min_quality:.2f})")
                if logger:
                    logger.log_problem(f"Sell-side fit quality below threshold ({fit_metrics_sell['r_squared']:.3f} < {min_quality:.2f})", "WARNING")
            
            # Analyze profit scenarios
            scenarios_df = run_analysis_step("Analyzing profit scenarios", analyze_profit_scenarios, logger,
                                           theta, p, theta_sell, p_sell, config['budget_usd'], 
                                           current_price, config['min_notional'], config.get('risk_adjustment_factor', 1.5),
                                           config.get('total_cost_pct', 0.25), df, config.get('max_analysis_hours', 720), '1h')
            
            if logger:
                logger.log_scenario_results(scenarios_df.to_dict('records'))
                logger.log_analysis_step("Profit scenario analysis completed", "SUCCESS")
            
            # Get optimal scenario
            very_aggressive_scenarios = scenarios_df[scenarios_df['profit_target_pct'] >= 200.0]
            logger.logger.info(f"Very aggressive scenarios found: {len(very_aggressive_scenarios)}")
            
            if len(very_aggressive_scenarios) > 0:
                optimal_scenario = get_optimal_scenario(very_aggressive_scenarios, 'expected_profit_per_dollar')
                logger.logger.info(f"FORCED SELECTION: Using very aggressive scenario for deep ladder demonstration")
            else:
                # Use the highest profit scenario available
                optimal_scenario = get_optimal_scenario(scenarios_df, 'expected_profit_per_dollar')
                logger.logger.info(f"Using highest profit scenario available: {optimal_scenario['profit_target_pct']:.1f}%")
            
            if logger:
                logger.log_analysis_step(f"Optimal scenario selected: {optimal_scenario['profit_target_pct']:.1f}% profit", "SUCCESS")
            
            # Calculate ladder depths
            ladder_depths = run_analysis_step(f"Calculating ladder depths for {optimal_scenario['profit_target_pct']:.1f}% profit", 
                                            calculate_ladder_depths, logger, theta, p, 
                                            num_rungs=int(optimal_scenario['num_rungs']),
                                            d_min=optimal_scenario['buy_depth_min'],
                                            d_max=optimal_scenario['buy_depth_max'],
                                            method='expected_value',
                                            current_price=current_price,
                                            profit_target_pct=optimal_scenario['profit_target_pct'])
            validate_ladder_depths(ladder_depths)
            
            # Optimize buy sizes
            allocations, alpha, expected_returns = run_analysis_step("Optimizing buy size allocations", 
                                                                   optimize_sizes, logger, ladder_depths, theta, p, config['budget_usd'], use_kelly=True)
            
            # Calculate sell targets
            risk_adjustment_factor = config.get('risk_adjustment_factor', 1.5)
            sell_depths, profit_targets = run_analysis_step("Calculating quantile-based sell targets for steeper slope", 
                                                           calculate_sell_ladder_depths, logger, theta_sell, p_sell, ladder_depths, 
                                                           optimal_scenario['profit_target_pct'], risk_adjustment_factor,
                                                           d_min_sell=optimal_scenario['sell_depth_min'],
                                                           d_max_sell=optimal_scenario['sell_depth_max'],
                                                           method='quantile',
                                                           current_price=current_price,
                                                           mean_reversion_rate=0.5)
            validate_sell_ladder_depths(sell_depths, profit_targets)
            
            # Optimize sell sizes
            buy_quantities = allocations / (current_price * (1 - ladder_depths / 100))
            sell_prices = current_price * (1 + sell_depths / 100)
            
            sell_quantities, actual_profits, alpha_sell = run_analysis_step("Optimizing sell size allocations", 
                                                                            optimize_sell_sizes, logger, buy_quantities, 
                                                                            current_price * (1 - ladder_depths / 100), 
                                                                            sell_depths, sell_prices, profit_targets, 
                                                                            theta_sell, p_sell, independent_optimization=True)
            
            # Build paired orders
            paired_orders_df = run_analysis_step("Building paired buy-sell order specifications", 
                                                build_paired_orders, logger, ladder_depths, allocations, sell_depths, 
                                                sell_quantities, profit_targets, current_price, theta, p, 
                                                theta_sell, p_sell, config['max_analysis_hours'])
            
            if logger:
                logger.log_order_results({
                    "paired_orders": paired_orders_df.to_dict('records'),
                    "total_pairs": len(paired_orders_df),
                    "total_expected_profit": paired_orders_df['expected_profit'].sum() if 'expected_profit' in paired_orders_df.columns else 0
                })
                logger.log_analysis_step("Paired order generation completed", "SUCCESS")
            
            # Perform sensitivity analysis
            rung_sensitivity_df = run_analysis_step("Performing rung sensitivity analysis", 
                                                   analyze_rung_sensitivity, logger, theta, p, theta_sell, p_sell, 
                                                   config['budget_usd'], current_price, optimal_scenario['profit_target_pct'])
            
            depth_sensitivity_df = run_analysis_step("Performing depth sensitivity analysis", 
                                                    analyze_depth_sensitivity, logger, theta, p, theta_sell, p_sell, 
                                                    config['budget_usd'], current_price, int(optimal_scenario['num_rungs']), 
                                                    optimal_scenario['profit_target_pct'])
            
            combined_sensitivity_df = run_analysis_step("Performing combined sensitivity analysis", 
                                                       analyze_combined_sensitivity, logger, theta, p, theta_sell, p_sell, 
                                                       config['budget_usd'], current_price, optimal_scenario['profit_target_pct'])
            
            if logger:
                logger.log_sensitivity_results({
                    "rung_sensitivity": rung_sensitivity_df.to_dict('records'),
                    "depth_sensitivity": depth_sensitivity_df.to_dict('records'),
                    "combined_sensitivity": combined_sensitivity_df.to_dict('records')
                })
                logger.log_analysis_step("Sensitivity analysis completed", "SUCCESS")
            
            # Perform validation
            validation_passed = run_analysis_step("Performing comprehensive validation", 
                                                 validate_analysis_results, logger, scenarios_df, optimal_scenario, 
                                                 fit_metrics, fit_metrics_sell, paired_orders_df)
            
            if logger:
                logger.log_validation_results({
                    "validation_passed": validation_passed,
                    "scenarios_count": len(scenarios_df),
                    "optimal_profit_target": optimal_scenario['profit_target_pct'],
                    "paired_orders_count": len(paired_orders_df)
                })
                logger.log_analysis_step("Comprehensive validation completed", "SUCCESS")
            
            # Export results
            run_analysis_step("Exporting paired orders CSV", export_paired_orders_csv, logger, paired_orders_df, 'output/paired_orders.csv')
            
            # Export original CSV for backward compatibility
            run_analysis_step("Exporting orders CSV", export_orders_csv, logger,
                             paired_orders_df[['rung', 'buy_depth_pct', 'buy_price', 'buy_qty', 'buy_notional']].rename(columns={
                                 'buy_depth_pct': 'depth_pct', 'buy_price': 'limit_price', 'buy_qty': 'quantity', 'buy_notional': 'notional'
                             }))
            
            # Export analysis data
            scenarios_df.to_csv('output/scenario_comparison.csv', index=False)
            rung_sensitivity_df.to_csv('output/rung_sensitivity.csv', index=False)
            depth_sensitivity_df.to_csv('output/depth_sensitivity.csv', index=False)
            combined_sensitivity_df.to_csv('output/combined_sensitivity.csv', index=False)
            
            # Create Excel workbook
            run_analysis_step("Creating comprehensive Excel workbook", create_excel_workbook, logger,
                             paired_orders_df, ladder_depths, allocations, theta, p, fit_metrics, 
                             config['budget_usd'], current_price, theta_sell, p_sell, fit_metrics_sell, 
                             sell_depths, actual_profits, scenarios_df, rung_sensitivity_df, 
                             depth_sensitivity_df, combined_sensitivity_df)
            
            # Create visualizations
            run_analysis_step("Creating visualizations", create_all_visualizations, logger,
                             depths, empirical_probs, theta, p, fit_metrics,
                             ladder_depths, allocations, paired_orders_df,
                             depths_upward, empirical_probs_upward, theta_sell, p_sell, fit_metrics_sell,
                             sell_depths, actual_profits, scenarios_df, optimal_scenario)
            
            # Create scenario analysis visualizations
            run_analysis_step("Creating scenario analysis visualizations", create_all_scenario_visualizations, logger,
                             scenarios_df, rung_sensitivity_df, depth_sensitivity_df, combined_sensitivity_df)
            
            # Print summary
            logger.logger.info("\n" + "=" * 60)
            logger.logger.info("SUMMARY")
            logger.logger.info("=" * 60)
            logger.logger.info(f"Symbol: {config['symbol']} | Price: ${current_price:.2f}")
            logger.logger.info(f"Buy Weibull: θ={theta:.3f}, p={p:.3f}, R²={fit_metrics['r_squared']:.4f}")
            logger.logger.info(f"Sell Weibull: θ={theta_sell:.3f}, p={p_sell:.3f}, R²={fit_metrics_sell['r_squared']:.4f}")
            logger.logger.info(f"Ladder: {len(ladder_depths)} rungs | Budget: ${config['budget_usd']:.0f}")
            logger.logger.info(f"Pairs: {len(paired_orders_df)} | Avg Profit: {paired_orders_df['profit_pct'].mean():.2f}%")
            logger.logger.info(f"Outputs: output/*.csv, output/ladder_report.xlsx, output/*.png")
            logger.logger.info("=" * 60)
            logger.logger.info("ANALYSIS COMPLETE")
            
            if logger:
                logger.log_analysis_step("Analysis completed successfully", "SUCCESS")
                logger.log_performance_metrics({
                    "timeframe": "1h",
                    "optimal_profit_target": optimal_scenario['profit_target_pct'],
                    "best_expected_return": optimal_scenario['expected_profit_per_dollar']
                })
        
        except Exception as e:
            logger.logger.error(f"\nError: {e}")
            if logger:
                logger.log_error(e, "Main execution")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())