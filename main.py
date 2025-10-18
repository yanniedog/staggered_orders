"""
Simplified main orchestrator for the staggered order ladder system.
Single timeframe (1h) analysis with paired orders and scenario analysis.
"""
import yaml
from typing import Dict
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Import our modules
from data_fetcher import fetch_solusdt_data, get_current_price
from touch_analysis import analyze_touch_probabilities
from touch_analysis_sell import analyze_upward_touch_probabilities
from weibull_fit import fit_weibull_tail, validate_fit_quality
from ladder_depths_buy import calculate_ladder_depths, validate_ladder_depths
from ladder_depths_sell import calculate_sell_ladder_depths, validate_sell_ladder_depths
from size_optimizer_buy import optimize_sizes
from size_optimizer_sell import optimize_sell_sizes
from order_builder import build_paired_orders, export_paired_orders_csv, export_orders_csv
from visualizer import create_all_visualizations
from excel_export import create_excel_workbook

# Import scenario analysis modules
from scenario_analyzer import analyze_profit_scenarios, get_optimal_scenario
from sensitivity_analyzer import analyze_rung_sensitivity, analyze_depth_sensitivity, analyze_combined_sensitivity
from scenario_visualizer import create_all_scenario_visualizations

# Import validation and logging
from validation import validate_analysis_results
from logger import LoggingContext


def load_config() -> dict:
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create the configuration file.")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        raise


def cleanup_output_directory():
    """Clean up all files in output directory at startup"""
    import shutil
    import glob
    
    output_dir = "output"
    
    if os.path.exists(output_dir):
        print("Cleaning up previous output files...")
        
        # Get all files in output directory
        files_to_delete = glob.glob(os.path.join(output_dir, "*"))
        
        for file_path in files_to_delete:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  Deleted: {os.path.basename(file_path)}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  Deleted directory: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  Warning: Could not delete {file_path}: {e}")
        
        print(f"Cleaned up {len(files_to_delete)} files/directories")
    else:
        print("Output directory does not exist, will be created")
    
    print()


def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("    STAGGERED ORDER LADDER SYSTEM")
    print("    Data-Driven SOLUSDT Wick Capture (1h)")
    print("=" * 60)
    print()


def print_summary(config: dict, theta: float, p: float, fit_metrics: dict,
                 depths: np.ndarray, allocations: np.ndarray, paired_orders_df: pd.DataFrame,
                 current_price: float, theta_sell: float, p_sell: float, 
                 fit_metrics_sell: dict, sell_depths: np.ndarray, actual_profits: np.ndarray):
    """Print comprehensive summary"""
    print("\n" + "=" * 60)
    print("                    SUMMARY")
    print("=" * 60)
    
    print(f"\nMARKET DATA:")
    print(f"  Symbol: {config['symbol']}")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  Lookback: {config['lookback_days']} days")
    print(f"  Timeframe: 1h")
    
    print(f"\nBUY-SIDE WEIBULL FIT:")
    print(f"  Theta (scale): {theta:.3f}")
    print(f"  P (shape): {p:.3f}")
    print(f"  R²: {fit_metrics['r_squared']:.4f}")
    print(f"  RMSE: {fit_metrics['rmse']:.6f}")
    print(f"  Data Points: {fit_metrics['n_points']}")
    
    print(f"\nSELL-SIDE WEIBULL FIT:")
    print(f"  Theta (scale): {theta_sell:.3f}")
    print(f"  P (shape): {p_sell:.3f}")
    print(f"  R²: {fit_metrics_sell['r_squared']:.4f}")
    print(f"  RMSE: {fit_metrics_sell['rmse']:.6f}")
    print(f"  Data Points: {fit_metrics_sell['n_points']}")
    
    print(f"\nLADDER CONFIGURATION:")
    print(f"  Number of Rungs: {len(depths)}")
    print(f"  Buy Depth Range: {depths[0]:.3f}% - {depths[-1]:.3f}%")
    print(f"  Sell Depth Range: {sell_depths[0]:.3f}% - {sell_depths[-1]:.3f}%")
    
    print(f"\nSIZE ALLOCATION:")
    print(f"  Total Budget: ${config['budget_usd']:.2f}")
    print(f"  Total Allocation: ${np.sum(allocations):.2f}")
    print(f"  Allocation Ratio: {np.max(allocations)/np.min(allocations):.2f}")
    
    print(f"\nPAIRED ORDERS:")
    print(f"  Valid Pairs: {len(paired_orders_df)}")
    print(f"  Total Expected Profit: ${paired_orders_df['profit_usd'].sum():.2f}")
    print(f"  Average Profit: {paired_orders_df['profit_pct'].mean():.2f}%")
    print(f"  Profit Range: {paired_orders_df['profit_pct'].min():.2f}% - {paired_orders_df['profit_pct'].max():.2f}%")
    
    print(f"\nOUTPUTS:")
    print(f"  Paired Orders CSV: output/paired_orders.csv")
    print(f"  Orders CSV: output/orders.csv")
    print(f"  Excel Report: output/ladder_report.xlsx")
    print(f"  Visualizations: output/*.png")
    
    print("\n" + "=" * 60)


def main():
    """Main execution function - single timeframe analysis"""
    symbol = "SOLUSDT"
    
    with LoggingContext(output_dir="output", symbol=symbol) as logger:
        try:
            print_banner()
            cleanup_output_directory()
            logger.log_analysis_step("Starting staggered order ladder analysis", "SUCCESS")
            
            # Load configuration
            print("Loading configuration...")
            config = load_config()
            logger.log_configuration(config)
            
            # Create output directory
            if not os.path.exists('output'):
                os.makedirs('output')
                print("Created output directory")
                logger.log_analysis_step("Created output directory", "SUCCESS")
            
            print(f"Lookback: {config['lookback_days']} days")
            print(f"Max analysis window: {config.get('max_analysis_hours', 720)} hours")
            logger.log_data_summary({
                "lookback_days": config['lookback_days'],
                "max_analysis_hours": config.get('max_analysis_hours', 720),
                "timeframe": "1h"
            })
            
            # Fetch data for 1h timeframe
            print(f"\nFetching 1h data...")
            try:
                df = fetch_solusdt_data()
                print(f"Loaded {len(df)} candles from {df['open_time'].min()} to {df['open_time'].max()}")
                
                if logger:
                    logger.log_data_summary({
                        "timeframe": "1h",
                        "candles": len(df),
                        "start_date": str(df['open_time'].min()),
                        "end_date": str(df['open_time'].max())
                    })
            except Exception as e:
                print(f"Error fetching data: {e}")
                if logger:
                    logger.log_error(e, "Data fetching")
                raise
            
            # Get current price
            try:
                current_price = get_current_price()
                print(f"Current {config['symbol']} price: ${current_price:.2f}")
            except Exception as e:
                print(f"Error getting current price: {e}")
                if logger:
                    logger.log_error(e, "Current price fetching")
                raise
            
            # Analyze touch probabilities (buy side)
            print(f"\nAnalyzing buy-side touch probabilities...")
            try:
                depths, empirical_probs = analyze_touch_probabilities(df, config['max_analysis_hours'], '1h')
                
                if logger:
                    logger.log_analysis_step("Buy-side touch probability analysis completed", "SUCCESS")
            except Exception as e:
                print(f"Error in buy-side touch analysis: {e}")
                if logger:
                    logger.log_error(e, "Buy-side touch analysis")
                raise
            
            # Analyze upward touch probabilities (sell side)
            print(f"\nAnalyzing sell-side touch probabilities...")
            try:
                depths_upward, empirical_probs_upward = analyze_upward_touch_probabilities(df, config['max_analysis_hours'], '1h')
                
                if logger:
                    logger.log_analysis_step("Sell-side touch probability analysis completed", "SUCCESS")
            except Exception as e:
                print(f"Error in sell-side touch analysis: {e}")
                if logger:
                    logger.log_error(e, "Sell-side touch analysis")
                raise
            
            # Fit Weibull distribution for buy side
            print("\nFitting Weibull tail distribution for buy side...")
            try:
                theta, p, fit_metrics = fit_weibull_tail(depths, empirical_probs)
                
                if logger:
                    logger.log_weibull_fit("buy", fit_metrics)
            except Exception as e:
                print(f"Error fitting buy-side Weibull: {e}")
                if logger:
                    logger.log_error(e, "Buy-side Weibull fitting")
                raise
            
            # Fit Weibull distribution for sell side
            print("\nFitting Weibull tail distribution for sell side...")
            try:
                theta_sell, p_sell, fit_metrics_sell = fit_weibull_tail(depths_upward, empirical_probs_upward)
                
                if logger:
                    logger.log_weibull_fit("sell", fit_metrics_sell)
            except Exception as e:
                print(f"Error fitting sell-side Weibull: {e}")
                if logger:
                    logger.log_error(e, "Sell-side Weibull fitting")
                raise
            
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
            print("\nAnalyzing profit scenarios...")
            scenarios_df = analyze_profit_scenarios(
                theta, p, theta_sell, p_sell, config['budget_usd'], 
                current_price, config['min_notional'], config.get('risk_adjustment_factor', 1.5),
                config.get('total_cost_pct', 0.25), df, config.get('max_analysis_hours', 720), '1h'
            )
            
            if logger:
                logger.log_scenario_results(scenarios_df.to_dict('records'))
                logger.log_analysis_step("Profit scenario analysis completed", "SUCCESS")
            
            # Get optimal scenario - FORCE VERY AGGRESSIVE FOR DEEP LADDERS
            # Find the scenario with highest profit target (200% profit = very aggressive)
            very_aggressive_scenarios = scenarios_df[scenarios_df['profit_target_pct'] >= 200.0]
            if len(very_aggressive_scenarios) > 0:
                # Select the very aggressive scenario with best expected return
                optimal_scenario = get_optimal_scenario(very_aggressive_scenarios, 'expected_profit_per_dollar_per_month')
                print(f"FORCED SELECTION: Using very aggressive scenario for deep ladder demonstration")
            else:
                # Fallback to normal optimal selection
                optimal_scenario = get_optimal_scenario(scenarios_df, 'expected_profit_per_dollar_per_month')
            
            if logger:
                logger.log_analysis_step(f"Optimal scenario selected: {optimal_scenario['profit_target_pct']:.1f}% profit", "SUCCESS")
            
            # Calculate ladder depths for optimal scenario with expected value positioning
            print(f"\nCalculating ladder depths for optimal scenario: {optimal_scenario['profit_target_pct']:.1f}% profit...")
            ladder_depths = calculate_ladder_depths(
                theta, p, 
                num_rungs=int(optimal_scenario['num_rungs']),
                d_min=optimal_scenario['buy_depth_min'],
                d_max=optimal_scenario['buy_depth_max'],
                method='expected_value',
                current_price=current_price,
                profit_target_pct=optimal_scenario['profit_target_pct']
            )
            validate_ladder_depths(ladder_depths)
            
            # Optimize buy sizes with Kelly + monotonicity
            print("\nOptimizing buy size allocations...")
            allocations, alpha, expected_returns = optimize_sizes(
                ladder_depths, theta, p, config['budget_usd'],
                use_kelly=True
            )
            
            # Calculate probability-optimized sell targets
            print("\nCalculating probability-optimized sell targets...")
            risk_adjustment_factor = config.get('risk_adjustment_factor', 1.5)
            
            sell_depths, profit_targets = calculate_sell_ladder_depths(
                theta_sell, p_sell, ladder_depths, optimal_scenario['profit_target_pct'], 
                risk_adjustment_factor,
                d_min_sell=optimal_scenario['sell_depth_min'],
                d_max_sell=optimal_scenario['sell_depth_max'],
                method='probability_optimized',
                current_price=current_price,
                mean_reversion_rate=0.5  # Could be calculated from historical data
            )
            validate_sell_ladder_depths(sell_depths, profit_targets)
            
            # Independent sell side optimization
            print("\nOptimizing sell size allocations independently...")
            buy_quantities = allocations / (current_price * (1 - ladder_depths / 100))
            sell_prices = current_price * (1 + sell_depths / 100)
            
            sell_quantities, actual_profits, alpha_sell = optimize_sell_sizes(
                buy_quantities, current_price * (1 - ladder_depths / 100), 
                sell_depths, sell_prices, profit_targets, theta_sell, p_sell,
                independent_optimization=True
            )
            
            # Build paired orders
            print("\nBuilding paired buy-sell order specifications...")
            paired_orders_df = build_paired_orders(
                ladder_depths, allocations, sell_depths, sell_quantities, 
                profit_targets, current_price, theta, p, theta_sell, p_sell,
                config['max_analysis_hours']
            )
            
            if logger:
                logger.log_order_results({
                    "paired_orders": paired_orders_df.to_dict('records'),
                    "total_pairs": len(paired_orders_df),
                    "total_expected_profit": paired_orders_df['expected_profit'].sum() if 'expected_profit' in paired_orders_df.columns else 0
                })
                logger.log_analysis_step("Paired order generation completed", "SUCCESS")
            
            # Perform sensitivity analysis
            print("\nPerforming sensitivity analysis...")
            rung_sensitivity_df = analyze_rung_sensitivity(
                theta, p, theta_sell, p_sell, config['budget_usd'], 
                current_price, optimal_scenario['profit_target_pct']
            )
            
            depth_sensitivity_df = analyze_depth_sensitivity(
                theta, p, theta_sell, p_sell, config['budget_usd'], 
                current_price, int(optimal_scenario['num_rungs']), optimal_scenario['profit_target_pct']
            )
            
            combined_sensitivity_df = analyze_combined_sensitivity(
                theta, p, theta_sell, p_sell, config['budget_usd'], 
                current_price, optimal_scenario['profit_target_pct']
            )
            
            if logger:
                logger.log_sensitivity_results({
                    "rung_sensitivity": rung_sensitivity_df.to_dict('records'),
                    "depth_sensitivity": depth_sensitivity_df.to_dict('records'),
                    "combined_sensitivity": combined_sensitivity_df.to_dict('records')
                })
                logger.log_analysis_step("Sensitivity analysis completed", "SUCCESS")
            
            # Perform comprehensive validation
            print("\nPerforming comprehensive validation...")
            validation_passed = validate_analysis_results(
                scenarios_df, optimal_scenario, fit_metrics, fit_metrics_sell, paired_orders_df, logger
            )
            
            if logger:
                logger.log_validation_results({
                    "validation_passed": validation_passed,
                    "scenarios_count": len(scenarios_df),
                    "optimal_profit_target": optimal_scenario['profit_target_pct'],
                    "paired_orders_count": len(paired_orders_df)
                })
                logger.log_analysis_step("Comprehensive validation completed", "SUCCESS")
            
            # Export results
            print(f"\nExporting results...")
            
            # Export paired orders CSV
            export_paired_orders_csv(paired_orders_df, 'output/paired_orders.csv')
            
            # Export original CSV for backward compatibility
            export_orders_csv(paired_orders_df[['rung', 'buy_depth_pct', 'buy_price', 'buy_qty', 'buy_notional']].rename(columns={
                'buy_depth_pct': 'depth_pct', 'buy_price': 'limit_price', 'buy_qty': 'quantity', 'buy_notional': 'notional'
            }))
            
            # Export scenario analysis data
            scenarios_df.to_csv('output/scenario_comparison.csv', index=False)
            
            # Export sensitivity analysis
            rung_sensitivity_df.to_csv('output/rung_sensitivity.csv', index=False)
            depth_sensitivity_df.to_csv('output/depth_sensitivity.csv', index=False)
            combined_sensitivity_df.to_csv('output/combined_sensitivity.csv', index=False)
            
            # Create comprehensive Excel workbook
            print(f"\nCreating comprehensive Excel workbook...")
            create_excel_workbook(
                paired_orders_df, ladder_depths, allocations, 
                theta, p, fit_metrics, 
                config['budget_usd'], current_price,
                theta_sell, p_sell, fit_metrics_sell, 
                sell_depths, actual_profits,
                scenarios_df, rung_sensitivity_df, 
                depth_sensitivity_df, combined_sensitivity_df
            )
            
            # Create visualizations
            print(f"\nCreating visualizations...")
            create_all_visualizations(
                depths, empirical_probs, theta, p, fit_metrics,
                ladder_depths, allocations, paired_orders_df,
                depths_upward, empirical_probs_upward, theta_sell, p_sell, fit_metrics_sell,
                sell_depths, actual_profits, scenarios_df, optimal_scenario
            )
            
            # Create scenario analysis visualizations
            create_all_scenario_visualizations(
                scenarios_df, rung_sensitivity_df, depth_sensitivity_df, combined_sensitivity_df
            )
            
            # Print summary
            print_summary(config, theta, p, fit_metrics, 
                         ladder_depths, allocations, 
                         paired_orders_df, current_price,
                         theta_sell, p_sell, 
                         fit_metrics_sell, sell_depths, 
                         actual_profits)
            
            print(f"\n{'='*60}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*60}")
            print(f"All outputs saved to output/ directory")
            print("\nStaggered order ladder system completed successfully!")
            
            if logger:
                logger.log_analysis_step("Analysis completed successfully", "SUCCESS")
                logger.log_performance_metrics({
                    "timeframe": "1h",
                    "optimal_profit_target": optimal_scenario['profit_target_pct'],
                    "best_expected_return": optimal_scenario['expected_profit_per_dollar']
                })
        
        except Exception as e:
            print(f"\nError: {e}")
            if logger:
                logger.log_error(e, "Main execution")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())