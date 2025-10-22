"""
Simplified main orchestrator for the staggered order ladder system.
"""
import pandas as pd
import numpy as np
import os
import warnings

# Import core modules
from config import load_config
from data_fetcher import fetch_solusdt_data, get_current_price
from analysis import analyze_touch_probabilities, fit_weibull_tail, calculate_ladder_depths, optimize_sizes
from order_builder import build_orders, export_orders_csv
from output import create_visualizations

def main():
    """Main execution function"""
    print("=" * 60)
    print("    STAGGERED ORDER LADDER SYSTEM")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"Budget: ${config['budget_usd']:.0f}")
    print(f"Lookback: {config['lookback_days']} days")
    
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
        print("Created output directory")
    
    # Fetch data
    print("Fetching data...")
    df = fetch_solusdt_data()
    print(f"Loaded {len(df)} candles")
    
    # Get current price
    current_price = get_current_price()
    print(f"Current {config['symbol']} price: ${current_price:.2f}")
    
    # Analyze touch probabilities
    print("Analyzing touch probabilities...")
    depths, probs = analyze_touch_probabilities(df, config['max_analysis_hours'])
    
    # Fit Weibull distribution
    print("Fitting Weibull distribution...")
    theta, p, metrics = fit_weibull_tail(depths, probs)
    print(f"Weibull fit: theta={theta:.3f}, p={p:.3f}, R2={metrics['r_squared']:.4f}")
    
    # Calculate ladder depths
    print("Calculating ladder depths...")
    ladder_depths = calculate_ladder_depths(theta, p, config['num_rungs'])
    
    # Optimize sizes
    print("Optimizing sizes...")
    allocations, alpha, expected_returns = optimize_sizes(ladder_depths, theta, p, config['budget_usd'])
    
    # Build orders
    print("Building orders...")
    orders_df = build_orders(ladder_depths, allocations, current_price)
    
    # Export results
    print("Exporting results...")
    export_orders_csv(orders_df, 'output/orders.csv')
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(depths, probs, theta, p, metrics, ladder_depths, allocations, orders_df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Symbol: {config['symbol']} | Price: ${current_price:.2f}")
    print(f"Weibull: theta={theta:.3f}, p={p:.3f}, R2={metrics['r_squared']:.4f}")
    print(f"Ladder: {len(ladder_depths)} rungs | Budget: ${config['budget_usd']:.0f}")
    print(f"Orders: {len(orders_df)} | Avg Profit: {orders_df['profit_pct'].mean():.2f}%")
    print("Outputs: output/orders.csv, output/*.png")
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    
    return 0

if __name__ == "__main__":
    exit(main())