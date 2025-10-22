"""
Simplified output module for visualizations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_visualizations(depths: np.ndarray, probs: np.ndarray, theta: float, p: float, 
                         metrics: dict, ladder_depths: np.ndarray, allocations: np.ndarray, 
                         orders_df: pd.DataFrame) -> None:
    """
    Create basic visualizations.
    
    Args:
        depths: Empirical depth values
        probs: Empirical probabilities
        theta: Weibull scale parameter
        p: Weibull shape parameter
        metrics: Fit metrics
        ladder_depths: Ladder depth values
        allocations: Allocation amounts
        orders_df: Orders DataFrame
    """
    print("Creating visualizations...")
    
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Plot 1: Touch probability fit
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, probs, alpha=0.6, label='Empirical', color='blue')
    
    # Plot fitted curve
    fit_depths = np.linspace(0, 20, 100)
    fit_probs = np.exp(-(fit_depths / theta) ** p)
    plt.plot(fit_depths, fit_probs, 'r-', linewidth=2, label=f'Weibull Fit (R²={metrics["r_squared"]:.3f})')
    
    plt.xlabel('Depth (%)')
    plt.ylabel('Touch Probability')
    plt.title('Touch Probability Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/touch_probability_fit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Ladder allocation
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(allocations)), allocations, color='green', alpha=0.7)
    plt.xlabel('Rung')
    plt.ylabel('Allocation ($)')
    plt.title('Ladder Allocation Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/ladder_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Order prices
    plt.figure(figsize=(10, 6))
    plt.scatter(orders_df['rung'], orders_df['limit_price'], color='red', s=50, alpha=0.7)
    plt.xlabel('Rung')
    plt.ylabel('Limit Price ($)')
    plt.title('Order Limit Prices')
    plt.grid(True, alpha=0.3)
    plt.savefig('output/order_prices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to output/")
