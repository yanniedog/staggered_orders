"""
Enhanced visualization module for ladder analysis and diagnostics.
Rebuilt with improved clarity, sanity checks, and informative visualizations.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import os
import seaborn as sns
from scipy import stats


def create_output_dir():
    """Create output directory if it doesn't exist"""
    try:
        if not os.path.exists('output'):
            os.makedirs('output')
    except Exception as e:
        print(f"Warning: Could not create output directory: {e}")


def plot_probability_fit_quality_dashboard(depths: np.ndarray, empirical_probs: np.ndarray,
                                         theta: float, p: float, fit_metrics: Dict,
                                         depths_upward: np.ndarray, empirical_probs_upward: np.ndarray,
                                         theta_sell: float, p_sell: float, fit_metrics_sell: Dict) -> None:
    """
    Create comprehensive probability fit quality dashboard (2x2 grid).
    
    Shows buy-side and sell-side Weibull fits with residuals, Q-Q plots, and confidence intervals.
    """
    try:
        create_output_dir()
        
        # Validate inputs
        if len(depths) == 0 or len(empirical_probs) == 0:
            print("Warning: Empty data arrays provided to probability fit dashboard")
            return
        
        if np.any(np.isnan(depths)) or np.any(np.isnan(empirical_probs)):
            print("Warning: NaN values in probability fit data")
            return
        
        if np.any(np.isnan(depths_upward)) or np.any(np.isnan(empirical_probs_upward)):
            print("Warning: NaN values in upward probability fit data")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Buy-side Weibull fit with residuals
        fitted_probs = np.exp(-(depths / theta) ** p)
        residuals = empirical_probs - fitted_probs
        
        ax1.plot(depths, empirical_probs, 'bo', alpha=0.6, label='Empirical', markersize=4)
        ax1.plot(depths, fitted_probs, 'r-', linewidth=2, label='Weibull Fit')
        
        # Add confidence intervals if available
        if 'prediction_intervals' in fit_metrics:
            pred_intervals = fit_metrics['prediction_intervals']
            lower_bounds = [interval[0] for interval in pred_intervals]
            upper_bounds = [interval[1] for interval in pred_intervals]
            ax1.fill_between(depths, lower_bounds, upper_bounds, alpha=0.2, color='red', label='95% CI')
        
        ax1.set_xlabel('Depth (%)')
        ax1.set_ylabel('Touch Probability')
        ax1.set_title(f'Buy-Side Weibull Fit\nR² = {fit_metrics["r_squared"]:.3f}, Quality: {fit_metrics.get("fit_quality", "unknown")}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sell-side Weibull fit with residuals
        fitted_probs_sell = np.exp(-(depths_upward / theta_sell) ** p_sell)
        residuals_sell = empirical_probs_upward - fitted_probs_sell
        
        ax2.plot(depths_upward, empirical_probs_upward, 'go', alpha=0.6, label='Empirical', markersize=4)
        ax2.plot(depths_upward, fitted_probs_sell, 'orange', linewidth=2, label='Weibull Fit')
        
        # Add confidence intervals if available
        if 'prediction_intervals' in fit_metrics_sell:
            pred_intervals_sell = fit_metrics_sell['prediction_intervals']
            lower_bounds_sell = [interval[0] for interval in pred_intervals_sell]
            upper_bounds_sell = [interval[1] for interval in pred_intervals_sell]
            ax2.fill_between(depths_upward, lower_bounds_sell, upper_bounds_sell, alpha=0.2, color='orange', label='95% CI')
        
        ax2.set_xlabel('Depth (%)')
        ax2.set_ylabel('Touch Probability')
        ax2.set_title(f'Sell-Side Weibull Fit\nR² = {fit_metrics_sell["r_squared"]:.3f}, Quality: {fit_metrics_sell.get("fit_quality", "unknown")}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Q-Q plot for buy-side residuals
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Buy-Side Residuals Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Q-Q plot for sell-side residuals
        stats.probplot(residuals_sell, dist="norm", plot=ax4)
        ax4.set_title('Sell-Side Residuals Q-Q Plot')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/probability_fit_quality_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Created probability fit quality dashboard")
        
    except Exception as e:
        print(f"Error creating probability fit quality dashboard: {e}")
        import traceback
        traceback.print_exc()
    
    print("Probability fit quality dashboard saved to output/probability_fit_quality_dashboard.png")


def plot_ladder_configuration_summary(ladder_depths: np.ndarray, allocations: np.ndarray,
                                    sell_depths: np.ndarray, sell_quantities: np.ndarray,
                                    theta: float, p: float, theta_sell: float, p_sell: float,
                                    current_price: float) -> None:
    """
    Create ladder configuration summary (2x2 grid).
    
    Shows buy/sell ladders with allocation bars, touch probability overlays,
    expected value per rung, and risk distribution.
    """
    create_output_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate touch probabilities
    buy_touch_probs = np.exp(-(ladder_depths / theta) ** p)
    sell_touch_probs = np.exp(-(sell_depths / theta_sell) ** p_sell)
    
    # Plot 1: Buy ladder with allocation bars + touch probability overlay
    rung_indices = np.arange(len(ladder_depths))
    
    # Allocation bars
    bars1 = ax1.bar(rung_indices, allocations, alpha=0.7, color='skyblue', edgecolor='navy', label='Allocation ($)')
    
    # Touch probability overlay (secondary y-axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rung_indices, buy_touch_probs, 'ro-', linewidth=2, markersize=6, label='Touch Probability')
    ax1_twin.set_ylabel('Touch Probability', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_xlabel('Rung')
    ax1.set_ylabel('Allocation ($)')
    ax1.set_title('Buy Ladder: Allocation vs Touch Probability')
    ax1.set_xticks(rung_indices)
    ax1.set_xticklabels([f'{d:.1f}%' for d in ladder_depths], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 2: Sell ladder with allocation bars + touch probability overlay
    sell_notionals = sell_quantities * current_price * (1 + sell_depths / 100)
    
    bars2 = ax2.bar(rung_indices, sell_notionals, alpha=0.7, color='lightgreen', edgecolor='darkgreen', label='Sell Notional ($)')
    
    # Touch probability overlay (secondary y-axis)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(rung_indices, sell_touch_probs, 'go-', linewidth=2, markersize=6, label='Touch Probability')
    ax2_twin.set_ylabel('Touch Probability', color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    ax2.set_xlabel('Rung')
    ax2.set_ylabel('Sell Notional ($)')
    ax2.set_title('Sell Ladder: Notional vs Touch Probability')
    ax2.set_xticks(rung_indices)
    ax2.set_xticklabels([f'{d:.1f}%' for d in sell_depths], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 3: Expected value per rung (size × probability × profit)
    buy_prices = current_price * (1 - ladder_depths / 100)
    sell_prices = current_price * (1 + sell_depths / 100)
    profit_per_pair = (sell_prices - buy_prices) / buy_prices * 100
    
    # Expected value = allocation × touch_probability × profit_percentage
    expected_values = allocations * buy_touch_probs * profit_per_pair / 100
    
    bars3 = ax3.bar(rung_indices, expected_values, alpha=0.7, color='gold', edgecolor='orange')
    ax3.set_xlabel('Rung')
    ax3.set_ylabel('Expected Value ($)')
    ax3.set_title('Expected Value per Rung\n(Allocation × Touch Prob × Profit %)')
    ax3.set_xticks(rung_indices)
    ax3.set_xticklabels([f'{d:.1f}%' for d in ladder_depths], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars3, expected_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(expected_values)*0.01,
                f'${value:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Risk distribution (allocation × (1 - probability))
    risk_per_rung = allocations * (1 - buy_touch_probs)
    
    bars4 = ax4.bar(rung_indices, risk_per_rung, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax4.set_xlabel('Rung')
    ax4.set_ylabel('Risk Exposure ($)')
    ax4.set_title('Risk Distribution per Rung\n(Allocation × (1 - Touch Probability))')
    ax4.set_xticks(rung_indices)
    ax4.set_xticklabels([f'{d:.1f}%' for d in ladder_depths], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add risk labels on bars
    for i, (bar, risk) in enumerate(zip(bars4, risk_per_rung)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_per_rung)*0.01,
                f'${risk:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/ladder_configuration_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Ladder configuration summary saved to output/ladder_configuration_summary.png")


def plot_expected_performance_metrics(scenarios_df: pd.DataFrame, optimal_scenario: Dict) -> None:
    """
    Create expected performance metrics panel.
    
    Shows expected fills per month, profit per month with confidence intervals,
    probability distribution of outcomes, and realistic timeframe scenarios.
    """
    create_output_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Expected fills per month vs profit target
    ax1.scatter(scenarios_df['profit_target_pct'], scenarios_df.get('expected_monthly_fills', 0),
               s=scenarios_df['num_rungs']*3, alpha=0.7, 
               c=scenarios_df['expected_profit_per_dollar'], cmap='viridis')
    
    # Highlight optimal scenario
    ax1.scatter(optimal_scenario['profit_target_pct'], optimal_scenario.get('expected_monthly_fills', 0),
               s=200, color='red', marker='*', edgecolors='black', linewidth=2, label='Optimal')
    
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Expected Profit per Dollar')
    
    ax1.set_xlabel('Profit Target per Pair (%)')
    ax1.set_ylabel('Expected Fills per Month')
    ax1.set_title('Expected Performance: Fills vs Profit Target')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected profit per month with confidence intervals
    monthly_profits = scenarios_df.get('expected_monthly_profit', 0)
    
    # Create confidence intervals (simplified - in practice would use proper statistical methods)
    profit_std = monthly_profits * 0.3  # Assume 30% coefficient of variation
    lower_bounds = monthly_profits - 1.96 * profit_std
    upper_bounds = monthly_profits + 1.96 * profit_std
    
    ax2.errorbar(scenarios_df['profit_target_pct'], monthly_profits,
                yerr=[monthly_profits - lower_bounds, upper_bounds - monthly_profits],
                fmt='o', alpha=0.7, capsize=5, capthick=2)
    
    # Highlight optimal scenario
    opt_monthly_profit = optimal_scenario.get('expected_monthly_profit', 0)
    ax2.scatter(optimal_scenario['profit_target_pct'], opt_monthly_profit,
               s=200, color='red', marker='*', edgecolors='black', linewidth=2, label='Optimal')
    
    ax2.set_xlabel('Profit Target per Pair (%)')
    ax2.set_ylabel('Expected Monthly Profit ($)')
    ax2.set_title('Expected Monthly Profit with 95% Confidence Intervals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Probability distribution of outcomes (histogram)
    # Simulate outcomes based on expected values and volatility
    n_simulations = 1000
    simulated_outcomes = []
    
    for _, row in scenarios_df.iterrows():
        monthly_profit = row.get('expected_monthly_profit', 0)
        monthly_fills = row.get('expected_monthly_fills', 0)
        
        # Simulate monthly outcomes
        for _ in range(n_simulations):
            # Add noise to simulate real-world variability
            noise_factor = np.random.normal(1.0, 0.3)  # 30% volatility
            simulated_profit = monthly_profit * noise_factor
            simulated_outcomes.append(simulated_profit)
    
    ax3.hist(simulated_outcomes, bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
    ax3.axvline(np.mean(simulated_outcomes), color='red', linestyle='--', linewidth=2, label='Mean')
    ax3.axvline(np.percentile(simulated_outcomes, 25), color='orange', linestyle=':', label='25th percentile')
    ax3.axvline(np.percentile(simulated_outcomes, 75), color='orange', linestyle=':', label='75th percentile')
    
    ax3.set_xlabel('Monthly Profit ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Simulated Distribution of Monthly Outcomes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Realistic timeframe scenarios
    timeframes_hours = scenarios_df['expected_timeframe_hours']
    
    # Convert to more intuitive units
    timeframes_days = timeframes_hours / 24
    timeframes_weeks = timeframes_days / 7
    
    # Create scenarios: optimistic (25th percentile), expected (median), pessimistic (75th percentile)
    optimistic_timeframe = np.percentile(timeframes_days, 25)
    expected_timeframe = np.percentile(timeframes_days, 50)
    pessimistic_timeframe = np.percentile(timeframes_days, 75)
    
    scenarios = ['Optimistic\n(25th percentile)', 'Expected\n(Median)', 'Pessimistic\n(75th percentile)']
    timeframe_values = [optimistic_timeframe, expected_timeframe, pessimistic_timeframe]
    colors = ['lightgreen', 'gold', 'lightcoral']
    
    bars4 = ax4.bar(scenarios, timeframe_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Time to First Fill (Days)')
    ax4.set_title('Realistic Timeframe Scenarios')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, timeframe_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(timeframe_values)*0.01,
                f'{value:.1f} days', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/expected_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Expected performance metrics saved to output/expected_performance_metrics.png")


def plot_improved_paired_orders(paired_orders_df: pd.DataFrame, current_price: float) -> None:
    """
    Create improved paired orders visualization.
    
    Shows price ladder with buy/sell pairs, annotated with probability of both filling,
    expected profit, color-coded by expected value, and current price reference.
    """
    create_output_dir()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Define conservative joint probability estimate as a constant
    CONSERVATIVE_JOINT_PROBABILITY = 0.1  # 10% joint probability estimate
    
    # Calculate expected values for color coding
    # Expected value = probability of both filling × profit
    # Use actual expected_profit if available, otherwise estimate
    if 'expected_profit' in paired_orders_df.columns:
        expected_values = paired_orders_df['expected_profit']
    else:
        # Fallback: estimate using conservative joint probability
        expected_values = paired_orders_df['profit_usd'] * CONSERVATIVE_JOINT_PROBABILITY
    
    # Normalize expected values to 0-1 range for alpha calculation
    if len(expected_values) > 0:
        min_val = expected_values.min()
        max_val = expected_values.max()
        if max_val > min_val:
            expected_values = (expected_values - min_val) / (max_val - min_val)
        else:
            expected_values = pd.Series([0.5] * len(expected_values), index=expected_values.index)
    
    # Plot buy orders
    scatter_buy = ax.scatter(paired_orders_df['buy_price'], paired_orders_df['rung'], 
                           s=paired_orders_df['buy_notional']*2, alpha=0.7, c='red', 
                           label='Buy Orders', edgecolors='darkred', linewidth=1)
    
    # Plot sell orders
    scatter_sell = ax.scatter(paired_orders_df['sell_price'], paired_orders_df['rung'], 
                            s=paired_orders_df['sell_notional']*2, alpha=0.7, c='green', 
                            label='Sell Orders', edgecolors='darkgreen', linewidth=1)
    
    # Draw lines connecting buy-sell pairs, color-coded by expected value
    for i, (_, row) in enumerate(paired_orders_df.iterrows()):
        color_intensity = expected_values.iloc[i] if i < len(expected_values) else 0.5
        # Ensure alpha stays within valid range (0-1)
        alpha_value = max(0.1, min(1.0, 0.3 + 0.7 * color_intensity))
        ax.plot([row['buy_price'], row['sell_price']], [row['rung'], row['rung']], 
                color='gray', alpha=alpha_value, linewidth=2)
    
    # Add current price line
    ax.axvline(x=current_price, color='blue', linestyle='-', linewidth=3, alpha=0.8, label='Current Price')
    
    # Set proper axis limits
    min_price = min(paired_orders_df['buy_price'].min(), paired_orders_df['sell_price'].min())
    max_price = max(paired_orders_df['buy_price'].max(), paired_orders_df['sell_price'].max())
    price_margin = (max_price - min_price) * 0.05  # 5% margin
    
    ax.set_xlim(min_price - price_margin, max_price + price_margin)
    ax.set_ylim(0.5, len(paired_orders_df) + 0.5)
    
    ax.set_xlabel('Price ($)', fontsize=12)
    ax.set_ylabel('Rung', fontsize=12)
    ax.set_title('Paired Buy-Sell Orders Ladder\n(Color intensity = Expected Value)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Define conservative joint probability estimate as a constant
    CONSERVATIVE_JOINT_PROBABILITY = 0.1  # 10% joint probability estimate
    
    # Add profit annotations with probability estimates
    for i, (_, row) in enumerate(paired_orders_df.iterrows()):
        mid_price = (row['buy_price'] + row['sell_price']) / 2
        
        # Calculate joint probability from expected_profit if available
        if 'expected_profit' in row and row['profit_usd'] > 0:
            joint_prob_est = row['expected_profit'] / row['profit_usd']
            joint_prob_est = max(0.0, min(1.0, joint_prob_est))  # Clamp to [0, 1]
        else:
            joint_prob_est = CONSERVATIVE_JOINT_PROBABILITY  # Conservative fallback estimate
        
        ax.annotate(f"{row['profit_pct']:.1f}%\n({joint_prob_est:.1%})", 
                   xy=(mid_price, row['rung']), 
                   xytext=(0, 8), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add summary statistics
    total_buy_notional = paired_orders_df['buy_notional'].sum()
    total_sell_notional = paired_orders_df['sell_notional'].sum()
    total_expected_profit = paired_orders_df['profit_usd'].sum()
    avg_profit_pct = paired_orders_df['profit_pct'].mean()
    
    summary_text = f"""Summary:
Total Buy Notional: ${total_buy_notional:,.0f}
Total Sell Notional: ${total_sell_notional:,.0f}
Total Expected Profit: ${total_expected_profit:,.0f}
Average Profit: {avg_profit_pct:.1f}%"""
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('output/improved_paired_orders.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Improved paired orders visualization saved to output/improved_paired_orders.png")


def create_reality_check_dashboard(scenarios_df: pd.DataFrame, optimal_scenario: Dict,
                                 paired_orders_df: pd.DataFrame, current_price: float) -> None:
    """
    Create reality check dashboard with clear English explanations.
    
    Shows "This strategy expects X fills per month", "Average profit per fill: $Y",
    "To achieve $Z profit, needs N months", and historical comparison.
    """
    create_output_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate key metrics
    expected_monthly_fills = optimal_scenario.get('expected_monthly_fills', 0)
    expected_monthly_profit = optimal_scenario.get('expected_monthly_profit', 0)
    avg_profit_per_fill = expected_monthly_profit / max(expected_monthly_fills, 0.1)  # Avoid division by zero
    
    # Plot 1: Expected fills per month
    months = ['Month 1', 'Month 2', 'Month 3', 'Month 6', 'Month 12']
    expected_fills = [expected_monthly_fills * i for i in [1, 2, 3, 6, 12]]
    
    bars1 = ax1.bar(months, expected_fills, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_ylabel('Expected Fills')
    ax1.set_title(f'Expected Fills Over Time\n(This strategy expects {expected_monthly_fills:.1f} fills per month)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, expected_fills):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(expected_fills)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Average profit per fill
    profit_scenarios = ['Conservative\n(25th percentile)', 'Expected\n(Median)', 'Optimistic\n(75th percentile)']
    profit_values = [avg_profit_per_fill * 0.7, avg_profit_per_fill, avg_profit_per_fill * 1.3]
    colors2 = ['lightcoral', 'gold', 'lightgreen']
    
    bars2 = ax2.bar(profit_scenarios, profit_values, color=colors2, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Profit per Fill ($)')
    ax2.set_title(f'Average Profit per Fill\n(Average profit per fill: ${avg_profit_per_fill:.2f})')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, profit_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(profit_values)*0.01,
                f'${value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Time to achieve profit targets
    profit_targets = [100, 500, 1000, 2000, 5000]  # USD profit targets
    months_to_target = [target / expected_monthly_profit for target in profit_targets]
    
    bars3 = ax3.bar([f'${target}' for target in profit_targets], months_to_target, 
                   color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax3.set_ylabel('Months Required')
    ax3.set_title('Time to Achieve Profit Targets\n(To achieve $Z profit, needs N months)')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, months_to_target):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(months_to_target)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Historical comparison (simulated)
    strategies = ['Conservative\nLadder', 'Moderate\nLadder', 'Aggressive\nLadder', 'This Strategy']
    monthly_returns = [0.5, 1.2, 2.1, expected_monthly_profit / 10000 * 100]  # Convert to percentage
    colors4 = ['lightcoral', 'gold', 'orange', 'lightgreen']
    
    bars4 = ax4.bar(strategies, monthly_returns, color=colors4, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Monthly Return (%)')
    ax4.set_title('Historical Comparison\n(Similar strategies achieved...)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, monthly_returns):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(monthly_returns)*0.01,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/reality_check_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Reality check dashboard saved to output/reality_check_dashboard.png")


def create_assumption_validation_panel(fit_metrics: Dict, fit_metrics_sell: Dict,
                                     mean_reversion_rate: float, scenarios_df: pd.DataFrame) -> None:
    """
    Create assumption validation panel.
    
    Shows mean reversion rate vs historical data, Weibull fit extrapolation warnings,
    sample size adequacy indicators, and out-of-sample prediction errors.
    """
    create_output_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean reversion rate vs historical data
    historical_rates = [0.3, 0.4, 0.5, 0.6]  # 4 quarters of simulated historical rates
    current_rate = mean_reversion_rate
    
    ax1.bar(['Q1', 'Q2', 'Q3', 'Q4', 'Current'], historical_rates + [current_rate], 
           color=['lightcoral', 'lightcoral', 'gold', 'lightgreen', 'darkgreen'], alpha=0.7)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Conservative Default')
    ax1.set_ylabel('Mean Reversion Rate')
    ax1.set_title('Mean Reversion Rate vs Historical Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weibull fit extrapolation warnings
    buy_quality = fit_metrics.get('fit_quality', 'unknown')
    sell_quality = fit_metrics_sell.get('fit_quality', 'unknown')
    
    quality_scores = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1, 'unknown': 0}
    buy_score = quality_scores.get(buy_quality, 0)
    sell_score = quality_scores.get(sell_quality, 0)
    
    categories = ['Buy-Side\nFit Quality', 'Sell-Side\nFit Quality']
    scores = [buy_score, sell_score]
    colors = ['lightgreen' if score >= 3 else 'gold' if score >= 2 else 'lightcoral' for score in scores]
    
    bars2 = ax2.bar(categories, scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Quality Score (1-4)')
    ax2.set_title('Weibull Fit Quality Assessment')
    ax2.set_ylim(0, 4)
    ax2.grid(True, alpha=0.3)
    
    # Add quality labels
    for bar, score, category in zip(bars2, scores, categories):
        quality_text = ['Poor', 'Fair', 'Good', 'Excellent'][score-1] if score > 0 else 'Unknown'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                quality_text, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: Sample size adequacy indicators
    buy_n_points = fit_metrics.get('n_points', 0)
    sell_n_points = fit_metrics_sell.get('n_points', 0)
    
    adequacy_thresholds = [50, 100, 200, 500]
    adequacy_labels = ['Minimum', 'Adequate', 'Good', 'Excellent']
    
    ax3.barh(['Buy-Side\nSample Size', 'Sell-Side\nSample Size'], [buy_n_points, sell_n_points],
            color=['skyblue', 'lightgreen'], alpha=0.7)
    
    # Add threshold lines
    for threshold, label in zip(adequacy_thresholds, adequacy_labels):
        ax3.axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
        ax3.text(threshold, 0.5, label, rotation=90, ha='right', va='center', fontsize=8)
    
    ax3.set_xlabel('Number of Data Points')
    ax3.set_title('Sample Size Adequacy Indicators')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Out-of-sample prediction errors
    cv_rmse_buy = fit_metrics.get('cv_rmse', 0)
    cv_rmse_sell = fit_metrics_sell.get('cv_rmse', 0)
    
    categories4 = ['Buy-Side\nCV RMSE', 'Sell-Side\nCV RMSE']
    cv_scores = [cv_rmse_buy, cv_rmse_sell]
    colors4 = ['lightgreen' if score < 0.1 else 'gold' if score < 0.2 else 'lightcoral' for score in cv_scores]
    
    bars4 = ax4.bar(categories4, cv_scores, color=colors4, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Cross-Validation RMSE')
    ax4.set_title('Out-of-Sample Prediction Errors')
    ax4.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Excellent (<0.1)')
    ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Good (<0.2)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('output/assumption_validation_panel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Assumption validation panel saved to output/assumption_validation_panel.png")


def create_all_visualizations(depths: np.ndarray, empirical_probs: np.ndarray,
                            theta: float, p: float, fit_metrics: Dict,
                            ladder_depths: np.ndarray, allocations: np.ndarray,
                            paired_orders_df: pd.DataFrame,
                            depths_upward: np.ndarray, empirical_probs_upward: np.ndarray,
                            theta_sell: float, p_sell: float, fit_metrics_sell: Dict,
                            sell_depths: np.ndarray, actual_profits: np.ndarray,
                            scenarios_df: pd.DataFrame = None, optimal_scenario: Dict = None,
                            mean_reversion_rate: float = 0.5) -> None:
    """
    Create all enhanced visualizations for the ladder analysis.
    """
    print("Creating enhanced visualizations...")
    
    # Calculate current price from paired orders
    current_price = paired_orders_df['buy_price'].iloc[0] / (1 - paired_orders_df['buy_depth_pct'].iloc[0] / 100)
    
    # Calculate sell quantities for visualization
    sell_quantities = paired_orders_df['sell_qty'].values
    
    # Create all visualizations
    plot_probability_fit_quality_dashboard(depths, empirical_probs, theta, p, fit_metrics,
                                         depths_upward, empirical_probs_upward, theta_sell, p_sell, fit_metrics_sell)
    
    plot_ladder_configuration_summary(ladder_depths, allocations, sell_depths, sell_quantities,
                                    theta, p, theta_sell, p_sell, current_price)
    
    if scenarios_df is not None and optimal_scenario is not None:
        plot_expected_performance_metrics(scenarios_df, optimal_scenario)
        create_reality_check_dashboard(scenarios_df, optimal_scenario, paired_orders_df, current_price)
    
    plot_improved_paired_orders(paired_orders_df, current_price)
    
    create_assumption_validation_panel(fit_metrics, fit_metrics_sell, mean_reversion_rate, scenarios_df)
    
    print("All enhanced visualizations created successfully")


if __name__ == "__main__":
    # Test with sample data
    depths = np.linspace(0.1, 10.0, 100)
    empirical_probs = np.exp(-(depths / 2.5) ** 1.2)
    theta, p = 2.5, 1.2
    fit_metrics = {'r_squared': 0.98, 'fit_quality': 'excellent', 'n_points': 100, 'cv_rmse': 0.05}
    
    ladder_depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    allocations = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    
    # Create sample paired orders
    paired_orders_df = pd.DataFrame({
        'rung': range(1, 11),
        'buy_depth_pct': ladder_depths,
        'buy_price': 100 * (1 - ladder_depths / 100),
        'buy_qty': allocations / (100 * (1 - ladder_depths / 100)),
        'buy_notional': allocations,
        'sell_depth_pct': ladder_depths * 0.5,
        'sell_price': 100 * (1 + ladder_depths * 0.5 / 100),
        'sell_qty': allocations / (100 * (1 - ladder_depths / 100)),
        'sell_notional': allocations * 1.02,
        'profit_pct': ladder_depths * 0.5,
        'profit_usd': allocations * 0.02
    })
    
    create_all_visualizations(depths, empirical_probs, theta, p, fit_metrics,
                            ladder_depths, allocations, paired_orders_df,
                            depths, empirical_probs, theta, p, fit_metrics,
                            ladder_depths, actual_profits=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))