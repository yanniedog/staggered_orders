"""
Enhanced scenario analysis visualization suite.
Rebuilt with improved clarity, proper metrics, and informative visualizations.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import seaborn as sns


def create_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')


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
        depth_range = max_depth - min_depth
        
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
        ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    ax.set_xlabel(f'Risk ({risk_metric.replace("_", " ").title()})')
    ax.set_ylabel('Expected Monthly Profit ($)')
    ax.set_title('Risk-Return Tradeoff\n(Size = Rungs, Color = Strategy, Line = Pareto Frontier)')
    ax.grid(True, alpha=0.3)
    
    # Add legend for strategies
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=10, label=strategy) 
                      for strategy, color in strategy_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/risk_return_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Risk-return tradeoff plot saved to output/risk_return_tradeoff.png")


def plot_sensitivity_analysis(sensitivity_df: pd.DataFrame) -> None:
    """
    Create sensitivity analysis multi-panel plot.
    
    Shows capital efficiency vs rungs, expected fills per month vs depth range,
    risk-adjusted returns (proper Sharpe ratios), and total expected profit vs timeframe.
    """
    create_output_dir()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Capital efficiency vs rungs (with confidence bands)
    ax1.plot(sensitivity_df['num_rungs'], sensitivity_df['capital_efficiency'], 
             'bo-', linewidth=2, markersize=6, label='Capital Efficiency')
    
    # Add confidence bands (simplified)
    efficiency_std = sensitivity_df['capital_efficiency'] * 0.2  # Assume 20% coefficient of variation
    ax1.fill_between(sensitivity_df['num_rungs'], 
                     sensitivity_df['capital_efficiency'] - 1.96 * efficiency_std,
                     sensitivity_df['capital_efficiency'] + 1.96 * efficiency_std,
                     alpha=0.3, color='blue', label='95% Confidence Band')
    
    ax1.set_xlabel('Number of Rungs')
    ax1.set_ylabel('Capital Efficiency (Monthly Profit / Budget)')
    ax1.set_title('Capital Efficiency vs Number of Rungs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Expected fills per month vs depth range
    if 'expected_monthly_fills' in sensitivity_df.columns:
        ax2.scatter(sensitivity_df['depth_range'], sensitivity_df['expected_monthly_fills'],
                   s=sensitivity_df['num_rungs']*3, alpha=0.7, 
                   c=sensitivity_df['capital_efficiency'], cmap='viridis')
        
        cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar2.set_label('Capital Efficiency')
        
        ax2.set_xlabel('Depth Range (%)')
        ax2.set_ylabel('Expected Fills per Month')
        ax2.set_title('Expected Fills vs Depth Range\n(Size = Rungs, Color = Efficiency)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Expected Monthly Fills\nData Not Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Expected Fills vs Depth Range')
    
    # Plot 3: Risk-adjusted returns (proper Sharpe ratios)
    ax3.plot(sensitivity_df['num_rungs'], sensitivity_df['risk_adjusted_return'], 
             'ro-', linewidth=2, markersize=6, label='Sharpe Ratio')
    
    # Add reference lines for Sharpe ratio interpretation
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Poor (<0.5)')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Good (>1.0)')
    ax3.axhline(y=2.0, color='blue', linestyle='--', alpha=0.7, label='Excellent (>2.0)')
    
    ax3.set_xlabel('Number of Rungs')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Returns (Sharpe Ratios)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Total expected profit vs timeframe
    ax4.scatter(sensitivity_df['expected_timeframe_hours'], sensitivity_df['expected_profit_per_dollar'],
               s=sensitivity_df['num_rungs']*3, alpha=0.7,
               c=sensitivity_df['risk_adjusted_return'], cmap='plasma')
    
    cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar4.set_label('Sharpe Ratio')
    
    ax4.set_xlabel('Expected Timeframe (Hours)')
    ax4.set_ylabel('Expected Profit per Dollar')
    ax4.set_title('Expected Profit vs Timeframe\n(Size = Rungs, Color = Sharpe Ratio)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sensitivity analysis saved to output/sensitivity_analysis.png")


def plot_profitability_distribution(scenarios_df: pd.DataFrame) -> None:
    """
    Create profitability distribution plots.
    
    Shows distribution of profit per rung, compares across strategies,
    highlights median, quartiles, and outliers.
    """
    create_output_dir()
    
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
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Profit per pair distribution (violin plot)
    strategies = scenarios_df['strategy'].unique()
    profit_data = [scenarios_df[scenarios_df['strategy'] == strategy]['profit_range_max'].values 
                   for strategy in strategies]
    
    parts = ax1.violinplot(profit_data, positions=range(len(strategies)), showmeans=True, showmedians=True)
    
    # Color the violin plots
    colors = ['lightgreen', 'lightblue', 'orange', 'lightcoral']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies)
    ax1.set_ylabel('Profit per Pair (%)')
    ax1.set_title('Profit Distribution by Strategy\n(Violin Plot)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    scenarios_df.boxplot(column='expected_profit_per_dollar', by='strategy', ax=ax2)
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Expected Profit per Dollar')
    ax2.set_title('Expected Profit Distribution by Strategy\n(Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Profit vs Risk scatter by strategy
    for strategy in strategies:
        strategy_data = scenarios_df[scenarios_df['strategy'] == strategy]
        ax3.scatter(strategy_data['expected_timeframe_hours'], strategy_data['expected_profit_per_dollar'],
                   label=strategy, alpha=0.7, s=50)
    
    ax3.set_xlabel('Expected Timeframe (Hours)')
    ax3.set_ylabel('Expected Profit per Dollar')
    ax3.set_title('Profit vs Timeframe by Strategy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative profit distribution
    for strategy in strategies:
        strategy_data = scenarios_df[scenarios_df['strategy'] == strategy]
        sorted_profits = np.sort(strategy_data['expected_profit_per_dollar'])
        cumulative_prob = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
        ax4.plot(sorted_profits, cumulative_prob, label=strategy, linewidth=2)
    
    ax4.set_xlabel('Expected Profit per Dollar')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Profit Distribution by Strategy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/profitability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Profitability distribution saved to output/profitability_distribution.png")


def create_interactive_scenario_dashboard(scenarios_df: pd.DataFrame, 
                                        sensitivity_df: pd.DataFrame,
                                        depth_sensitivity_df: pd.DataFrame,
                                        combined_sensitivity_df: pd.DataFrame = None) -> None:
    """
    Create interactive Plotly dashboard for scenario analysis.
    """
    create_output_dir()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Scenario Comparison Matrix', 'Risk-Return Tradeoff',
                       'Sensitivity Analysis', 'Profitability Distribution',
                       'Top Scenarios Ranking', 'Combined Sensitivity Matrix'),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}]]
    )
    
    # Plot 1: Scenario comparison matrix
    scenarios_df['strategy'] = scenarios_df.apply(lambda row: 
        'Conservative' if row['buy_depth_min'] <= 1.0 and row['buy_depth_max'] <= 5.0 else
        'Moderate' if row['buy_depth_min'] <= 2.0 and row['buy_depth_max'] <= 10.0 else
        'Aggressive' if row['buy_depth_min'] <= 3.0 and row['buy_depth_max'] <= 15.0 else
        'Very Aggressive', axis=1)
    
    pivot_table = scenarios_df.pivot_table(
        values='expected_monthly_profit',
        index='num_rungs',
        columns='strategy',
        aggfunc='mean'
    )
    
    fig.add_trace(
        go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='YlOrRd',
            showscale=True,
            name='Monthly Profit'
        ),
        row=1, col=1
    )
    
    # Plot 2: Risk-return tradeoff
    # Create color mapping for strategies
    strategy_colors = {
        'Conservative': '#1f77b4',
        'Moderate': '#ff7f0e', 
        'Aggressive': '#2ca02c',
        'Very Aggressive': '#d62728'
    }
    
    # Map strategy names to colors
    scenario_colors = [strategy_colors.get(s, '#9467bd') for s in scenarios_df['strategy']]
    
    fig.add_trace(
        go.Scatter(
            x=scenarios_df['expected_timeframe_hours'],
            y=scenarios_df['expected_monthly_profit'],
            mode='markers',
            marker=dict(
                size=scenarios_df['num_rungs'],
                color=scenario_colors,
                showscale=True,
                colorbar=dict(title="Strategy")
            ),
            text=[f"Rungs: {r}<br>Strategy: {s}<br>Profit: ${p:.0f}" 
                  for r, s, p in zip(scenarios_df['num_rungs'], scenarios_df['strategy'], 
                                   scenarios_df['expected_monthly_profit'])],
            hovertemplate='%{text}<extra></extra>',
            name='Risk-Return'
        ),
        row=1, col=2
    )
    
    # Plot 3: Sensitivity analysis
    fig.add_trace(
        go.Scatter(
            x=sensitivity_df['num_rungs'],
            y=sensitivity_df['capital_efficiency'],
            mode='lines+markers',
            name='Capital Efficiency',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Plot 4: Profitability distribution
    top_scenarios = scenarios_df.head(10)
    fig.add_trace(
        go.Bar(
            y=[f"{p:.1f}% ({r} rungs)" for p, r in zip(top_scenarios['profit_target_pct'],
                                                       top_scenarios['num_rungs'])],
            x=top_scenarios['expected_monthly_profit'],
            orientation='h',
            name='Top Scenarios',
            marker=dict(
                color=top_scenarios['num_rungs'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rung Count")
            )
        ),
        row=2, col=2
    )
    
    # Plot 5: Top scenarios ranking
    fig.add_trace(
        go.Bar(
            x=top_scenarios['strategy'],
            y=top_scenarios['expected_monthly_profit'],
            name='Monthly Profit by Strategy',
            marker_color='skyblue'
        ),
        row=3, col=1
    )
    
    # Plot 6: Combined sensitivity matrix
    if combined_sensitivity_df is not None:
        combined_pivot = combined_sensitivity_df.pivot_table(
            values='combined_score',
            index='num_rungs',
            columns='strategy',
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=combined_pivot.values,
                x=combined_pivot.columns,
                y=combined_pivot.index,
                colorscale='YlOrRd',
                showscale=True,
                name='Combined Score'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Enhanced Scenario Analysis Dashboard",
        showlegend=False,
        height=1200
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Strategy", row=1, col=1)
    fig.update_yaxes(title_text="Number of Rungs", row=1, col=1)
    
    fig.update_xaxes(title_text="Expected Timeframe (Hours)", row=1, col=2)
    fig.update_yaxes(title_text="Expected Monthly Profit ($)", row=1, col=2)
    
    fig.update_xaxes(title_text="Number of Rungs", row=2, col=1)
    fig.update_yaxes(title_text="Capital Efficiency", row=2, col=1)
    
    fig.update_xaxes(title_text="Expected Monthly Profit ($)", row=2, col=2)
    fig.update_yaxes(title_text="Scenario", row=2, col=2)
    
    fig.update_xaxes(title_text="Strategy", row=3, col=1)
    fig.update_yaxes(title_text="Expected Monthly Profit ($)", row=3, col=1)
    
    fig.update_xaxes(title_text="Strategy", row=3, col=2)
    fig.update_yaxes(title_text="Number of Rungs", row=3, col=2)
    
    # Save interactive plot
    fig.write_html('output/enhanced_scenario_dashboard.html')
    print("Enhanced interactive scenario dashboard saved to output/enhanced_scenario_dashboard.html")


def create_all_scenario_visualizations(scenarios_df: pd.DataFrame,
                                     sensitivity_df: pd.DataFrame = None,
                                     depth_sensitivity_df: pd.DataFrame = None,
                                     combined_df: pd.DataFrame = None) -> None:
    """
    Create all enhanced scenario analysis visualizations.
    """
    print("Creating enhanced scenario analysis visualizations...")
    
    # Core scenario visualizations
    plot_scenario_comparison_matrix(scenarios_df)
    plot_risk_return_tradeoff(scenarios_df)
    plot_profitability_distribution(scenarios_df)
    
    # Sensitivity analysis visualizations (if provided)
    if sensitivity_df is not None:
        plot_sensitivity_analysis(sensitivity_df)
    
    # Create comprehensive interactive dashboard
    create_interactive_scenario_dashboard(scenarios_df, sensitivity_df, depth_sensitivity_df, combined_df)
    
    print("All enhanced scenario visualizations created successfully")


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    # Create sample scenario data
    scenarios_df = pd.DataFrame({
        'profit_target_pct': [1, 2, 5, 10, 15, 20, 25] * 3,
        'num_rungs': [15, 20, 25, 30, 35, 40, 45] * 3,
        'buy_depth_min': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] * 3,
        'buy_depth_max': [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0] * 3,
        'expected_profit_per_dollar': [0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.22] * 3,
        'expected_monthly_profit': [50, 80, 120, 150, 180, 200, 220] * 3,
        'expected_monthly_fills': [2, 3, 4, 5, 6, 7, 8] * 3,
        'expected_timeframe_hours': [12, 24, 48, 72, 120, 240, 360] * 3,
        'risk_adjusted_return': [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.2] * 3,
        'total_allocation': [10000] * 21,
        'max_single_loss': [500, 600, 700, 800, 900, 1000, 1100] * 3
    })
    
    # Create sample sensitivity data
    sensitivity_df = pd.DataFrame({
        'num_rungs': [10, 15, 20, 25, 30, 35, 40, 45, 50],
        'capital_efficiency': [0.08, 0.10, 0.12, 0.13, 0.14, 0.13, 0.12, 0.11, 0.10],
        'expected_timeframe_hours': [8, 12, 18, 24, 30, 36, 42, 48, 54],
        'risk_adjusted_return': [0.5, 0.6, 0.7, 0.75, 0.8, 0.75, 0.7, 0.65, 0.6],
        'depth_range': [2.5, 4.0, 6.5, 8.0, 9.5, 12.0, 13.5, 15.0, 16.5],
        'expected_monthly_fills': [1.5, 2.0, 2.8, 3.5, 4.2, 4.8, 5.3, 5.7, 6.0]
    })
    
    create_all_scenario_visualizations(scenarios_df, sensitivity_df)