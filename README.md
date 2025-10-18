# Staggered Order Ladder System

A data-driven Python system for placing optimal limit buy orders in SOLUSDT to profit from sudden downward wicks. The system uses historical intraday behavior to fit a Weibull tail distribution and calculate optimal ladder depths and position sizes.

## Overview

This system implements a sophisticated approach to capturing mean-reverting price wicks by:

1. **Data Analysis**: Fetches SOLUSDT intraday data from Binance and analyzes maximum drops within configurable time horizons
2. **Statistical Modeling**: Fits a Weibull tail distribution to touch probabilities using maximum likelihood estimation
3. **Optimal Sizing**: Calculates monotone-increasing position sizes based on expected return per dollar
4. **Order Generation**: Creates exchange-compliant limit orders with proper tick/step size rounding
5. **Visualization**: Provides comprehensive plots and interactive dashboards for analysis validation

## Quick Start

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Parameters**:

   Edit `config.yaml` to set your preferences:

   - Budget and current price
   - Time horizon and lookback period
   - Ladder configuration (number of rungs, depth range)
   - Exchange constraints

3. **Run the System**:

   ```bash
   python main.py
   ```

4. **Review Outputs**:

   - `output/orders.csv` - Ready-to-place limit orders
   - `output/ladder_report.xlsx` - Comprehensive Excel analysis
   - `output/*.png` - Diagnostic plots
   - `output/ladder_dashboard.html` - Interactive dashboard

## Configuration Parameters

### Market Parameters

- `symbol`: Trading pair (default: "SOLUSDT")
- `budget_usd`: Total budget for ladder orders
- `current_price`: Current market price (auto-fetched if 0)

### Time Horizon & Data

- `horizon_minutes`: How long to keep ladder active (30-60 min recommended)
- `lookback_days`: Historical data window (90-180 days recommended)
- `candle_interval`: Binance kline interval ("1m", "5m", "15m", "1h")

### Ladder Configuration

- `num_rungs`: Number of ladder rungs (8-15 recommended)
- `top_rung_quantile`: How shallow the ladder starts (0.65-0.80)
- `bottom_rung_tail_prob`: Probability threshold for deepest rung (0.005-0.02)

### Exchange Constraints

- `tick_size`: Minimum price increment
- `step_size`: Minimum quantity increment
- `min_notional`: Minimum order value in USDT
- `min_qty`: Minimum quantity

## How It Works

### 1. Touch Probability Analysis

For each historical bar, the system looks ahead within the specified horizon and records the maximum drop from that bar's price. This creates an empirical distribution of downward wick depths.

### 2. Weibull Tail Fitting

The empirical touch probabilities are fitted to a Weibull tail distribution:

```
q(d) = exp(-(d/θ)^p)
```

Where:

- `θ` (theta) is the scale parameter (typical depth)
- `p` is the shape parameter (tail behavior)

### 3. Ladder Depth Calculation

- **Top rung**: Set by quantile of fitted distribution (e.g., 70th percentile)
- **Bottom rung**: Set by tail probability threshold (e.g., 1% chance)
- **Rung spacing**: Quantile-based spacing concentrates orders where marginal expected contribution is highest

### 4. Size Optimization

The system uses a monotone weight function:

```
w(d) = d^α × exp(-(d/θ)^p)
```

Where `α` is chosen to ensure deeper rungs always receive more allocation. This maximizes expected return per dollar while maintaining risk preferences.

### 5. Order Generation

- Limit prices: `P_k = P_0 × (1 - d_k/100)`
- Quantities: `Q_k = Allocation_k / P_k`
- All prices and quantities are rounded to exchange tick/step sizes
- Orders below minimum constraints are filtered out

## Output Files

### `orders.csv`

Ready-to-place limit orders with columns:

- `rung`: Ladder rung number
- `depth_pct`: Depth as percentage
- `limit_price`: Limit price in USD
- `quantity`: Order quantity
- `notional`: Order value in USD

### `ladder_report.xlsx`

Comprehensive Excel workbook with:

- **Order Specifications**: Complete order details
- **Parameters & Analysis**: Fitted parameters and summary statistics
- **Size Calculations**: Weight functions and allocation formulas
- **Touch Probability**: Detailed probability analysis

### Visualization Files

- `touch_fit.png`: Empirical vs fitted touch probability curve
- `expected_return_profile.png`: Expected return profile with ladder rungs
- `ladder_allocation.png`: Size distribution and order details
- `ladder_dashboard.html`: Interactive Plotly dashboard

## Parameter Tuning Guide

### Horizon Selection

- **Shorter horizons (30-45 min)**: Focus on true flash wicks, higher mean-reversion edge
- **Longer horizons (60+ min)**: Capture grinding sell-offs, lower edge but more fills

### Depth Range

- **Top rung quantile (u_min)**:
  - 0.65-0.70: More selective, higher edge per fill
  - 0.75-0.80: More fills, lower edge per fill
- **Bottom rung probability (p_min)**:
  - 0.005-0.01: Very selective, rare but high-value fills
  - 0.015-0.02: More frequent fills, moderate value

### Number of Rungs

- **8-10 rungs**: Simpler management, larger size per rung
- **12-15 rungs**: Better granularity, smaller size per rung

## Monitoring and Refitting

### When to Refit

- **Weekly**: Regular regime monitoring
- **After major market events**: Volatility regime changes
- **When fit quality degrades**: R² drops below 0.95
- **When ladder performance changes**: Fill patterns shift

### Fit Quality Indicators

- **R² > 0.95**: Excellent fit
- **R² 0.90-0.95**: Good fit, monitor closely
- **R² < 0.90**: Poor fit, consider different parameters or time window

### Regime Detection

Monitor these indicators for regime changes:

- Weibull parameters (θ, p) shifting significantly
- Touch probability curve shape changing
- Expected return profile mode shifting
- Fill frequency patterns changing

## Risk Management

### Position Sizing

- Total allocation never exceeds configured budget
- Monotone increasing sizes with depth (deeper = larger)
- Minimum order constraints enforced
- Maximum allocation ratio typically < 10:1

### Market Risk

- Orders are limit orders only (no market orders)
- Positions sized for mean-reversion (not trend-following)
- Horizon-based cancellation prevents stale orders
- Regular refitting adapts to changing market structure

### Operational Risk

- Comprehensive logging and diagnostics
- Fit quality validation prevents poor models
- Exchange constraint validation prevents rejected orders
- Visual validation through plots and dashboards

## Troubleshooting

### Common Issues

**"Poor fit quality" warning**:

- Increase lookback period
- Try different horizon length
- Check for data quality issues

**"Very tight spacing" warning**:

- Reduce number of rungs
- Adjust depth range parameters
- Check Weibull parameter values

**"Very high allocation ratio" warning**:

- Increase alpha parameter
- Adjust depth range
- Consider different budget allocation

**Orders below minimum constraints**:

- Increase budget
- Adjust depth range
- Check exchange constraint settings

### Data Issues

- Ensure stable internet connection for Binance API
- Check cache file permissions
- Verify symbol name and exchange availability

## Advanced Usage

### Custom Analysis

The modular design allows for custom analysis:

```python
from touch_analysis import analyze_touch_probabilities
from weibull_fit import fit_weibull_tail

# Custom analysis
depths, probs = analyze_touch_probabilities(df, horizon_minutes=30)
theta, p, metrics = fit_weibull_tail(depths, probs)
```

### Batch Processing

Process multiple symbols or timeframes:

```python
symbols = ['SOLUSDT', 'ETHUSDT', 'BTCUSDT']
for symbol in symbols:
    # Update config and run analysis
    pass
```

### Integration

The system outputs are designed for easy integration:

- CSV format for order management systems
- Excel formulas for manual recalculation
- JSON-serializable parameters for API integration

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review configuration parameters
3. Examine fit quality metrics
4. Validate with visualization outputs

The system is designed to be robust and self-diagnostic, with comprehensive logging and validation at each step.
