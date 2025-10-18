# Interactive Staggered Order Ladder GUI

A professional, real-time interactive web-based GUI for visualizing and analyzing staggered order ladder configurations. Built with Dash and Plotly for maximum interactivity and performance.

## Features

### 🎛️ Interactive Controls
- **Aggression Level Slider**: 10 levels controlling depth range from conservative (0.5-5%) to very aggressive (5-30%)
- **Number of Rungs Slider**: 5-50 rungs with real-time validation
- **Timeframe Slider**: Historical analysis window from 1 hour to 30 days
- **Budget Input**: Configurable total budget with live updates
- **Current Price Display**: Live price updates with refresh button

### 📊 Real-time Visualizations (10+ Charts)

**Row 1: Core Ladder Visualizations**
1. **Ladder Configuration Chart**: Interactive scatter showing buy/sell rungs with prices and sizes
2. **Touch Probability Curves**: Buy-side and sell-side Weibull fits with current ladder overlay

**Row 2: Probability Analysis**
3. **Individual Rung Touch Probabilities**: Bar chart for each rung within selected timeframe
4. **Historical Touch Frequency**: Histogram showing how many times price crossed each rung level

**Row 3: Profitability Metrics**
5. **Expected Profit Distribution**: Box plot showing profit distribution across rungs
6. **Risk-Return Profile**: Scatter plot of expected return vs probability for each rung

**Row 4: Time-based Analysis**
7. **Touch vs Time Analysis**: Line chart showing cumulative touch probability over time
8. **Expected Performance Metrics**: KPI cards showing total expected profit, monthly fills, capital efficiency

**Row 5: Sensitivity & Validation**
9. **Allocation Distribution**: Pie chart showing capital allocation across rungs
10. **Fit Quality Dashboard**: Mini gauges showing Weibull fit quality metrics

### ⚡ Performance Features
- **Real-time Updates**: All charts update instantly as you move sliders
- **Debounced Calculations**: 300ms debouncing prevents excessive recalculation
- **LRU Caching**: Intelligent caching of calculations for identical parameters
- **Optimized Pipeline**: Complete recalculation in <500ms
- **Loading States**: Visual feedback during calculations

### 🎨 Professional Design
- **Dark Theme**: Sleek dark interface with professional color palette
- **Responsive Layout**: Adapts to different screen sizes
- **Interactive Charts**: Zoom, pan, hover tooltips on all visualizations
- **Smooth Animations**: Polished transitions and hover effects

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Historical Data (Optional)
```bash
python main.py
```
This creates the historical data cache used for touch frequency analysis.

### 3. Launch GUI
```bash
python run_gui.py
```
Or directly:
```bash
python gui_app.py
```

The GUI will open in your browser at `http://localhost:8050`

## Usage Guide

### Basic Operation
1. **Adjust Aggression Level**: Move the slider from 1 (conservative) to 10 (very aggressive)
2. **Set Number of Rungs**: Choose between 5-50 rungs for ladder granularity
3. **Select Timeframe**: Choose analysis window from 1 hour to 30 days
4. **Set Budget**: Enter your total budget in USD
5. **Watch Real-time Updates**: All charts update automatically as you adjust parameters

### Understanding the Visualizations

**Ladder Configuration Chart**
- Shows buy orders (green) and sell orders (red) as scatter points
- X-axis: Price levels
- Y-axis: Capital allocation
- Current price shown as dashed line

**Touch Probability Curves**
- Left: Buy-side touch probabilities vs depth
- Right: Sell-side touch probabilities vs depth
- Based on Weibull distribution fits

**Historical Touch Frequency**
- Shows how many times price touched each rung level
- Based on historical data analysis
- Updates with timeframe selection

**Risk-Return Profile**
- X-axis: Joint touch probability (risk)
- Y-axis: Expected profit percentage (return)
- Color: Depth level
- Size: Allocation amount

**Performance Metrics**
- **Total Expected Profit**: Monthly profit in USD
- **Monthly Fills**: Expected number of fills per month
- **Capital Efficiency**: Profit as percentage of budget
- **Expected Timeframe**: Average time between fills

### Advanced Features

**Export Configuration**
- Click "Export Configuration" to download current settings
- Saves as JSON for later use or sharing

**Price Refresh**
- Click "Refresh Price" to get latest market price
- Updates automatically every 5 seconds

**Real-time Analysis**
- All calculations use cached Weibull parameters for speed
- Historical analysis updates based on selected timeframe
- Touch frequency analysis provides realistic expectations

## Technical Details

### Architecture
- **Frontend**: Dash (React-based web framework)
- **Charts**: Plotly (interactive, professional visualizations)
- **Styling**: Dash Bootstrap Components + Custom CSS
- **Backend**: Python with existing ladder analysis modules
- **Caching**: LRU cache for expensive calculations

### Performance Optimizations
- **Debounced Updates**: Prevents excessive recalculation during slider movement
- **LRU Caching**: Caches ladder configurations for identical parameters
- **Vectorized Operations**: Uses NumPy for fast mathematical operations
- **Lazy Loading**: Only calculates visible charts
- **Background Updates**: Non-blocking price updates

### Data Flow
1. User adjusts slider → Debounced callback triggered
2. Calculate depth range from aggression level
3. Generate ladder using existing modules
4. Optimize sizes using Kelly criterion
5. Calculate all metrics (touch probabilities, expected profits)
6. Update all 10+ visualizations simultaneously
7. Cache results for identical parameter combinations

### File Structure
```
├── gui_app.py              # Main Dash application
├── gui_calculator.py       # Fast calculation engine
├── gui_visualizations.py   # Chart generation
├── gui_historical.py       # Historical analysis
├── config.py              # Configuration loader
├── assets/
│   └── style.css          # Custom styling
├── run_gui.py             # Launcher script
└── requirements.txt       # Dependencies
```

## Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
pip install -r requirements.txt
```

**GUI doesn't start**
- Check that all GUI files are present
- Ensure port 8050 is not in use
- Try running `python main.py` first

**Charts show "Error loading data"**
- Run `python main.py` to generate historical data
- Check that `cache_SOLUSDT_1h_1095d.csv` exists
- Verify internet connection for price updates

**Slow performance**
- Reduce number of rungs for faster calculation
- Close other browser tabs
- Check system resources

### Performance Tips
- Use fewer rungs (10-20) for faster updates
- Shorter timeframes (1-7 days) for quicker historical analysis
- Close unused browser tabs
- Ensure stable internet connection

## Customization

### Modifying Aggression Levels
Edit `gui_calculator.py` in the `_get_depth_range_for_aggression()` method:
```python
depth_mappings = {
    1: (0.5, 5.0),    # Conservative
    2: (1.0, 8.0),    # Conservative+
    # ... add more levels
}
```

### Adding New Visualizations
1. Add new chart function to `gui_visualizations.py`
2. Add new `dcc.Graph` component to layout in `gui_app.py`
3. Add output to main callback
4. Update `create_all_charts()` method

### Styling Changes
Edit `assets/style.css` for custom colors, fonts, and layout:
```css
:root {
    --primary-color: #007bff;
    --background-color: #1a1a1a;
    --text-color: #ffffff;
}
```

## Integration

### With Existing System
The GUI seamlessly integrates with the existing ladder analysis system:
- Uses same calculation modules (`ladder_depths.py`, `size_optimizer.py`)
- Leverages existing Weibull fitting (`weibull_fit.py`)
- Reads same configuration (`config.yaml`)
- Uses same historical data cache

### API Integration
The GUI can be extended to:
- Connect to live trading APIs
- Export orders directly to exchanges
- Integrate with portfolio management systems
- Add real-time market data feeds

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the technical documentation
3. Examine the console output for errors
4. Verify all dependencies are installed

The GUI is designed to be robust and self-diagnostic, with comprehensive error handling and user feedback.
