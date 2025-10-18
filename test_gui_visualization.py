#!/usr/bin/env python3
"""Test script to check if GUI visualization creation works properly"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    # Change to the correct directory
    os.chdir(r'C:\code\staggered_orders')

    print("Testing GUI visualization creation...")

    # Test data loading
    from gui_calculator import LadderCalculator
    from gui_historical import HistoricalAnalyzer
    from gui_visualizations import VisualizationEngine

    # Initialize components
    calculator = LadderCalculator()
    historical = HistoricalAnalyzer()
    visualizer = VisualizationEngine(historical)

    print("Components initialized successfully")

    # Test ladder calculation
    print("Testing ladder calculation...")
    ladder_data = calculator.calculate_ladder_configuration(
        aggression_level=5,
        num_rungs=20,
        timeframe_hours=720,
        budget=10000
    )

    if ladder_data and 'buy_depths' in ladder_data:
        print("Ladder calculation successful")
        print(f"Buy depths: {len(ladder_data['buy_depths'])} levels")
        print(f"Sell depths: {len(ladder_data['sell_depths'])} levels")
    else:
        print("ERROR: Ladder calculation failed!")
        sys.exit(1)

    # Test visualization creation
    print("Testing visualization creation...")
    figures = visualizer.create_all_charts(ladder_data, 720)

    if figures and len(figures) == 9:
        print("All 9 visualizations created successfully")
        for i, fig in enumerate(figures):
            print(f"  Figure {i+1}: {fig.layout.title.text if hasattr(fig.layout, 'title') and fig.layout.title else 'Unnamed'}")
    else:
        print(f"ERROR: Expected 9 figures, got {len(figures) if figures else 0}")
        sys.exit(1)

    # Test KPI calculation
    kpis = calculator.calculate_kpis(ladder_data)
    print(f"KPIs calculated: {kpis}")

    print("\nAll GUI tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
