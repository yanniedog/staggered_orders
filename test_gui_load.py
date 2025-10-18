#!/usr/bin/env python3
"""Test script to verify GUI components load correctly without starting server"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    # Change to the correct directory
    os.chdir(r'C:\code\staggered_orders')

    print("Testing GUI component loading...")

    # Test GUI app initialization
    print("Testing GUI app initialization...")
    from gui_app import InteractiveLadderGUI

    # Initialize GUI (this should load all data)
    print("Creating InteractiveLadderGUI instance...")
    gui = InteractiveLadderGUI()

    print("GUI initialized successfully!")
    print(f"Calculator: {type(gui.calculator).__name__}")
    print(f"Historical: {type(gui.historical).__name__}")
    print(f"Visualizer: {type(gui.visualizer).__name__}")

    # Test if calculator has data loaded
    if gui.calculator.historical_data is not None:
        print(f"Calculator has historical data: {len(gui.calculator.historical_data)} rows")
    else:
        print("Calculator historical data is None")

    if gui.historical.historical_data is not None:
        print(f"Historical analyzer has data: {len(gui.historical.historical_data)} rows")
    else:
        print("Historical analyzer data is None")

    # Test a simple calculation
    print("Testing ladder calculation...")
    ladder_data = gui.calculator.calculate_ladder_configuration(
        aggression_level=5,
        num_rungs=10,
        timeframe_hours=168,
        budget=5000
    )

    if ladder_data and 'buy_depths' in ladder_data:
        print("Ladder calculation successful!")
        print(f"Generated {len(ladder_data['buy_depths'])} buy levels")
        print(f"Generated {len(ladder_data['sell_depths'])} sell levels")
        print(f"Current price: ${ladder_data['current_price']:.2f}")
    else:
        print("ERROR: Ladder calculation failed!")
        sys.exit(1)

    # Test visualization creation
    print("Testing visualization creation...")
    figures = gui.visualizer.create_all_charts(ladder_data, 168)

    if figures and len(figures) == 9:
        print("All visualizations created successfully!")
        for i, fig in enumerate(figures):
            title = fig.layout.title.text if hasattr(fig.layout, 'title') and fig.layout.title else 'Unnamed'
            print(f"  Figure {i+1}: {title}")
    else:
        print(f"ERROR: Expected 9 figures, got {len(figures) if figures else 0}")
        sys.exit(1)

    print("\nAll GUI loading tests passed! The GUI should work correctly.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
