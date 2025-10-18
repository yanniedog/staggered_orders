#!/usr/bin/env python3
"""Test script to check if data loading works properly"""

import pandas as pd
import sys
import os

try:
    # Change to the correct directory
    os.chdir(r'C:\code\staggered_orders')

    print("Testing data loading...")
    df = pd.read_csv('cache_SOLUSDT_1h_1095d.csv')

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if open_time is properly parsed as datetime
    if 'open_time' in df.columns:
        print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        print(f"Sample open_time values: {df['open_time'].head()}")
    else:
        print("ERROR: open_time column not found!")
        sys.exit(1)

    # Test historical analyzer loading
    print("\nTesting HistoricalAnalyzer...")
    from gui_historical import HistoricalAnalyzer

    analyzer = HistoricalAnalyzer()
    if analyzer.historical_data is not None:
        print("Historical data loaded successfully")
        print(f"Loaded {len(analyzer.historical_data)} rows")
    else:
        print("ERROR: Historical data failed to load!")
        sys.exit(1)

    # Test calculator loading
    print("\nTesting LadderCalculator...")
    from gui_calculator import LadderCalculator

    calculator = LadderCalculator()
    if calculator.historical_data is not None:
        print("Calculator historical data loaded successfully")
    else:
        print("WARNING: Calculator historical data is None (expected if no cache file)")

    print("\nAll tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
