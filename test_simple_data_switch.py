#!/usr/bin/env python3
"""Simple test for timeframe-based data switching"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

try:
    # Change to the correct directory
    os.chdir(r'C:\code\staggered_orders')

    print("Testing simple data switching...")

    from data_manager import data_manager

    # Test 1h data loading
    print("\n=== Testing 1h Data Loading ===")
    data_1h, interval_1h = data_manager.load_data(720)
    print(f"1h data: {len(data_1h) if data_1h is not None else 0} candles, interval: {interval_1h}")

    # Test 1d data loading
    print("\n=== Testing 1d Data Loading ===")
    data_1d, interval_1d = data_manager.load_data(8760)
    print(f"1d data: {len(data_1d) if data_1d is not None else 0} candles, interval: {interval_1d}")

    print("\nSimple test completed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
