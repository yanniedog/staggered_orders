#!/usr/bin/env python3
"""
Debug script to test Weibull fitting.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from touch_analysis import analyze_touch_probabilities
from weibull_fit import fit_weibull_tail
from data_fetcher import fetch_solusdt_data
import numpy as np

# Load data
df = fetch_solusdt_data()
print(f"Loaded {len(df)} historical candles")

# Test direct Weibull calculation
print('Testing direct Weibull calculation...')
try:
    depths, probs = analyze_touch_probabilities(df, 24, '1h', direction='buy')
    print(f'Success! Got {len(depths)} depths and {len(probs)} probabilities')
    print(f'Sample: depth={depths[0]:.2f}%, prob={probs[0]:.4f}')

    # Now test Weibull fitting
    print('Testing Weibull fitting...')
    theta, p, metrics = fit_weibull_tail(depths, probs)
    print(f'Fit successful! theta={theta:.3f}, p={p:.3f}')
    print(f'R²: {metrics.get("r_squared", 0):.4f}')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
