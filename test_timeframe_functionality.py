#!/usr/bin/env python3
"""
Test script to verify timeframe functionality works correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui_calculator import LadderCalculator
from touch_analysis import analyze_touch_probabilities
from data_fetcher import fetch_solusdt_data

def test_timeframe_functionality():
    """Test that timeframe affects probability calculations"""
    print("Testing timeframe functionality...")

    # Load historical data
    try:
        df = fetch_solusdt_data()
        print(f"Loaded {len(df)} historical candles")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # Initialize calculator
    calculator = LadderCalculator()
    calculator.historical_data = df

    # Test different timeframes
    timeframes_to_test = [24, 168, 720]  # 1 day, 1 week, 1 month

    results = {}

    for timeframe_hours in timeframes_to_test:
        print(f"\nTesting {timeframe_hours}h timeframe...")

        # Calculate ladder configuration (this will internally recalculate Weibull parameters)
        ladder_data = calculator.calculate_ladder_configuration(
            aggression_level=5,
            num_rungs=10,
            timeframe_hours=timeframe_hours,
            budget=10000.0
        )

        # Get the timeframe-specific Weibull parameters from the result
        weibull_params = ladder_data['weibull_params']

        # Store key metrics
        results[timeframe_hours] = {
            'buy_theta': weibull_params['buy']['theta'],
            'buy_p': weibull_params['buy']['p'],
            'sell_theta': weibull_params['sell']['theta'],
            'sell_p': weibull_params['sell']['p'],
            'touch_probabilities': ladder_data['buy_touch_probs'],
            'expected_profit_per_dollar': ladder_data['expected_profit_per_dollar'],
            'expected_timeframe_hours': ladder_data['expected_timeframe_hours']
        }

        print(f"  Buy Weibull: theta={weibull_params['buy']['theta']:.3f}, p={weibull_params['buy']['p']:.3f}")
        print(f"  Sell Weibull: theta={weibull_params['sell']['theta']:.3f}, p={weibull_params['sell']['p']:.3f}")
        print(f"  Touch probabilities: {ladder_data['buy_touch_probs'][:3]}...")
        print(f"  Expected profit per dollar: {ladder_data['expected_profit_per_dollar']:.4f}")
        print(f"  Expected timeframe: {ladder_data['expected_timeframe_hours']:.1f}h")

    # Check that different timeframes produce different results
    print("\n=== VERIFICATION ===")

    # Check that Weibull parameters change with timeframe
    buy_thetas = [results[tf]['buy_theta'] for tf in timeframes_to_test]
    buy_ps = [results[tf]['buy_p'] for tf in timeframes_to_test]

    print(f"Buy theta values: {buy_thetas}")
    print(f"Buy p values: {buy_ps}")

    # Parameters should be different for different timeframes
    theta_changed = len(set([round(t, 3) for t in buy_thetas])) > 1
    p_changed = len(set([round(p, 3) for p in buy_ps])) > 1

    print(f"Theta parameters changed: {theta_changed}")
    print(f"P parameters changed: {p_changed}")

    if theta_changed and p_changed:
        print("SUCCESS: Timeframe affects Weibull parameters correctly")
    else:
        print("FAILURE: Timeframe does not affect Weibull parameters")
        return False

    # Check that touch probabilities change with timeframe
    prob_sets = [results[tf]['touch_probabilities'] for tf in timeframes_to_test]
    probs_changed = any(
        not all(abs(a - b) < 0.001 for a, b in zip(prob_set1, prob_set2))
        for i, prob_set1 in enumerate(prob_sets)
        for j, prob_set2 in enumerate(prob_sets)
        if i != j
    )

    print(f"Touch probabilities changed: {probs_changed}")

    if probs_changed:
        print("SUCCESS: Timeframe affects touch probabilities correctly")
    else:
        print("FAILURE: Timeframe does not affect touch probabilities")
        return False

    # Check that expected profit per dollar changes appropriately
    profit_per_dollar = [results[tf]['expected_profit_per_dollar'] for tf in timeframes_to_test]
    print(f"Profit per dollar values: {[round(p, 4) for p in profit_per_dollar]}")

    # Longer timeframes should generally have higher probabilities and potentially different profitability
    if len(set([round(p, 4) for p in profit_per_dollar])) > 1:
        print("SUCCESS: Timeframe affects expected profitability correctly")
    else:
        print("WARNING: Timeframe may not be affecting profitability as expected")

    print("\nAll tests passed! Timeframe functionality is working correctly.")
    return True

if __name__ == "__main__":
    success = test_timeframe_functionality()
    sys.exit(0 if success else 1)
