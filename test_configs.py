#!/usr/bin/env python3
"""
Test different configurations for $50,000 SOL investment
"""
from gui_calculator import LadderCalculator
from gui_visualizations import VisualizationEngine
import json

def test_configurations():
    calc = LadderCalculator()
    ve = VisualizationEngine()

    # Test different configurations for $50,000 SOL
    configs = [
        {'name': 'Conservative', 'aggression': 2, 'rungs': 15, 'timeframe': 720, 'distribution': 'equal_notional'},
        {'name': 'Moderate', 'aggression': 3, 'rungs': 20, 'timeframe': 720, 'distribution': 'kelly_optimized'},
        {'name': 'Aggressive', 'aggression': 4, 'rungs': 25, 'timeframe': 720, 'distribution': 'kelly_optimized'},
        {'name': 'Short-term', 'aggression': 3, 'rungs': 20, 'timeframe': 168, 'distribution': 'linear_increase'},
        {'name': 'Long-term', 'aggression': 3, 'rungs': 20, 'timeframe': 2160, 'distribution': 'fibonacci_weighted'}
    ]

    budget = 50000
    print(f'Testing configurations for ${budget:,} SOL investment')
    print('=' * 60)

    results = []
    
    for config in configs:
        try:
            data = calc.calculate_ladder_configuration(
                aggression_level=config['aggression'],
                num_rungs=config['rungs'],
                timeframe_hours=config['timeframe'],
                budget=budget,
                quantity_distribution=config['distribution'],
                crypto_symbol='SOLUSDT'
            )
            
            result = {
                'name': config['name'],
                'aggression': config['aggression'],
                'rungs': config['rungs'],
                'timeframe': config['timeframe'],
                'distribution': config['distribution'],
                'current_price': data['current_price'],
                'total_buy_allocation': sum(data['buy_allocations']),
                'expected_monthly_profit': data.get('expected_monthly_profit', 0),
                'capital_efficiency': data.get('expected_profit_per_dollar', 0) * 100,
                'monthly_fills': data.get('expected_monthly_fills', 0)
            }
            results.append(result)
            
            print(f"{config['name']} Configuration:")
            print(f"  Aggression: {config['aggression']}, Rungs: {config['rungs']}, Timeframe: {config['timeframe']}h")
            print(f"  Current Price: ${data['current_price']:.2f}")
            print(f"  Total Buy Allocation: ${sum(data['buy_allocations']):.2f}")
            print(f"  Expected Monthly Profit: ${data.get('expected_monthly_profit', 0):.2f}")
            print(f"  Capital Efficiency: {data.get('expected_profit_per_dollar', 0) * 100:.2f}%")
            print(f"  Monthly Fills: {data.get('expected_monthly_fills', 0):.1f}")
            print()
            
        except Exception as e:
            print(f"{config['name']} Configuration FAILED: {e}")
            print()
    
    # Find best configuration
    if results:
        print("=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)
        
        # Best for profit
        best_profit = max(results, key=lambda x: x['expected_monthly_profit'])
        print(f"Highest Expected Profit: {best_profit['name']}")
        print(f"  Monthly Profit: ${best_profit['expected_monthly_profit']:.2f}")
        print(f"  Settings: Aggression {best_profit['aggression']}, {best_profit['rungs']} rungs, {best_profit['timeframe']}h")
        print()
        
        # Best for efficiency
        best_efficiency = max(results, key=lambda x: x['capital_efficiency'])
        print(f"Highest Capital Efficiency: {best_efficiency['name']}")
        print(f"  Efficiency: {best_efficiency['capital_efficiency']:.2f}%")
        print(f"  Settings: Aggression {best_efficiency['aggression']}, {best_efficiency['rungs']} rungs, {best_efficiency['timeframe']}h")
        print()
        
        # Most balanced
        balanced = max(results, key=lambda x: x['expected_monthly_profit'] * x['capital_efficiency'])
        print(f"Most Balanced (Profit × Efficiency): {balanced['name']}")
        print(f"  Monthly Profit: ${balanced['expected_monthly_profit']:.2f}")
        print(f"  Efficiency: {balanced['capital_efficiency']:.2f}%")
        print(f"  Settings: Aggression {balanced['aggression']}, {balanced['rungs']} rungs, {balanced['timeframe']}h")

if __name__ == "__main__":
    test_configurations()
