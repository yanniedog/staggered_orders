#!/usr/bin/env python3
"""
Realistic analysis for $50,000 SOL investment
"""
from gui_calculator import LadderCalculator
import numpy as np

def analyze_realistic_configs():
    calc = LadderCalculator()
    
    # More realistic configurations for $50,000 SOL
    configs = [
        {'name': 'Conservative', 'aggression': 2, 'rungs': 10, 'timeframe': 720, 'distribution': 'equal_notional'},
        {'name': 'Moderate', 'aggression': 3, 'rungs': 15, 'timeframe': 720, 'distribution': 'kelly_optimized'},
        {'name': 'Balanced', 'aggression': 3, 'rungs': 20, 'timeframe': 720, 'distribution': 'linear_increase'},
        {'name': 'Aggressive', 'aggression': 4, 'rungs': 20, 'timeframe': 720, 'distribution': 'kelly_optimized'},
        {'name': 'Short-term', 'aggression': 3, 'rungs': 15, 'timeframe': 168, 'distribution': 'equal_notional'},
        {'name': 'Long-term', 'aggression': 3, 'rungs': 25, 'timeframe': 2160, 'distribution': 'fibonacci_weighted'}
    ]

    budget = 50000
    print(f'REALISTIC ANALYSIS for ${budget:,} SOL investment')
    print('=' * 70)
    print('Current SOL Price: ~$185')
    print('=' * 70)

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
            
            # Calculate realistic metrics
            total_buy_allocation = sum(data['buy_allocations'])
            current_price = data['current_price']
            total_sol_quantity = total_buy_allocation / current_price
            
            # Calculate average allocation per rung
            avg_allocation = total_buy_allocation / len(data['buy_allocations'])
            
            # Calculate price range
            buy_prices = data['buy_prices']
            min_price = min(buy_prices)
            max_price = max(buy_prices)
            price_range_pct = ((max_price - min_price) / current_price) * 100
            
            result = {
                'name': config['name'],
                'aggression': config['aggression'],
                'rungs': config['rungs'],
                'timeframe': config['timeframe'],
                'distribution': config['distribution'],
                'current_price': current_price,
                'total_buy_allocation': total_buy_allocation,
                'total_sol_quantity': total_sol_quantity,
                'avg_allocation_per_rung': avg_allocation,
                'min_price': min_price,
                'max_price': max_price,
                'price_range_pct': price_range_pct,
                'monthly_fills': data.get('expected_monthly_fills', 0)
            }
            results.append(result)
            
            print(f"{config['name']} Configuration:")
            print(f"  Settings: Aggression {config['aggression']}, {config['rungs']} rungs, {config['timeframe']}h")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Total SOL Quantity: {total_sol_quantity:.2f} SOL")
            print(f"  Price Range: ${min_price:.2f} - ${max_price:.2f} ({price_range_pct:.1f}% range)")
            print(f"  Avg Allocation per Rung: ${avg_allocation:.0f}")
            print(f"  Expected Monthly Fills: {data.get('expected_monthly_fills', 0):.1f}")
            print()
            
        except Exception as e:
            print(f"{config['name']} Configuration FAILED: {e}")
            print()
    
    # Provide recommendations
    if results:
        print("=" * 70)
        print("RECOMMENDATIONS FOR $50,000 SOL:")
        print("=" * 70)
        
        # Best for risk management (conservative)
        conservative = next((r for r in results if r['name'] == 'Conservative'), None)
        if conservative:
            print("CONSERVATIVE APPROACH:")
            print(f"   • Aggression: {conservative['aggression']}, {conservative['rungs']} rungs")
            print(f"   • Price Range: {conservative['price_range_pct']:.1f}% (${conservative['min_price']:.2f} - ${conservative['max_price']:.2f})")
            print(f"   • Best for: Risk-averse investors, stable returns")
            print()
        
        # Best for balanced approach
        balanced = next((r for r in results if r['name'] == 'Balanced'), None)
        if balanced:
            print("BALANCED APPROACH:")
            print(f"   • Aggression: {balanced['aggression']}, {balanced['rungs']} rungs")
            print(f"   • Price Range: {balanced['price_range_pct']:.1f}% (${balanced['min_price']:.2f} - ${balanced['max_price']:.2f})")
            print(f"   • Best for: Most investors, good risk/reward balance")
            print()
        
        # Best for aggressive growth
        aggressive = next((r for r in results if r['name'] == 'Aggressive'), None)
        if aggressive:
            print("AGGRESSIVE APPROACH:")
            print(f"   • Aggression: {aggressive['aggression']}, {aggressive['rungs']} rungs")
            print(f"   • Price Range: {aggressive['price_range_pct']:.1f}% (${aggressive['min_price']:.2f} - ${aggressive['max_price']:.2f})")
            print(f"   • Best for: Growth-focused investors, higher risk tolerance")
            print()
        
        print("KEY INSIGHTS:")
        print("   • All configurations allocate the full $50,000 budget")
        print("   • More rungs = better price coverage but smaller individual orders")
        print("   • Higher aggression = deeper price levels (more risk, more potential reward)")
        print("   • Kelly Optimized distribution allocates more to higher-probability rungs")
        print("   • Equal Notional provides consistent dollar amounts per rung")

if __name__ == "__main__":
    analyze_realistic_configs()
