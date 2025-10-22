#!/usr/bin/env python3
"""
Debug script to check ladder data structure
"""
from gui_calculator import LadderCalculator

def debug_ladder_data():
    calc = LadderCalculator()
    data = calc.calculate_ladder_configuration(
        aggression_level=3, 
        num_rungs=20, 
        timeframe_hours=720, 
        budget=50000, 
        quantity_distribution='kelly_optimized', 
        crypto_symbol='SOLUSDT'
    )
    
    print('Available keys in ladder data:')
    for key in sorted(data.keys()):
        print(f'  {key}')
    print()
    
    print('Sample values:')
    print(f'  current_price: {data.get("current_price", "N/A")}')
    print(f'  buy_allocations: {len(data.get("buy_allocations", []))} items')
    print(f'  total_buy_allocation: {data.get("total_buy_allocation", "N/A")}')
    
    # Check if buy_allocations exists and calculate total
    if 'buy_allocations' in data:
        total = sum(data['buy_allocations'])
        print(f'  Calculated total buy allocation: ${total:.2f}')

if __name__ == "__main__":
    debug_ladder_data()
