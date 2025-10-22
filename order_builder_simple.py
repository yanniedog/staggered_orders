"""
Simplified order builder.
"""
import pandas as pd
import numpy as np
from typing import Tuple

def build_orders(depths: np.ndarray, allocations: np.ndarray, current_price: float) -> pd.DataFrame:
    """
    Build order specifications from ladder depths and allocations.
    
    Args:
        depths: Array of ladder depths (percentages)
        allocations: Array of allocations (USD)
        current_price: Current market price
    
    Returns:
        DataFrame with order specifications
    """
    print(f"Building {len(depths)} orders...")
    
    orders = []
    
    for i, (depth, allocation) in enumerate(zip(depths, allocations)):
        # Calculate limit price
        limit_price = current_price * (1 - depth / 100)
        
        # Calculate quantity
        quantity = allocation / limit_price
        
        # Calculate profit percentage
        profit_pct = depth  # Simplified: profit equals depth
        
        order = {
            'rung': i + 1,
            'depth_pct': depth,
            'limit_price': limit_price,
            'quantity': quantity,
            'notional': allocation,
            'profit_pct': profit_pct
        }
        
        orders.append(order)
    
    orders_df = pd.DataFrame(orders)
    print(f"Built {len(orders_df)} orders")
    
    return orders_df

def export_orders_csv(orders_df: pd.DataFrame, filename: str) -> None:
    """
    Export orders to CSV file.
    
    Args:
        orders_df: DataFrame with orders
        filename: Output filename
    """
    print(f"Exporting orders to {filename}...")
    orders_df.to_csv(filename, index=False)
    print(f"Exported {len(orders_df)} orders")
