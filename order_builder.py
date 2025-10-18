"""
Order specification builder with exchange constraints.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import yaml
from weibull_fit import weibull_tail


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def round_to_tick_size(price: float, tick_size: float) -> float:
    """
    Round price to exchange tick size.
    
    Args:
        price: Raw price
        tick_size: Exchange tick size
    
    Returns:
        Rounded price
    """
    # Simple and robust rounding
    return round(price / tick_size) * tick_size


def round_to_step_size(quantity: float, step_size: float) -> float:
    """
    Round quantity to exchange step size.
    
    Args:
        quantity: Raw quantity
        step_size: Exchange step size
    
    Returns:
        Rounded quantity
    """
    # Simple and robust rounding
    return round(quantity / step_size) * step_size


def calculate_allocations(depths: np.ndarray, current_price: float,
                         distribution_method: str) -> np.ndarray:
    """
    Calculate allocation amounts using different distribution methods.

    Args:
        depths: Array of depth percentages
        current_price: Current market price
        distribution_method: Distribution method to use

    Returns:
        Array of allocation amounts
    """
    # For now, use a simple budget-based calculation
    # In a real implementation, this would use the actual budget
    budget = 10000  # Default budget - should be passed as parameter
    num_rungs = len(depths)

    if distribution_method == 'equal_quantity':
        # Equal quantity per rung
        prices = current_price * (1 - depths / 100)
        equal_qty = budget / num_rungs / prices.mean()
        quantities = np.full(num_rungs, equal_qty)
        allocations = quantities * prices
        total_allocation = allocations.sum()
        allocations = allocations * (budget / total_allocation)
        return allocations

    elif distribution_method == 'equal_notional':
        # Equal notional value per rung
        return np.full(num_rungs, budget / num_rungs)

    elif distribution_method == 'linear_increase':
        # Linear increase from first to last rung
        weights = np.linspace(0.5, 2.0, num_rungs)
        weights = weights / weights.sum()
        return weights * budget

    elif distribution_method == 'exponential_increase':
        # Exponential increase
        weights = np.exp(np.linspace(-1, 1, num_rungs))
        weights = weights / weights.sum()
        return weights * budget

    elif distribution_method == 'risk_parity':
        # Risk-parity approach
        risk_weights = 1.0 / (depths + 1)
        risk_weights = risk_weights / risk_weights.sum()
        return risk_weights * budget

    else:
        # Default to equal notional
        return np.full(num_rungs, budget / num_rungs)


def calculate_order_specs(depths: np.ndarray, allocations: np.ndarray,
                        current_price: float, distribution_method: str = 'price_weighted') -> pd.DataFrame:
    """
    Calculate order specifications for ladder rungs.

    Args:
        depths: Array of ladder depths
        allocations: Array of size allocations (can be None for auto-calculation)
        current_price: Current market price
        distribution_method: Method for distributing quantities if allocations not provided

    Returns:
        DataFrame with order specifications
    """
    config = load_config()
    
    tick_size = config['tick_size']
    step_size = config['step_size']
    min_notional = config['min_notional']
    min_qty = config['min_qty']
    
    orders = []

    # If allocations not provided or distribution method specified, calculate them
    if allocations is None or distribution_method != 'price_weighted':
        allocations = calculate_allocations(depths, current_price, distribution_method)

    for i, (depth, allocation) in enumerate(zip(depths, allocations)):
        # Calculate limit price
        limit_price = current_price * (1 - depth / 100)
        limit_price = round_to_tick_size(limit_price, tick_size)

        # Calculate quantity
        quantity = allocation / limit_price
        quantity = round_to_step_size(quantity, step_size)
        
        # Calculate notional
        notional = limit_price * quantity
        
        # Check minimum constraints
        if notional < min_notional or quantity < min_qty:
            print(f"Warning: Rung {i+1} below minimum constraints")
            print(f"  Notional: ${notional:.2f} < ${min_notional}")
            print(f"  Quantity: {quantity:.3f} < {min_qty}")
            continue
        
        orders.append({
            'rung': i + 1,
            'depth_pct': depth,
            'limit_price': limit_price,
            'quantity': quantity,
            'notional': notional,
            'allocation': allocation
        })
    
    df = pd.DataFrame(orders)
    
    print(f"Generated {len(df)} valid orders:")
    print(f"  Price range: ${df['limit_price'].min():.2f} - ${df['limit_price'].max():.2f}")
    print(f"  Quantity range: {df['quantity'].min():.3f} - {df['quantity'].max():.3f}")
    print(f"  Total notional: ${df['notional'].sum():.2f}")
    
    return df


def validate_orders(df: pd.DataFrame) -> bool:
    """
    Validate order specifications against exchange rules.
    
    Args:
        df: DataFrame with order specifications
    
    Returns:
        True if all orders are valid
    """
    config = load_config()
    
    min_notional = config['min_notional']
    min_qty = config['min_qty']
    tick_size = config['tick_size']
    step_size = config['step_size']
    
    # Check minimum notional
    below_notional = df['notional'] < min_notional
    if below_notional.any():
        print(f"Error: {below_notional.sum()} orders below minimum notional")
        return False
    
    # Check minimum quantity
    below_qty = df['quantity'] < min_qty
    if below_qty.any():
        print(f"Error: {below_qty.sum()} orders below minimum quantity")
        return False
    
    # Check tick size compliance with proper tolerance
    # Prices should be multiples of tick_size
    price_ticks = (df['limit_price'] / tick_size).round(0)
    price_ticks_int = price_ticks.astype(int)
    # Use explicit tolerance: 0.1% relative tolerance or 1e-5 absolute tolerance
    if not np.allclose(price_ticks, price_ticks_int, rtol=1e-3, atol=1e-5):
        misaligned = ~np.isclose(price_ticks, price_ticks_int, rtol=1e-3, atol=1e-5)
        print(f"Error: {misaligned.sum()} prices not aligned with tick size {tick_size}")
        print(f"First few misaligned prices: {df['limit_price'][misaligned].head()}")
        return False
    
    # Check step size compliance with proper tolerance
    # Quantities should be multiples of step_size
    qty_steps = (df['quantity'] / step_size).round(0)
    qty_steps_int = qty_steps.astype(int)
    # Use explicit tolerance: 0.1% relative tolerance or 1e-5 absolute tolerance
    if not np.allclose(qty_steps, qty_steps_int, rtol=1e-3, atol=1e-5):
        misaligned = ~np.isclose(qty_steps, qty_steps_int, rtol=1e-3, atol=1e-5)
        print(f"Error: {misaligned.sum()} quantities not aligned with step size {step_size}")
        print(f"First few misaligned quantities: {df['quantity'][misaligned].head()}")
        return False
    
    # Check monotonicity
    if not df['limit_price'].is_monotonic_decreasing:
        print("Warning: Prices not monotonically decreasing")
    
    if not df['notional'].is_monotonic_increasing:
        print("Warning: Notionals not monotonically increasing")
    
    print("Order validation passed")
    return True


def export_orders_csv(df: pd.DataFrame, filename: str = "orders.csv") -> None:
    """
    Export orders to CSV file.
    
    Args:
        df: DataFrame with order specifications
        filename: Output filename
    """
    # Select columns for CSV export
    export_df = df[['rung', 'depth_pct', 'limit_price', 'quantity', 'notional']].copy()
    
    # Round for display
    export_df['depth_pct'] = export_df['depth_pct'].round(3)
    export_df['limit_price'] = export_df['limit_price'].round(2)
    export_df['quantity'] = export_df['quantity'].round(3)
    export_df['notional'] = export_df['notional'].round(2)
    
    export_df.to_csv(filename, index=False)
    print(f"Orders exported to {filename}")


def build_sell_orders(sell_depths: np.ndarray, sell_quantities: np.ndarray, 
                     current_price: float) -> pd.DataFrame:
    """
    Calculate sell order specifications for ladder rungs.
    
    Args:
        sell_depths: Array of sell depths (percentages above current price)
        sell_quantities: Array of sell quantities
        current_price: Current market price
    
    Returns:
        DataFrame with sell order specifications
    """
    config = load_config()
    
    tick_size = config['tick_size']
    step_size = config['step_size']
    min_notional = config['min_notional']
    min_qty = config['min_qty']
    
    sell_orders = []
    
    for i, (depth, quantity) in enumerate(zip(sell_depths, sell_quantities)):
        # Calculate limit price (above current price)
        limit_price = current_price * (1 + depth / 100)
        limit_price = round_to_tick_size(limit_price, tick_size)
        
        # Use provided quantity (already matched to buy quantity)
        quantity = round_to_step_size(quantity, step_size)
        
        # Calculate notional
        notional = limit_price * quantity
        
        # Check minimum constraints
        if notional < min_notional or quantity < min_qty:
            print(f"Warning: Sell rung {i+1} below minimum constraints")
            print(f"  Notional: ${notional:.2f} < ${min_notional}")
            print(f"  Quantity: {quantity:.3f} < {min_qty}")
            continue
        
        sell_orders.append({
            'rung': i + 1,
            'depth_pct': depth,
            'limit_price': limit_price,
            'quantity': quantity,
            'notional': notional
        })
    
    df = pd.DataFrame(sell_orders)
    
    print(f"Generated {len(df)} valid sell orders:")
    print(f"  Price range: ${df['limit_price'].min():.2f} - ${df['limit_price'].max():.2f}")
    print(f"  Quantity range: {df['quantity'].min():.3f} - {df['quantity'].max():.3f}")
    print(f"  Total notional: ${df['notional'].sum():.2f}")
    
    return df


def build_paired_orders(buy_depths: np.ndarray, buy_allocations: np.ndarray,
                       sell_depths: np.ndarray, sell_quantities: np.ndarray,
                       profit_targets: np.ndarray, current_price: float,
                       theta: float, p: float, theta_sell: float, p_sell: float,
                       max_analysis_hours: int) -> pd.DataFrame:
    """
    Build paired buy-sell orders with profit calculations.
    
    Args:
        buy_depths: Array of buy depths
        buy_allocations: Array of buy allocations
        sell_depths: Array of sell depths
        sell_quantities: Array of sell quantities
        profit_targets: Array of profit targets
        current_price: Current market price
        theta: Weibull shape parameter for buy-side
        p: Weibull scale parameter for buy-side
        theta_sell: Weibull shape parameter for sell-side
        p_sell: Weibull scale parameter for sell-side
        max_analysis_hours: Maximum analysis window in hours
    
    Returns:
        DataFrame with paired order specifications including expected profit
    """
    print("Building paired buy-sell orders...")
    
    # Build buy orders
    buy_orders_df = calculate_order_specs(buy_depths, buy_allocations, current_price)
    
    # Build sell orders
    sell_orders_df = build_sell_orders(sell_depths, sell_quantities, current_price)
    
    # Ensure same number of orders
    min_orders = min(len(buy_orders_df), len(sell_orders_df))
    buy_orders_df = buy_orders_df.iloc[:min_orders]
    sell_orders_df = sell_orders_df.iloc[:min_orders]
    
    # Create paired orders DataFrame
    paired_orders = []
    
    for i in range(min_orders):
        buy_order = buy_orders_df.iloc[i]
        sell_order = sell_orders_df.iloc[i]
        
        # Calculate profit metrics
        buy_notional = buy_order['notional']
        sell_notional = sell_order['notional']
        profit_usd = sell_notional - buy_notional
        profit_pct = (profit_usd / buy_notional) * 100
        
        # Calculate expected profit based on touch probabilities
        buy_depth = buy_order['depth_pct']
        sell_depth = sell_order['depth_pct']
        
        # Calculate touch probabilities using Weibull distribution
        buy_touch_prob = weibull_tail(np.array([buy_depth]), theta, p)[0]
        sell_touch_prob = weibull_tail(np.array([sell_depth]), theta_sell, p_sell)[0]
        
        # Joint probability (simplified: independent events)
        joint_prob = buy_touch_prob * sell_touch_prob
        
        # Expected profit = profit_usd * joint_probability
        expected_profit = profit_usd * joint_prob
        
        paired_orders.append({
            'rung': i + 1,
            'buy_depth_pct': buy_order['depth_pct'],
            'buy_price': buy_order['limit_price'],
            'buy_qty': buy_order['quantity'],
            'buy_notional': buy_notional,
            'sell_depth_pct': sell_order['depth_pct'],
            'sell_price': sell_order['limit_price'],
            'sell_qty': sell_order['quantity'],
            'sell_notional': sell_notional,
            'profit_pct': profit_pct,
            'profit_usd': profit_usd,
            'expected_profit': expected_profit,
            'target_profit_pct': profit_targets[i] if i < len(profit_targets) else 0.0
        })
    
    paired_df = pd.DataFrame(paired_orders)
    
    # Validate paired orders
    validate_paired_orders(paired_df)
    
    print(f"Paired order building complete: {len(paired_df)} pairs")
    print(f"  Total buy notional: ${paired_df['buy_notional'].sum():.2f}")
    print(f"  Total sell notional: ${paired_df['sell_notional'].sum():.2f}")
    print(f"  Total potential profit: ${paired_df['profit_usd'].sum():.2f}")
    print(f"  Total expected profit: ${paired_df['expected_profit'].sum():.2f}")
    print(f"  Average profit: {paired_df['profit_pct'].mean():.2f}%")
    
    return paired_df


def validate_paired_orders(df: pd.DataFrame) -> bool:
    """
    Validate paired order specifications.
    
    Args:
        df: DataFrame with paired order specifications
    
    Returns:
        True if all orders are valid
    """
    # Check quantity matching
    qty_mismatch = ~np.isclose(df['buy_qty'], df['sell_qty'], rtol=1e-6)
    if qty_mismatch.any():
        print(f"Warning: {qty_mismatch.sum()} pairs have quantity mismatches")
    
    # Check profitability - CRITICAL: Stop execution if unprofitable pairs found
    unprofitable = df['profit_pct'] <= 0
    if unprofitable.any():
        print(f"ERROR: {unprofitable.sum()} pairs are unprofitable!")
        print("Unprofitable pairs:")
        for idx, row in df[unprofitable].iterrows():
            print(f"  Rung {row['rung']}: Buy ${row['buy_notional']:.2f} @ ${row['buy_price']:.2f} -> "
                  f"Sell ${row['sell_notional']:.2f} @ ${row['sell_price']:.2f} = {row['profit_pct']:.2f}%")
        raise ValueError(f"Found {unprofitable.sum()} unprofitable pairs. Fix ladder configuration before proceeding.")
    
    # Check price ordering
    if not df['buy_price'].is_monotonic_decreasing:
        print("Warning: Buy prices not monotonically decreasing")
    
    if not df['sell_price'].is_monotonic_increasing:
        print("Warning: Sell prices not monotonically increasing")
    
    # Check profit distribution
    profit_ratio = df['profit_pct'].max() / df['profit_pct'].min()
    if profit_ratio > 10:
        print(f"Warning: Very high profit ratio: {profit_ratio:.2f}")
    
    print("Paired order validation passed - all pairs are profitable")
    return True


def export_paired_orders_csv(df: pd.DataFrame, filename: str = "paired_orders.csv") -> None:
    """
    Export paired orders to CSV file.
    
    Args:
        df: DataFrame with paired order specifications
        filename: Output filename
    """
    # Select columns for CSV export
    export_df = df[['rung', 'buy_depth_pct', 'buy_price', 'buy_qty', 'buy_notional',
                   'sell_depth_pct', 'sell_price', 'sell_qty', 'sell_notional',
                   'profit_pct', 'profit_usd', 'expected_profit', 'target_profit_pct']].copy()
    
    # Round for display
    export_df['buy_depth_pct'] = export_df['buy_depth_pct'].round(3)
    export_df['buy_price'] = export_df['buy_price'].round(2)
    export_df['buy_qty'] = export_df['buy_qty'].round(3)
    export_df['buy_notional'] = export_df['buy_notional'].round(2)
    export_df['sell_depth_pct'] = export_df['sell_depth_pct'].round(3)
    export_df['sell_price'] = export_df['sell_price'].round(2)
    export_df['sell_qty'] = export_df['sell_qty'].round(3)
    export_df['sell_notional'] = export_df['sell_notional'].round(2)
    export_df['profit_pct'] = export_df['profit_pct'].round(2)
    export_df['profit_usd'] = export_df['profit_usd'].round(2)
    export_df['expected_profit'] = export_df['expected_profit'].round(2)
    export_df['target_profit_pct'] = export_df['target_profit_pct'].round(2)
    
    export_df.to_csv(filename, index=False)
    print(f"Paired orders exported to {filename}")


def build_orders(depths: np.ndarray, allocations: np.ndarray, 
                current_price: float) -> pd.DataFrame:
    """
    Complete order building process.
    
    Args:
        depths: Array of ladder depths
        allocations: Array of size allocations
        current_price: Current market price
    
    Returns:
        DataFrame with order specifications
    """
    print("Building order specifications...")
    
    # Calculate order specs
    df = calculate_order_specs(depths, allocations, current_price)
    
    # Validate orders
    if not validate_orders(df):
        raise ValueError("Order validation failed")
    
    # Export to CSV
    export_orders_csv(df)
    
    print(f"Order building complete: {len(df)} orders")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    allocations = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
    current_price = 100.0
    
    df = build_orders(depths, allocations, current_price)
    
    print("\nOrder specifications:")
    print(df.to_string(index=False))
