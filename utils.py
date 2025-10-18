"""
Common utility functions used across multiple modules.
Eliminates code duplication for price/depth conversions and formatting.
"""
import numpy as np


def depth_to_price(depth: float, current_price: float) -> float:
    """Convert depth percentage to actual price"""
    return current_price * (1 - depth / 100)


def price_to_depth(price: float, current_price: float) -> float:
    """Convert actual price to depth percentage"""
    return (current_price - price) / current_price * 100


def get_price_levels(depths: np.ndarray, current_price: float) -> np.ndarray:
    """Convert array of depths to price levels"""
    return current_price * (1 - depths / 100)


def get_sell_price_levels(sell_depths: np.ndarray, current_price: float) -> np.ndarray:
    """Convert array of sell depths to price levels"""
    return current_price * (1 + sell_depths / 100)


def format_price_label(price: float) -> str:
    """Format price for display labels"""
    return f"${price:.2f}"


def format_timeframe(timeframe_hours: int) -> str:
    """Format timeframe hours into human-readable display"""
    if timeframe_hours < 24:
        return f"{timeframe_hours}h"
    elif timeframe_hours < 168:
        days = timeframe_hours // 24
        return f"{days}d"
    elif timeframe_hours < 8760:
        weeks = timeframe_hours // 168
        return f"{weeks}w"
    else:
        years = timeframe_hours // 8760
        return f"{years}y"

