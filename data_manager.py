"""
Data Management Module
Handles loading and caching of historical data for different timeframes and intervals.
"""

import pandas as pd
import os
import numpy as np
from typing import Optional, Tuple
import requests
from datetime import datetime, timedelta
import time


class DataManager:
    """Manages historical data for different intervals and timeframes"""

    def __init__(self):
        self.data_cache = {}
        self.current_price = None

    def get_data_interval(self, timeframe_hours: int) -> str:
        """
        Determine the appropriate data interval based on timeframe.

        Args:
            timeframe_hours: Analysis timeframe in hours

        Returns:
            '1h' for timeframes ≤ 3 months (2160 hours), '1d' for longer timeframes
        """
        three_months_hours = 2160  # 3 months * 30 days * 24 hours

        if timeframe_hours <= three_months_hours:
            return '1h'
        else:
            return '1d'

    def get_cache_filename(self, interval: str, symbol: str = 'SOLUSDT') -> str:
        """Get the cache filename for a given interval and symbol"""
        if interval == '1h':
            return f'cache_{symbol}_1h_1095d.csv'
        elif interval == '1d':
            return f'cache_{symbol}_1d_10y.csv'
        else:
            raise ValueError(f"Unsupported interval: {interval}")

    def load_data(self, timeframe_hours: int, symbol: str = 'SOLUSDT') -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load historical data for the appropriate interval based on timeframe.

        Args:
            timeframe_hours: Analysis timeframe in hours
            symbol: Trading symbol

        Returns:
            Tuple of (data_df, interval_used)
        """
        interval = self.get_data_interval(timeframe_hours)
        cache_file = self.get_cache_filename(interval, symbol)

        # Check if data is already cached in memory
        cache_key = f"{interval}_{symbol}"
        if cache_key in self.data_cache:
            print(f"Using cached {interval} data for {symbol}")
            return self.data_cache[cache_key], interval

        # Try to load from cache file
        if os.path.exists(cache_file):
            try:
                print(f"Loading {interval} data from {cache_file}")
                df = pd.read_csv(cache_file)

                # Convert timestamp columns if present
                if 'open_time' in df.columns:
                    df['open_time'] = pd.to_datetime(df['open_time'])

                # Cache in memory
                self.data_cache[cache_key] = df
                print(f"Loaded {len(df)} {interval} candles from cache")
                return df, interval

            except Exception as e:
                print(f"Error loading cached {interval} data: {e}")

        # If cache doesn't exist or failed to load, fetch new data
        print(f"Cache file {cache_file} not found or invalid, fetching new {interval} data...")
        df = self.fetch_historical_data(interval, symbol, timeframe_hours)

        if df is not None:
            # Cache the data
            self.data_cache[cache_key] = df

            # Save to cache file for future use
            try:
                os.makedirs('cache', exist_ok=True)
                df.to_csv(cache_file, index=False)
                print(f"Saved {len(df)} {interval} candles to {cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache file: {e}")

        return df, interval

    def fetch_historical_data(self, interval: str, symbol: str, timeframe_hours: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Binance API.

        Args:
            interval: Data interval ('1h' or '1d')
            symbol: Trading symbol
            timeframe_hours: Timeframe for data fetching

        Returns:
            DataFrame with historical data or None if failed
        """
        try:
            # Import here to avoid circular imports
            from data_fetcher import fetch_solusdt_data

            print(f"Fetching {interval} data for {symbol}...")

            # Determine lookback period based on interval and timeframe
            if interval == '1h':
                # For 1h data, fetch enough for the timeframe plus buffer
                lookback_days = min(int(timeframe_hours / 24) + 30, 1095)  # Max 3 years for 1h
                df = fetch_solusdt_data(interval='1h', lookback_days=lookback_days)
            elif interval == '1d':
                # For 1d data, fetch enough for the timeframe
                lookback_years = min(int(timeframe_hours / (24 * 365)) + 1, 10)  # Max 10 years
                df = fetch_solusdt_data(interval='1d', lookback_days=lookback_years * 365)
            else:
                raise ValueError(f"Unsupported interval: {interval}")

            if df is not None and len(df) > 0:
                print(f"Successfully fetched {len(df)} {interval} candles")
                return df
            else:
                print(f"Failed to fetch {interval} data or no data available")
                return None

        except Exception as e:
            print(f"Error fetching {interval} data: {e}")
            return None

    def get_current_price(self, symbol: str = 'SOLUSDT') -> float:
        """Get current cryptocurrency price for the specified symbol"""
        try:
            from data_fetcher import get_current_price
            price = get_current_price(symbol)
            # Cache the most recent price
            self.current_price = price
            return price
        except Exception as e:
            print(f"Warning: Could not fetch current price for {symbol}: {e}")
            # Return cached price if available, otherwise fallback
            if self.current_price is not None:
                print(f"Using cached price: ${self.current_price:.2f}")
                return self.current_price
            else:
                print("No cached price available, using fallback: $100.00")
                return 100.0

    def clear_cache(self):
        """Clear in-memory data cache"""
        self.data_cache.clear()


# Global data manager instance
data_manager = DataManager()
