"""
Streamlined data fetcher for SOLUSDT from Binance API with decorator-based retry logic.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import functools
from typing import Tuple, Optional, Callable, Any
from config import load_config


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      exponential: bool = True, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        exponential: Whether to use exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = base_delay * (2 ** attempt) if exponential else base_delay
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
            
            print(f"All {max_retries + 1} attempts failed")
            raise last_exception
        
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(requests.exceptions.RequestException,))
def _make_api_request(url: str, params: dict, timeout: int = 30) -> dict:
    """Make API request with retry logic."""
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """Fetch kline data from Binance API with optimized pagination"""
    if end_time <= start_time:
        raise ValueError(f"Invalid time range: start_time={start_time}, end_time={end_time}")
    
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start = start_time
    batch_count = 0
    
    print(f"Fetching {symbol} data...")
    
    while current_start < end_time:
        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}...")
        
        params = {'symbol': symbol, 'interval': interval, 'startTime': current_start, 'endTime': end_time, 'limit': 1000}
        
        try:
            klines = _make_api_request(url, params)
            if not klines or not isinstance(klines, list):
                break
            
            all_data.extend(klines)
            current_start = klines[-1][6] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    if not all_data:
        raise ValueError("No data retrieved")
    
    print(f"Fetched {len(all_data)} candles in {batch_count} batches")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                         'taker_buy_quote', 'ignore'])
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df[~((df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0))]
    
    if len(df) == 0:
        raise ValueError("No valid data after cleaning")
    
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


def get_cache_filename(symbol: str, interval: str, lookback_days: int) -> str:
    """Generate cache filename for data"""
    return f"cache_{symbol}_{interval}_{lookback_days}d.csv"


def is_cache_valid(filename: str, max_age_hours: int) -> bool:
    """Check if cached data is still valid"""
    return os.path.exists(filename) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))).total_seconds() / 3600 < max_age_hours


def fetch_solusdt_data(interval: str = '1h', lookback_days: int = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch SOLUSDT data with caching and graceful fallback.

    Args:
        interval: Kline interval ('1h', '1d', etc.)
        lookback_days: Number of days to look back (uses config default if None)
        force_refresh: Force refresh even if cache exists

    Returns:
        DataFrame with OHLCV data
    """
    config = load_config()
    symbol = config['symbol']
    if lookback_days is None:
        lookback_days = config['lookback_days']
    cache_hours = config['cache_hours']

    cache_file = get_cache_filename(symbol, interval, lookback_days)
    
    # Check cache first
    if not force_refresh and config['cache_data'] and is_cache_valid(cache_file, cache_hours):
        print(f"Loading cached data from {cache_file}")
        try:
            df = pd.read_csv(cache_file, parse_dates=['open_time'])
            
            # Validate cached data
            if len(df) == 0 or df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
                print("Warning: Cached data invalid, will refetch")
            elif (df['high'] < df['low']).any() or (df['open'] <= 0).any() or (df['close'] <= 0).any():
                print("Warning: Cached data contains invalid prices, will refetch")
            else:
                print(f"Cached data validated: {len(df)} candles from {df['open_time'].min()} to {df['open_time'].max()}")
                return df
        except Exception as e:
            print(f"Error loading cached data: {e}")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    print(f"Fetching {symbol} data from {start_time} to {end_time}")
    print(f"Interval: {interval}, Lookback: {lookback_days} days")
    
    # Try to fetch data with fallback for shorter periods
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Adjust lookback period if this is a retry
            if attempt > 0:
                fallback_days = max(90, lookback_days // (2 ** attempt))
                fallback_start_time = end_time - timedelta(days=fallback_days)
                print(f"Retry attempt {attempt + 1}: Trying shorter period ({fallback_days} days)")
            else:
                fallback_start_time = start_time
            
            # Fetch data
            df = fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=int(fallback_start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000)
            )
            
            if len(df) == 0:
                raise ValueError("No data returned from API")
            
            print(f"Successfully fetched {len(df)} candles")
            print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            # Cache the data
            if config['cache_data']:
                df.to_csv(cache_file, index=False)
                print(f"Data cached to {cache_file}")
            
            return df
            
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            
            if attempt >= max_attempts:
                print("All fetch attempts failed")
                raise
            
            print(f"Retrying with shorter time period...")
            time.sleep(2 ** attempt)


@retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(requests.exceptions.RequestException,))
def get_current_price(symbol: str = 'SOLUSDT') -> float:
    """Get current cryptocurrency price from Binance"""
    data = _make_api_request("https://api.binance.com/api/v3/ticker/price", {'symbol': symbol}, timeout=10)
    
    if 'price' not in data:
        raise ValueError("Invalid response: missing 'price'")
    
    price = float(data['price'])
    if price <= 0 or (symbol in ['BTCUSDT', 'ETHUSDT'] and price > 100000) or (symbol not in ['BTCUSDT', 'ETHUSDT'] and price > 10000):
        raise ValueError(f"Invalid price: {price}")
    
    print(f"Current {symbol} price: ${price:.2f}")
    return price


def get_fallback_price() -> float:
    """Get fallback price from cached data or reasonable default"""
    try:
        # Try to get price from cached data
        cache_file = get_cache_filename("SOLUSDT", "1h", 1095)
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['open_time'])
            if len(df) > 0:
                latest_price = df['close'].iloc[-1]
                print(f"Using latest cached price: ${latest_price:.2f}")
                return float(latest_price)
    except Exception as e:
        print(f"Could not get price from cache: {e}")
    
    # Use reasonable default for SOLUSDT
    default_price = 180.0
    print(f"Using default price: ${default_price:.2f}")
    print("Warning: Using default price - results may not reflect current market conditions")
    return default_price


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_solusdt_data()
    print(f"Fetched {len(df)} candles")
    print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    current_price = get_current_price()
    print(f"Current SOLUSDT price: ${current_price:.2f}")
