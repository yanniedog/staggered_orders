"""
Data fetcher for SOLUSDT from Binance API with local caching.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import yaml
from typing import Tuple, Optional


def load_config() -> dict:
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    """
    Fetch kline data from Binance API with optimized pagination for large datasets.
    
    Args:
        symbol: Trading pair (e.g., 'SOLUSDT')
        interval: Kline interval (e.g., '1m')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
    
    Returns:
        DataFrame with OHLCV data
    """
    url = "https://api.binance.com/api/v3/klines"
    
    all_data = []
    current_start = start_time
    batch_count = 0
    
    print(f"Fetching {symbol} data in batches...")
    
    # Calculate total expected batches for progress
    total_time_ms = end_time - start_time
    if interval == '1h':
        candles_per_batch = 1000
        ms_per_candle = 60 * 60 * 1000  # 1 hour in ms
    elif interval == '1m':
        candles_per_batch = 1000
        ms_per_candle = 60 * 1000  # 1 minute in ms
    elif interval == '1d':
        candles_per_batch = 1000
        ms_per_candle = 24 * 60 * 60 * 1000  # 1 day in ms
    else:
        candles_per_batch = 1000
        ms_per_candle = 60 * 60 * 1000  # Default to 1 hour
    
    expected_batches = int(np.ceil(total_time_ms / (candles_per_batch * ms_per_candle)))
    print(f"Expected batches: ~{expected_batches}")
    
    # Validate time range
    if total_time_ms <= 0:
        raise ValueError(f"Invalid time range: start_time={start_time}, end_time={end_time}")
    
    # Check if time range is too large (more than 3 years)
    max_time_ms = 3 * 365 * 24 * 60 * 60 * 1000  # 3 years in ms
    if total_time_ms > max_time_ms:
        print(f"Warning: Requesting {total_time_ms / (365 * 24 * 60 * 60 * 1000):.1f} years of data")
        print("This may take a very long time and could hit API limits")
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while current_start < end_time:
        batch_count += 1
        if batch_count % 10 == 0:
            progress_pct = min(100, (batch_count / expected_batches) * 100)
            print(f"  Batch {batch_count}/{expected_batches} ({progress_pct:.1f}%)...")
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        retry_count = 0
        backoff_seconds = 1
        batch_success = False
        
        while retry_count < max_retries and not batch_success:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                klines = response.json()
                
                # Validate response format
                if not isinstance(klines, list):
                    raise ValueError(f"Invalid response format: expected list, got {type(klines)}")
                
                if not klines:
                    print(f"  No data returned for batch {batch_count}, stopping")
                    break
                
                # Validate kline format (should have 12 fields)
                if len(klines[0]) != 12:
                    raise ValueError(f"Invalid kline format: expected 12 fields, got {len(klines[0])}")
                
                all_data.extend(klines)
                
                # Update start_time for next batch
                current_start = klines[-1][6] + 1  # Close time + 1ms
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.1)
                
                # Success - reset consecutive failures
                consecutive_failures = 0
                batch_success = True
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Error: {consecutive_failures} consecutive failures")
                    print("Aborting data fetch due to repeated API failures")
                    raise
                
                if retry_count >= max_retries:
                    print(f"Error fetching batch {batch_count} after {max_retries} retries: {e}")
                    print("Aborting data fetch due to repeated failures")
                    raise
                else:
                    print(f"Error fetching batch {batch_count} (attempt {retry_count}/{max_retries}): {e}")
                    print(f"Retrying in {backoff_seconds} seconds...")
                    import time
                    time.sleep(backoff_seconds)
                    # Exponential backoff: double the wait time for next retry
                    backoff_seconds *= 2
            
            except Exception as e:
                print(f"Unexpected error in batch {batch_count}: {e}")
                raise
    
    if not all_data:
        raise ValueError("No data retrieved from Binance API")
    
    print(f"Fetched {len(all_data)} candles in {batch_count} batches")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert to proper types with error handling
    try:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check for any NaN values after conversion
        nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
        if nan_counts.any():
            print(f"Warning: Found NaN values after conversion: {nan_counts.to_dict()}")
            # Remove rows with NaN values
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            print(f"Removed {nan_counts.sum()} rows with NaN values")
        
        # Validate price data integrity
        invalid_rows = (df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)
        if invalid_rows.any():
            print(f"Warning: Found {invalid_rows.sum()} rows with invalid price data")
            df = df[~invalid_rows]
            print(f"Removed {invalid_rows.sum()} rows with invalid prices")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        raise
    
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


def get_cache_filename(symbol: str, interval: str, lookback_days: int) -> str:
    """Generate cache filename for data"""
    return f"cache_{symbol}_{interval}_{lookback_days}d.csv"


def is_cache_valid(filename: str, max_age_hours: int) -> bool:
    """Check if cached data is still valid"""
    if not os.path.exists(filename):
        return False
    
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filename))
    return file_age.total_seconds() / 3600 < max_age_hours


def fetch_solusdt_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch SOLUSDT 1h data with caching and graceful fallback.
    
    Args:
        force_refresh: Force refresh even if cache exists
    
    Returns:
        DataFrame with OHLCV data
    """
    config = load_config()
    
    symbol = config['symbol']
    lookback_days = config['lookback_days']
    cache_hours = config['cache_hours']
    interval = '1h'  # Hardcoded to 1h
    
    cache_file = get_cache_filename(symbol, interval, lookback_days)
    
    # Check cache first
    if not force_refresh and config['cache_data'] and is_cache_valid(cache_file, cache_hours):
        print(f"Loading cached data from {cache_file}")
        try:
            df = pd.read_csv(cache_file, parse_dates=['open_time'])
            
            # Validate cached data integrity
            if len(df) == 0:
                print("Warning: Cached data is empty, will refetch")
            elif df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
                print("Warning: Cached data contains NaN values, will refetch")
            elif (df['high'] < df['low']).any() or (df['open'] <= 0).any() or (df['close'] <= 0).any():
                print("Warning: Cached data contains invalid prices, will refetch")
            else:
                print(f"Cached data validated: {len(df)} candles from {df['open_time'].min()} to {df['open_time'].max()}")
                return df
        except Exception as e:
            print(f"Error loading cached data: {e}")
            print("Will refetch data from API")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    print(f"Fetching {symbol} data from {start_time} to {end_time}")
    print(f"Interval: {interval}, Lookback: {lookback_days} days")
    
    # Estimate total candles for progress reporting
    estimated_candles = lookback_days * 24
    print(f"Estimated candles to fetch: ~{estimated_candles:,}")
    
    # Try to fetch data with fallback for shorter periods if needed
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Adjust lookback period if this is a retry
            if attempt > 0:
                fallback_days = max(90, lookback_days // (2 ** attempt))  # Minimum 90 days
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
            
            # Validate fetched data
            if len(df) == 0:
                raise ValueError("No data returned from API")
            
            # Check data quality
            nan_counts = df[['open', 'high', 'low', 'close', 'volume']].isna().sum()
            if nan_counts.any():
                print(f"Warning: API returned data with NaN values: {nan_counts.to_dict()}")
            
            invalid_prices = (df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)
            if invalid_prices.any():
                print(f"Warning: API returned {invalid_prices.sum()} rows with invalid prices")
            
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
            import time
            time.sleep(2 ** attempt)  # Exponential backoff between attempts


def get_current_price() -> float:
    """Get current SOLUSDT price from Binance with error handling"""
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {'symbol': 'SOLUSDT'}
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response format
            if 'price' not in data:
                raise ValueError(f"Invalid response format: missing 'price' field")
            
            price = float(data['price'])
            
            # Validate price is reasonable
            if price <= 0:
                raise ValueError(f"Invalid price: {price}")
            
            if price > 10000:  # SOLUSDT shouldn't be > $10,000
                raise ValueError(f"Unrealistic price: {price}")
            
            print(f"Current SOLUSDT price: ${price:.2f}")
            return price
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error fetching current price after {max_retries} retries: {e}")
                print("Using fallback price estimation from cached data")
                return get_fallback_price()
            else:
                print(f"Error fetching current price (attempt {retry_count}/{max_retries}): {e}")
                import time
                time.sleep(1)
        
        except Exception as e:
            print(f"Unexpected error fetching current price: {e}")
            return get_fallback_price()
    
    return get_fallback_price()


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
