"""
Simplified data fetcher for SOLUSDT from Binance API.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from config import load_config

def fetch_solusdt_data() -> pd.DataFrame:
    """
    Fetch SOLUSDT data from Binance API.
    
    Returns:
        DataFrame with OHLCV data
    """
    config = load_config()
    
    # Check for cached data
    cache_file = f"cache_{config['symbol']}_1h_{config['lookback_days']}d.csv"
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file)
        df['open_time'] = pd.to_datetime(df['open_time'])
        return df
    
    print(f"Fetching {config['symbol']} data from Binance...")
    
    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=config['lookback_days'])
    
    # Fetch data from Binance
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': config['symbol'],
        'interval': '1h',
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': 1000
    }
    
    all_data = []
    while True:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update start time for next batch
            params['startTime'] = data[-1][0] + 1
            
            # Rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    if not all_data:
        raise ValueError("No data fetched from Binance")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert data types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    # Keep only necessary columns
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Cache the data
    df.to_csv(cache_file, index=False)
    print(f"Cached data to {cache_file}")
    
    return df

def get_current_price(symbol: str = None) -> float:
    """
    Get current SOLUSDT price from Binance.
    
    Args:
        symbol: Symbol to get price for (ignored, uses config)
    
    Returns:
        Current price in USD
    """
    config = load_config()
    
    try:
        url = f"https://api.binance.com/api/v3/ticker/price"
        params = {'symbol': config['symbol']}
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        price = float(data['price'])
        
        return price
        
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return 100.0  # Default fallback price