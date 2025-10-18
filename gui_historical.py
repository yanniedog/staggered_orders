"""
GUI Historical Analysis Module
Analyzes historical touch frequency data for visualization in the interactive GUI.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class HistoricalAnalyzer:
    """Analyzes historical data for touch frequency visualization"""
    
    def __init__(self):
        self.historical_data = None
        self.touch_cache = {}
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load historical data from cache file"""
        try:
            print("Loading historical data for touch frequency analysis...")
            self.historical_data = pd.read_csv('cache_SOLUSDT_1h_1095d.csv')
            
            # Convert timestamp columns
            if 'open_time' in self.historical_data.columns:
                self.historical_data['open_time'] = pd.to_datetime(self.historical_data['open_time'])
            
            print(f"Loaded {len(self.historical_data)} historical candles")
            
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            self.historical_data = None
    
    def analyze_touch_frequency(self, ladder_depths: np.ndarray, timeframe_hours: int, 
                              current_price: float) -> Dict:
        """
        Analyze historical touch frequency for given ladder depths.
        
        Args:
            ladder_depths: Array of depth percentages
            timeframe_hours: Analysis window in hours
            current_price: Current market price
        
        Returns:
            Dictionary with touch frequency data
        """
        if self.historical_data is None:
            return self._get_mock_touch_frequency(ladder_depths, timeframe_hours)
        
        try:
            # Calculate price levels for each depth
            price_levels = current_price * (1 - ladder_depths / 100)
            
            # Analyze touch frequency for each level
            touch_counts = []
            touch_dates = []
            
            for i, (depth, price_level) in enumerate(zip(ladder_depths, price_levels)):
                touches = self._count_price_touches(price_level, timeframe_hours)
                touch_counts.append(touches['count'])
                touch_dates.append(touches['dates'])
            
            # Calculate frequencies per day
            hours_per_day = 24
            days_in_timeframe = timeframe_hours / hours_per_day
            frequencies_per_day = np.array(touch_counts) / days_in_timeframe if days_in_timeframe > 0 else np.zeros_like(touch_counts)
            
            return {
                'depths': ladder_depths,
                'price_levels': price_levels,
                'touch_counts': touch_counts,
                'frequencies_per_day': frequencies_per_day,
                'timeframe_hours': timeframe_hours,
                'timeframe_days': days_in_timeframe,
                'touch_dates': touch_dates,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Error in touch frequency analysis: {e}")
            return self._get_mock_touch_frequency(ladder_depths, timeframe_hours)
    
    def _count_price_touches(self, price_level: float, timeframe_hours: int) -> Dict:
        """
        Count how many times price touched a specific level within timeframe.
        
        Args:
            price_level: Price level to check
            timeframe_hours: Analysis window in hours
        
        Returns:
            Dictionary with count and dates
        """
        try:
            # Get recent data within timeframe
            recent_data = self._get_recent_data(timeframe_hours)
            
            if recent_data is None or len(recent_data) == 0:
                return {'count': 0, 'dates': []}
            
            # Check for touches (price went below the level)
            touches = []
            
            for idx, row in recent_data.iterrows():
                # Check if low price touched or went below the level
                if row['low'] <= price_level:
                    touches.append(row['open_time'])
            
            return {
                'count': len(touches),
                'dates': touches
            }
            
        except Exception as e:
            print(f"Error counting price touches: {e}")
            return {'count': 0, 'dates': []}
    
    def _get_recent_data(self, timeframe_hours: int) -> Optional[pd.DataFrame]:
        """Get recent data within specified timeframe"""
        try:
            if self.historical_data is None:
                return None
            
            # Get the most recent data point
            latest_time = self.historical_data['open_time'].max()
            
            # Calculate start time
            start_time = latest_time - pd.Timedelta(hours=timeframe_hours)
            
            # Filter data
            recent_data = self.historical_data[
                self.historical_data['open_time'] >= start_time
            ].copy()
            
            return recent_data
            
        except Exception as e:
            print(f"Error getting recent data: {e}")
            return None
    
    def _get_mock_touch_frequency(self, ladder_depths: np.ndarray, timeframe_hours: int) -> Dict:
        """Generate mock touch frequency data for testing"""
        # Mock data based on depth - deeper levels touched less frequently
        base_frequency = 10.0  # Base touches per day
        depth_factor = np.exp(-ladder_depths / 5.0)  # Exponential decay with depth
        
        touch_counts = (base_frequency * depth_factor * timeframe_hours / 24).astype(int)
        frequencies_per_day = base_frequency * depth_factor
        
        return {
            'depths': ladder_depths,
            'price_levels': np.zeros_like(ladder_depths),  # Placeholder
            'touch_counts': touch_counts.tolist(),
            'frequencies_per_day': frequencies_per_day,
            'timeframe_hours': timeframe_hours,
            'timeframe_days': timeframe_hours / 24,
            'touch_dates': [[] for _ in ladder_depths],  # Empty lists
            'current_price': 100.0  # Placeholder
        }
    
    def analyze_touch_vs_time(self, ladder_depths: np.ndarray, timeframe_hours: int,
                            current_price: float) -> Dict:
        """
        Analyze cumulative touch probability over time for each depth level.
        
        Args:
            ladder_depths: Array of depth percentages
            timeframe_hours: Analysis window in hours
            current_price: Current market price
        
        Returns:
            Dictionary with time-series touch data
        """
        if self.historical_data is None:
            return self._get_mock_touch_vs_time(ladder_depths, timeframe_hours)
        
        try:
            # Get recent data
            recent_data = self._get_recent_data(timeframe_hours)
            
            if recent_data is None or len(recent_data) == 0:
                return self._get_mock_touch_vs_time(ladder_depths, timeframe_hours)
            
            # Calculate price levels
            price_levels = current_price * (1 - ladder_depths / 100)
            
            # Create time series for each depth
            time_series_data = {}
            
            for depth, price_level in zip(ladder_depths, price_levels):
                # Calculate cumulative touches over time
                cumulative_touches = []
                timestamps = []
                
                for idx, row in recent_data.iterrows():
                    timestamps.append(row['open_time'])
                    
                    # Count touches up to this point
                    touches_up_to_now = recent_data[
                        (recent_data['open_time'] <= row['open_time']) &
                        (recent_data['low'] <= price_level)
                    ]
                    
                    cumulative_touches.append(len(touches_up_to_now))
                
                time_series_data[f'depth_{depth:.1f}'] = {
                    'timestamps': timestamps,
                    'cumulative_touches': cumulative_touches,
                    'depth': depth,
                    'price_level': price_level
                }
            
            return {
                'time_series': time_series_data,
                'timeframe_hours': timeframe_hours,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Error in touch vs time analysis: {e}")
            return self._get_mock_touch_vs_time(ladder_depths, timeframe_hours)
    
    def _get_mock_touch_vs_time(self, ladder_depths: np.ndarray, timeframe_hours: int) -> Dict:
        """Generate mock touch vs time data"""
        # Create mock time series
        hours = np.arange(0, timeframe_hours, 1)
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(hours), freq='H')
        
        time_series_data = {}
        
        for depth in ladder_depths:
            # Mock cumulative touches with some randomness
            base_rate = 0.1 * np.exp(-depth / 5.0)  # Deeper = slower rate
            cumulative_touches = np.cumsum(np.random.poisson(base_rate, len(hours)))
            
            time_series_data[f'depth_{depth:.1f}'] = {
                'timestamps': timestamps.tolist(),
                'cumulative_touches': cumulative_touches.tolist(),
                'depth': depth,
                'price_level': 100.0 * (1 - depth / 100)  # Placeholder
            }
        
        return {
            'time_series': time_series_data,
            'timeframe_hours': timeframe_hours,
            'current_price': 100.0
        }
    
    def get_depth_statistics(self, ladder_depths: np.ndarray, timeframe_hours: int) -> Dict:
        """
        Get statistical summary for depth analysis.
        
        Args:
            ladder_depths: Array of depth percentages
            timeframe_hours: Analysis window in hours
        
        Returns:
            Dictionary with depth statistics
        """
        try:
            touch_data = self.analyze_touch_frequency(ladder_depths, timeframe_hours, 100.0)
            
            return {
                'min_depth': np.min(ladder_depths),
                'max_depth': np.max(ladder_depths),
                'avg_depth': np.mean(ladder_depths),
                'depth_range': np.max(ladder_depths) - np.min(ladder_depths),
                'total_touches': sum(touch_data['touch_counts']),
                'avg_touches_per_depth': np.mean(touch_data['touch_counts']),
                'timeframe_hours': timeframe_hours,
                'timeframe_days': timeframe_hours / 24
            }
            
        except Exception as e:
            print(f"Error getting depth statistics: {e}")
            return {
                'min_depth': 0.0,
                'max_depth': 0.0,
                'avg_depth': 0.0,
                'depth_range': 0.0,
                'total_touches': 0,
                'avg_touches_per_depth': 0.0,
                'timeframe_hours': timeframe_hours,
                'timeframe_days': timeframe_hours / 24
            }
    
    def clear_cache(self):
        """Clear analysis cache"""
        self.touch_cache.clear()
