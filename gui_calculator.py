"""
GUI Calculation Engine
Fast computation module for real-time updates in the interactive GUI.
Wraps existing modules for efficient recalculation with caching.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from functools import lru_cache
import time
import warnings

# Import existing modules
from ladder_depths import calculate_ladder_depths, calculate_sell_ladder_depths
from size_optimizer import optimize_sizes, optimize_sell_sizes
from touch_analysis import analyze_touch_probabilities, analyze_upward_touch_probabilities
from weibull_fit import fit_weibull_tail
from data_fetcher import get_current_price
from analysis import weibull_touch_probability

warnings.filterwarnings('ignore')

class LadderCalculator:
    """Fast calculation engine for GUI with caching and optimization"""
    
    def __init__(self):
        self.cache = {}
        self.last_calculation = None
        self.current_price = None
        self.weibull_params = None
        self.historical_data = None
        
        # Load initial data
        self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial data and Weibull parameters"""
        try:
            print("Loading initial data for GUI...")
            
            # Get current price
            self.current_price = get_current_price()
            
            # Load historical data (this would be cached from main.py)
            # For now, we'll use placeholder data structure
            self.historical_data = self._load_cached_data()
            
            # Calculate initial Weibull parameters
            self.weibull_params = self._calculate_weibull_params()
            
            print("Initial data loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load initial data: {e}")
            self.current_price = 100.0  # Fallback
            self.weibull_params = {
                'buy': {'theta': 2.5, 'p': 1.2},
                'sell': {'theta': 2.5, 'p': 1.2}
            }
    
    def _load_cached_data(self):
        """Load cached historical data"""
        try:
            # Try to load from cache file
            import pandas as pd
            df = pd.read_csv('cache_SOLUSDT_1h_1095d.csv')
            return df
        except:
            # Return placeholder data structure
            return None
    
    def _calculate_weibull_params(self):
        """Calculate Weibull parameters from historical data"""
        if self.historical_data is None:
            return {
                'buy': {'theta': 2.5, 'p': 1.2},
                'sell': {'theta': 2.5, 'p': 1.2}
            }
        
        try:
            # Analyze touch probabilities
            depths, empirical_probs = analyze_touch_probabilities(
                self.historical_data, 720, '1h'
            )
            depths_upward, empirical_probs_upward = analyze_upward_touch_probabilities(
                self.historical_data, 720, '1h'
            )
            
            # Fit Weibull distributions
            theta, p, fit_metrics = fit_weibull_tail(depths, empirical_probs)
            theta_sell, p_sell, fit_metrics_sell = fit_weibull_tail(depths_upward, empirical_probs_upward)
            
            return {
                'buy': {'theta': theta, 'p': p, 'fit_metrics': fit_metrics},
                'sell': {'theta': theta_sell, 'p': p_sell, 'fit_metrics': fit_metrics_sell}
            }
            
        except Exception as e:
            print(f"Warning: Could not calculate Weibull parameters: {e}")
            return {
                'buy': {'theta': 2.5, 'p': 1.2},
                'sell': {'theta': 2.5, 'p': 1.2}
            }
    
    @lru_cache(maxsize=100)
    def calculate_ladder_configuration(self, aggression_level: int, num_rungs: int, 
                                     timeframe_hours: int, budget: float) -> Dict:
        """
        Calculate complete ladder configuration with caching.
        
        Args:
            aggression_level: 1-10 controlling depth range
            num_rungs: Number of ladder rungs
            timeframe_hours: Analysis timeframe
            budget: Total budget in USD
        
        Returns:
            Dictionary with complete ladder data
        """
        try:
            # Map aggression level to depth range
            depth_ranges = self._get_depth_range_for_aggression(aggression_level)
            d_min, d_max = depth_ranges
            
            # Get Weibull parameters
            theta = self.weibull_params['buy']['theta']
            p = self.weibull_params['buy']['p']
            theta_sell = self.weibull_params['sell']['theta']
            p_sell = self.weibull_params['sell']['p']
            
            # Calculate buy ladder depths
            buy_depths = calculate_ladder_depths(
                theta, p, num_rungs=num_rungs,
                d_min=d_min, d_max=d_max,
                method='expected_value',
                current_price=self.current_price,
                profit_target_pct=50.0  # Default profit target
            )
            
            # Calculate sell ladder depths
            sell_depths, profit_targets = calculate_sell_ladder_depths(
                theta_sell, p_sell, buy_depths,
                profit_target_pct=50.0,
                risk_adjustment_factor=1.5,
                d_min_sell=d_min * 0.3,
                d_max_sell=d_max * 0.8,
                method='quantile',
                current_price=self.current_price,
                mean_reversion_rate=0.5
            )
            
            # Optimize buy sizes
            buy_allocations, alpha, expected_returns = optimize_sizes(
                buy_depths, theta, p, budget, use_kelly=True
            )
            
            # Calculate buy quantities and prices
            buy_prices = self.current_price * (1 - buy_depths / 100)
            buy_quantities = buy_allocations / buy_prices
            
            # Optimize sell sizes
            sell_prices = self.current_price * (1 + sell_depths / 100)
            sell_quantities, actual_profits, alpha_sell = optimize_sell_sizes(
                buy_quantities, buy_prices, sell_depths, sell_prices,
                profit_targets, theta_sell, p_sell, independent_optimization=True
            )
            
            # Calculate touch probabilities
            buy_touch_probs = np.array([
                weibull_touch_probability(d, theta, p) for d in buy_depths
            ])
            sell_touch_probs = np.array([
                weibull_touch_probability(d, theta_sell, p_sell) for d in sell_depths
            ])
            
            # Calculate joint probabilities
            joint_probs = buy_touch_probs * sell_touch_probs
            
            # Calculate expected profits per pair
            profit_per_pair = (sell_prices - buy_prices) / buy_prices * 100
            
            # Calculate expected profit per dollar
            expected_profit_per_dollar = np.sum(joint_probs * profit_per_pair * buy_quantities) / np.sum(buy_allocations)
            
            # Calculate timeframe metrics
            avg_joint_prob = np.mean(joint_probs)
            expected_timeframe_hours = 1.0 / avg_joint_prob if avg_joint_prob > 0 else np.inf
            
            # Calculate monthly metrics
            hours_per_month = 24 * 30
            expected_monthly_fills = hours_per_month / expected_timeframe_hours if expected_timeframe_hours < np.inf else 0
            expected_monthly_profit = expected_monthly_fills * np.sum(buy_allocations) * expected_profit_per_dollar / 100
            
            # Create comprehensive result
            result = {
                'aggression_level': aggression_level,
                'num_rungs': num_rungs,
                'timeframe_hours': timeframe_hours,
                'budget': budget,
                'current_price': self.current_price,
                'buy_depths': buy_depths,
                'sell_depths': sell_depths,
                'buy_prices': buy_prices,
                'sell_prices': sell_prices,
                'buy_allocations': buy_allocations,
                'sell_quantities': sell_quantities,
                'buy_quantities': buy_quantities,
                'buy_touch_probs': buy_touch_probs,
                'sell_touch_probs': sell_touch_probs,
                'joint_probs': joint_probs,
                'profit_per_pair': profit_per_pair,
                'actual_profits': actual_profits,
                'profit_targets': profit_targets,
                'expected_profit_per_dollar': expected_profit_per_dollar,
                'expected_timeframe_hours': expected_timeframe_hours,
                'expected_monthly_fills': expected_monthly_fills,
                'expected_monthly_profit': expected_monthly_profit,
                'weibull_params': self.weibull_params,
                'depth_range': (d_min, d_max),
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Error in ladder calculation: {e}")
            return self._get_fallback_configuration(aggression_level, num_rungs, budget)
    
    def _get_depth_range_for_aggression(self, aggression_level: int) -> Tuple[float, float]:
        """Map aggression level to depth range"""
        depth_mappings = {
            1: (0.5, 5.0),    # Conservative
            2: (1.0, 8.0),    # Conservative+
            3: (1.5, 12.0),   # Moderate
            4: (2.0, 15.0),   # Moderate+
            5: (2.5, 18.0),   # Aggressive
            6: (3.0, 20.0),   # Aggressive+
            7: (3.5, 22.0),   # Very Aggressive
            8: (4.0, 25.0),   # Very Aggressive+
            9: (4.5, 28.0),   # Extreme
            10: (5.0, 30.0)   # Extreme+
        }
        return depth_mappings.get(aggression_level, (2.5, 18.0))
    
    def _get_fallback_configuration(self, aggression_level: int, num_rungs: int, budget: float) -> Dict:
        """Get fallback configuration when calculation fails"""
        d_min, d_max = self._get_depth_range_for_aggression(aggression_level)
        
        # Simple linear spacing
        buy_depths = np.linspace(d_min, d_max, num_rungs)
        sell_depths = np.linspace(d_min * 0.3, d_max * 0.8, num_rungs)
        
        buy_prices = self.current_price * (1 - buy_depths / 100)
        sell_prices = self.current_price * (1 + sell_depths / 100)
        
        # Equal allocations
        buy_allocations = np.full(num_rungs, budget / num_rungs)
        buy_quantities = buy_allocations / buy_prices
        sell_quantities = buy_quantities  # Simplified
        
        return {
            'aggression_level': aggression_level,
            'num_rungs': num_rungs,
            'budget': budget,
            'current_price': self.current_price,
            'buy_depths': buy_depths,
            'sell_depths': sell_depths,
            'buy_prices': buy_prices,
            'sell_prices': sell_prices,
            'buy_allocations': buy_allocations,
            'sell_quantities': sell_quantities,
            'buy_quantities': buy_quantities,
            'buy_touch_probs': np.full(num_rungs, 0.1),
            'sell_touch_probs': np.full(num_rungs, 0.1),
            'joint_probs': np.full(num_rungs, 0.01),
            'profit_per_pair': (sell_prices - buy_prices) / buy_prices * 100,
            'actual_profits': np.full(num_rungs, 5.0),
            'profit_targets': np.full(num_rungs, 5.0),
            'expected_profit_per_dollar': 0.05,
            'expected_timeframe_hours': 100.0,
            'expected_monthly_fills': 7.2,
            'expected_monthly_profit': budget * 0.05,
            'weibull_params': self.weibull_params,
            'depth_range': (d_min, d_max),
            'timestamp': time.time()
        }
    
    def calculate_kpis(self, ladder_data: Dict) -> Dict:
        """Calculate key performance indicators"""
        try:
            total_profit = ladder_data['expected_monthly_profit']
            monthly_fills = ladder_data['expected_monthly_fills']
            capital_efficiency = total_profit / ladder_data['budget'] * 100
            timeframe = ladder_data['expected_timeframe_hours']
            
            return {
                'total_profit': f"${total_profit:,.0f}",
                'monthly_fills': f"{monthly_fills:.1f}",
                'capital_efficiency': f"{capital_efficiency:.1f}%",
                'timeframe': f"{timeframe:.0f}h"
            }
        except Exception as e:
            return {
                'total_profit': "N/A",
                'monthly_fills': "N/A", 
                'capital_efficiency': "N/A",
                'timeframe': "N/A"
            }
    
    def get_historical_touch_frequency(self, ladder_data: Dict, timeframe_hours: int) -> Dict:
        """Get historical touch frequency data for visualization"""
        try:
            if self.historical_data is None:
                return self._get_mock_touch_frequency(ladder_data)
            
            # This would analyze historical data for actual touch frequencies
            # For now, return mock data based on Weibull probabilities
            buy_depths = ladder_data['buy_depths']
            buy_touch_probs = ladder_data['buy_touch_probs']
            
            # Convert probabilities to expected frequencies
            # Assuming 1 hour bars, timeframe_hours gives us the window
            expected_touches = buy_touch_probs * timeframe_hours
            
            return {
                'depths': buy_depths,
                'frequencies': expected_touches,
                'timeframe_hours': timeframe_hours
            }
            
        except Exception as e:
            print(f"Error calculating touch frequency: {e}")
            return self._get_mock_touch_frequency(ladder_data)
    
    def _get_mock_touch_frequency(self, ladder_data: Dict) -> Dict:
        """Get mock touch frequency data"""
        buy_depths = ladder_data['buy_depths']
        buy_touch_probs = ladder_data['buy_touch_probs']
        
        # Mock frequency calculation
        frequencies = buy_touch_probs * 100  # Scale for visualization
        
        return {
            'depths': buy_depths,
            'frequencies': frequencies,
            'timeframe_hours': ladder_data['timeframe_hours']
        }
    
    def clear_cache(self):
        """Clear calculation cache"""
        self.cache.clear()
        self.calculate_ladder_configuration.cache_clear()
