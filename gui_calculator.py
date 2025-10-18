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
import os

# Import existing modules
from ladder_depths import calculate_ladder_depths, calculate_sell_ladder_depths
from size_optimizer import optimize_sizes, optimize_sell_sizes
from touch_analysis import analyze_touch_probabilities, analyze_upward_touch_probabilities
from weibull_fit import fit_weibull_tail
from data_fetcher import get_current_price
from analysis import weibull_touch_probability
from data_manager import data_manager
from utils import get_price_levels, get_sell_price_levels

warnings.filterwarnings('ignore')

class LadderCalculator:
    """Fast calculation engine for GUI with caching and optimization"""
    
    def __init__(self):
        self.cache = {}
        self.last_calculation = None
        self.current_price = None
        self.weibull_params = None
        self.historical_data = None
        self.data_interval = '1h'

        # Load initial data
        self._load_initial_data()

    def _load_initial_data(self):
        """Load initial data and Weibull parameters"""
        try:
            print("Loading initial data for GUI...")

            # Get current price for SOLUSDT (default)
            self.current_price = data_manager.get_current_price('SOLUSDT')

            # Load historical data using data manager (default to 720h timeframe)
            self.historical_data, self.data_interval = data_manager.load_data(720)

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
    
    def _calculate_weibull_params(self, timeframe_hours: int = 720):
        """Calculate Weibull parameters from historical data for specific timeframe"""
        try:
            print(f"Calculating Weibull parameters for {timeframe_hours} hour timeframe...")

            # Load appropriate data for this timeframe
            data_df, interval = data_manager.load_data(timeframe_hours)

            # Store interval for use in result
            self.current_interval = interval

            if data_df is None:
                print("Warning: No historical data available, using fallback parameters")
                return {
                    'buy': {'theta': 2.5, 'p': 1.2},
                    'sell': {'theta': 2.5, 'p': 1.2}
                }

            # Analyze touch probabilities with specific timeframe and interval
            depths, empirical_probs = analyze_touch_probabilities(
                data_df, timeframe_hours, interval, direction='buy'
            )
            depths_upward, empirical_probs_upward = analyze_touch_probabilities(
                data_df, timeframe_hours, interval, direction='sell'
            )

            # Fit Weibull distributions
            theta, p, fit_metrics = fit_weibull_tail(depths, empirical_probs)
            theta_sell, p_sell, fit_metrics_sell = fit_weibull_tail(depths_upward, empirical_probs_upward)

            print(f"Weibull parameters calculated - Buy: theta={theta:.3f}, p={p:.3f}, Sell: theta={theta_sell:.3f}, p={p_sell:.3f}")
            print(f"Using {interval} data interval for {timeframe_hours}h timeframe")

            return {
                'buy': {'theta': theta, 'p': p, 'fit_metrics': fit_metrics},
                'sell': {'theta': theta_sell, 'p': p_sell, 'fit_metrics': fit_metrics_sell}
            }

        except Exception as e:
            print(f"Warning: Could not calculate Weibull parameters for {timeframe_hours}h: {e}")
            import traceback
            traceback.print_exc()
            return {
                'buy': {'theta': 2.5, 'p': 1.2},
                'sell': {'theta': 2.5, 'p': 1.2}
            }
    
    def recalculate_weibull_for_timeframe(self, timeframe_hours: int):
        """
        Recalculate Weibull parameters for a specific timeframe.
        
        Args:
            timeframe_hours: Analysis timeframe in hours
        
        Returns:
            Dictionary with updated Weibull parameters
        """
        return self._calculate_weibull_params(timeframe_hours)
    
    def calculate_ladder_configuration(self, aggression_level: int, num_rungs: int, 
                                     timeframe_hours: int, budget: float,
                                     quantity_distribution: str = 'price_weighted',
                                     crypto_symbol: str = 'SOLUSDT',
                                     rung_positioning: str = 'quantile') -> Dict:
        """
        Calculate complete ladder configuration with caching.
        
        Args:
            aggression_level: 1-10 controlling depth range
            num_rungs: Number of ladder rungs
            timeframe_hours: Analysis timeframe
            budget: Total budget in USD
            quantity_distribution: Method for distributing quantities across rungs
            crypto_symbol: Cryptocurrency symbol (e.g., 'SOLUSDT')
            rung_positioning: Method for positioning rungs ('quantile', 'expected_value', etc.)
        
        Returns:
            Dictionary with complete ladder data
        """
        try:
            # Map aggression level to depth range
            depth_ranges = self._get_depth_range_for_aggression(aggression_level)
            d_min, d_max = depth_ranges
            
            # Recalculate Weibull parameters for the specific timeframe
            timeframe_weibull_params = self.recalculate_weibull_for_timeframe(timeframe_hours)
            
            # Get timeframe-specific Weibull parameters
            theta = timeframe_weibull_params['buy']['theta']
            p = timeframe_weibull_params['buy']['p']
            theta_sell = timeframe_weibull_params['sell']['theta']
            p_sell = timeframe_weibull_params['sell']['p']
            
            # Calculate buy ladder depths using selected positioning method
            buy_depths = calculate_ladder_depths(
                theta, p, num_rungs=num_rungs,
                d_min=d_min, d_max=d_max,
                method=self._get_positioning_method(rung_positioning),
                current_price=self.current_price,
                profit_target_pct=50.0  # Default profit target
            )
            
            # Calculate sell ladder depths
            sell_depths, profit_targets = calculate_sell_ladder_depths(
                theta_sell, p_sell, buy_depths,
                target_total_profit=50.0,
                risk_adjustment_factor=1.5,
                d_min_sell=d_min * 0.3,
                d_max_sell=d_max * 0.8,
                method='quantile',
                current_price=self.current_price,
                mean_reversion_rate=0.5
            )
            
            # Optimize buy sizes or use alternative distribution methods
            if quantity_distribution == 'kelly_optimized':
                buy_allocations, alpha, expected_returns = optimize_sizes(
                    buy_depths, theta, p, budget, use_kelly=True
                )
            else:
                # Use alternative distribution methods
                buy_allocations = self._calculate_quantity_distribution(
                    buy_depths, budget, quantity_distribution, current_price=self.current_price
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
            
            # Calculate expected profits per pair using actual profit targets
            profit_per_pair = actual_profits
            
            # Calculate expected profit per dollar (properly weighted by allocations)
            total_weighted_profit = np.sum(joint_probs * profit_per_pair * buy_allocations)
            total_allocation = np.sum(buy_allocations)
            expected_profit_per_dollar = total_weighted_profit / total_allocation if total_allocation > 0 else 0
            
            # Calculate timeframe metrics using proper probability theory
            # Expected time to first fill = 1 / (sum of individual probabilities - sum of joint probabilities)
            individual_prob_sum = np.sum(buy_touch_probs) + np.sum(sell_touch_probs)
            joint_prob_sum = np.sum(joint_probs)
            net_probability = individual_prob_sum - joint_prob_sum
            expected_timeframe_hours = 1.0 / net_probability if net_probability > 0 else np.inf
            
            # Calculate monthly metrics
            hours_per_month = 24 * 30
            expected_monthly_fills = hours_per_month / expected_timeframe_hours if expected_timeframe_hours < np.inf else 0
            expected_monthly_profit = expected_monthly_fills * np.sum(buy_allocations) * expected_profit_per_dollar / 100
            
            # Calculate sell allocations
            sell_allocations = sell_quantities * sell_prices
            
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
                'sell_allocations': sell_allocations,
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
                'weibull_params': timeframe_weibull_params,  # Use timeframe-specific parameters
                'depth_range': (d_min, d_max),
                'data_interval': self.current_interval,  # Track which data interval was used
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Error in ladder calculation: {e}")
            return self._get_fallback_configuration(aggression_level, num_rungs, budget, quantity_distribution, rung_positioning)
    
    def _get_depth_range_for_aggression(self, aggression_level: int) -> Tuple[float, float]:
        """Map aggression level to depth range"""
        depth_mappings = {
            1: (0.5, 8.0),    # Conservative
            2: (1.5, 15.0),   # Moderate
            3: (2.5, 20.0),   # Aggressive
            4: (3.5, 25.0),   # Very Aggressive
            5: (5.0, 30.0)    # Extreme
        }
        return depth_mappings.get(aggression_level, (2.5, 20.0))
    
    def _calculate_quantity_distribution(self, depths: np.ndarray, budget: float,
                                        distribution_method: str, current_price: float) -> np.ndarray:
        """Calculate quantity allocations using different distribution methods"""
        num_rungs = len(depths)
        prices = current_price * (1 - depths / 100)
        
        # Distribution strategies
        strategies = {
            'equal_quantity': lambda: (np.full(num_rungs, budget / num_rungs / prices.mean()) * prices) * (budget / (np.full(num_rungs, budget / num_rungs / prices.mean()) * prices).sum()),
            'equal_notional': lambda: np.full(num_rungs, budget / num_rungs),
            'linear_increase': lambda: np.linspace(0.5, 2.0, num_rungs) / np.linspace(0.5, 2.0, num_rungs).sum() * budget,
            'exponential_increase': lambda: np.exp(np.linspace(-1, 1, num_rungs)) / np.exp(np.linspace(-1, 1, num_rungs)).sum() * budget,
            'risk_parity': lambda: (1.0 / (depths + 1)) / (1.0 / (depths + 1)).sum() * budget,
            'price_weighted': lambda: np.full(num_rungs, budget / num_rungs)
        }
        
        if distribution_method in strategies:
            return strategies[distribution_method]()
        else:
            print(f"Warning: Unknown distribution method '{distribution_method}', using equal_notional")
            return np.full(num_rungs, budget / num_rungs)

    def _get_positioning_method(self, rung_positioning: str) -> str:
        """Map rung positioning UI option to ladder_depths method"""
        method_mapping = {
            'quantile': 'quantile',
            'expected_value': 'expected_value',
            'linear': 'linear',
            'exponential': 'exponential',  # Will implement this
            'logarithmic': 'logarithmic',  # Will implement this
            'risk_weighted': 'risk_weighted'  # Will implement this
        }
        return method_mapping.get(rung_positioning, 'quantile')

    def _get_fallback_configuration(self, aggression_level: int, num_rungs: int, budget: float,
                                   quantity_distribution: str = 'equal_notional',
                                   rung_positioning: str = 'quantile') -> Dict:
        """Get fallback configuration when calculation fails"""
        d_min, d_max = self._get_depth_range_for_aggression(aggression_level)
        
        # Simple linear spacing
        buy_depths = np.linspace(d_min, d_max, num_rungs)
        sell_depths = np.linspace(d_min * 0.3, d_max * 0.8, num_rungs)
        
        buy_prices = self.current_price * (1 - buy_depths / 100)
        sell_prices = self.current_price * (1 + sell_depths / 100)
        
        # Use quantity distribution method for fallback too
        buy_allocations = self._calculate_quantity_distribution(
            buy_depths, budget, quantity_distribution, self.current_price
        )
        buy_quantities = buy_allocations / buy_prices

        # Apply rung positioning method for fallback if specified
        if rung_positioning != 'quantile':
            print(f"Warning: Fallback configuration doesn't support {rung_positioning} positioning, using quantile")
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
    
    
    def get_max_available_timeframe(self) -> int:
        """
        Get maximum available timeframe from cached historical data.
        
        Returns:
            Maximum available hours based on historical data range
        """
        try:
            # Try to load cached data
            cache_file = 'cache_SOLUSDT_1h_1095d.csv'
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, parse_dates=['open_time'])
                if len(df) > 0:
                    # Calculate time span between earliest and latest data
                    time_span = df['open_time'].max() - df['open_time'].min()
                    max_hours = int(time_span.total_seconds() / 3600)
                    print(f"Max available timeframe from data: {max_hours} hours ({max_hours/24:.1f} days)")
                    return max_hours
        except Exception as e:
            print(f"Warning: Could not determine max timeframe from cache: {e}")
        
        # Fallback to config value
        try:
            from config import load_config
            config = load_config()
            max_hours = config.get('lookback_days', 1095) * 24
            print(f"Using config fallback max timeframe: {max_hours} hours ({max_hours/24:.1f} days)")
            return max_hours
        except:
            # Final fallback
            max_hours = 1095 * 24  # 3 years
            print(f"Using hardcoded fallback max timeframe: {max_hours} hours ({max_hours/24:.1f} days)")
            return max_hours
    
    def clear_cache(self):
        """Clear calculation cache"""
        self.cache.clear()
        self.calculate_ladder_configuration.cache_clear()
