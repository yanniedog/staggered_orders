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
from gui_strategies import QuantityDistributionFactory, RungPositioningFactory

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
        """Load initial data and Weibull parameters with optimizations"""
        try:
            print("Loading initial data for GUI...")

            # Get current price for SOLUSDT (default) - cache this
            if not hasattr(self, '_current_price_cache') or self._current_price_cache is None:
                self.current_price = data_manager.get_current_price('SOLUSDT')
                self._current_price_cache = self.current_price
            else:
                self.current_price = self._current_price_cache

            # Load historical data using data manager (default to 720h timeframe) - cache this too
            cache_key = f"720_SOLUSDT"
            if not hasattr(self, '_data_cache') or cache_key not in self._data_cache:
                self.historical_data, self.data_interval = data_manager.load_data(720)
                if not hasattr(self, '_data_cache'):
                    self._data_cache = {}
                self._data_cache[cache_key] = (self.historical_data, self.data_interval)
            else:
                self.historical_data, self.data_interval = self._data_cache[cache_key]

            # Calculate initial Weibull parameters - also cache these
            if not hasattr(self, '_weibull_cache') or '720' not in self._weibull_cache:
                self.weibull_params = self._calculate_weibull_params()
                if not hasattr(self, '_weibull_cache'):
                    self._weibull_cache = {}
                self._weibull_cache['720'] = self.weibull_params
            else:
                self.weibull_params = self._weibull_cache['720']

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
            # Get current price for the selected cryptocurrency
            current_price = data_manager.get_current_price(crypto_symbol)
            
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
            positioning_method = self._get_positioning_method(rung_positioning)
            
            # Handle advanced positioning methods
            if positioning_method in ['support_resistance', 'volume_profile', 'touch_pattern', 
                                     'adaptive_probability', 'fibonacci', 'dynamic_density']:
                buy_depths = self._calculate_advanced_positioning(
                    positioning_method, num_rungs, d_min, d_max, theta, p, 
                    current_price, timeframe_hours
                )
            else:
                # Use standard ladder_depths methods
                buy_depths = calculate_ladder_depths(
                    theta, p, num_rungs=num_rungs,
                    d_min=d_min, d_max=d_max,
                    method=positioning_method,
                    current_price=current_price,
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
                current_price=current_price,
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
                    buy_depths, budget, quantity_distribution, current_price=current_price
                )
            
            # Calculate buy quantities and prices
            buy_prices = current_price * (1 - buy_depths / 100)
            buy_quantities = buy_allocations / buy_prices
            
            # Optimize sell sizes
            sell_prices = current_price * (1 + sell_depths / 100)
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
                'crypto_symbol': crypto_symbol,
                'current_price': current_price,
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
        """Calculate quantity allocations using strategy pattern"""
        strategy = QuantityDistributionFactory.create_strategy(distribution_method)
        return strategy.calculate(depths, budget, current_price, self.weibull_params)
    
    def _adaptive_kelly_allocation(self, depths: np.ndarray, touch_probs: np.ndarray, 
                                   budget: float, prices: np.ndarray) -> np.ndarray:
        """Adaptive Kelly criterion with risk adjustment"""
        # Expected returns based on depth (more depth = more potential return)
        expected_returns = depths / 10.0  # Simplified return model
        
        # Kelly fraction with safety factor
        kelly_fractions = (touch_probs * expected_returns) / (expected_returns + 1)
        kelly_fractions = np.clip(kelly_fractions, 0.05, 0.25)  # Safety limits
        
        # Normalize to budget
        allocations = kelly_fractions / kelly_fractions.sum() * budget
        return allocations
    
    def _volatility_weighted_allocation(self, depths: np.ndarray, budget: float, 
                                       prices: np.ndarray) -> np.ndarray:
        """Allocate based on inverse volatility (more stable = more allocation)"""
        # Use depth as proxy for volatility risk
        volatility_proxy = depths
        inverse_vol = 1.0 / (volatility_proxy + 1)
        weights = inverse_vol / inverse_vol.sum()
        return weights * budget
    
    def _sharpe_maximizing_allocation(self, depths: np.ndarray, touch_probs: np.ndarray,
                                      budget: float, prices: np.ndarray) -> np.ndarray:
        """Maximize Sharpe ratio across rungs"""
        expected_returns = depths / 10.0  # Potential return
        risks = 1.0 / (touch_probs + 0.01)  # Risk as inverse of touch probability
        
        sharpe_ratios = expected_returns / risks
        sharpe_ratios = np.clip(sharpe_ratios, 0, np.percentile(sharpe_ratios, 90))
        
        weights = sharpe_ratios / sharpe_ratios.sum()
        return weights * budget
    
    def _fibonacci_weighted_allocation(self, num_rungs: int, budget: float) -> np.ndarray:
        """Allocate using Fibonacci sequence weighting"""
        # Generate Fibonacci sequence
        if num_rungs == 1:
            return np.array([budget])
        elif num_rungs == 2:
            fib_weights = np.array([1.0, 1.0])
        else:
            # Generate Fibonacci numbers
            fib_sequence = [1, 1]
            for i in range(2, num_rungs):
                fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
            fib_weights = np.array(fib_sequence)
        
        # Option 1: Allocate more to deeper levels (higher Fibonacci numbers)
        # Option 2: Allocate more to shallower levels (reverse)
        # Using forward sequence (more allocation to deeper rungs - more aggressive)
        
        # Normalize to budget
        allocations = (fib_weights / fib_weights.sum()) * budget
        
        print(f"Fibonacci allocation: Min=${allocations.min():.2f}, Max=${allocations.max():.2f}, Ratio={allocations.max()/allocations.min():.2f}x")
        return allocations
    
    def _calculate_advanced_positioning(self, method: str, num_rungs: int, d_min: float, 
                                       d_max: float, theta: float, p: float,
                                       current_price: float, timeframe_hours: int) -> np.ndarray:
        """Calculate advanced rung positioning using strategy pattern"""
        strategy = RungPositioningFactory.create_strategy(method)
        weibull_params = {'buy': {'theta': theta, 'p': p}} if theta and p else None
        return strategy.calculate(num_rungs, d_min, d_max, current_price, timeframe_hours, weibull_params)
    
    def _support_resistance_positioning(self, num_rungs: int, d_min: float, d_max: float,
                                       current_price: float) -> np.ndarray:
        """Position rungs near support/resistance levels based on historical price clusters"""
        if self.historical_data is None or len(self.historical_data) < 100:
            print("Warning: Insufficient data for support/resistance analysis, using linear")
            return np.linspace(d_min, d_max, num_rungs)
        
        try:
            # Get recent price data
            recent_prices = self.historical_data['close'].values[-2000:]  # Last 2000 candles
            
            # Calculate price levels as percentages below current
            price_depths = (1 - recent_prices / current_price) * 100
            price_depths = price_depths[(price_depths >= d_min) & (price_depths <= d_max)]
            
            if len(price_depths) < 10:
                return np.linspace(d_min, d_max, num_rungs)
            
            # Find density peaks (support levels) using histogram
            hist, bin_edges = np.histogram(price_depths, bins=50)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find local maxima (support zones)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, prominence=np.percentile(hist, 60))
            support_levels = bin_centers[peaks]
            
            if len(support_levels) < num_rungs:
                # Supplement with linear spacing
                linear_depths = np.linspace(d_min, d_max, num_rungs - len(support_levels))
                all_depths = np.concatenate([support_levels, linear_depths])
                depths = np.sort(all_depths)[:num_rungs]
            else:
                # Use top support levels
                support_strengths = hist[peaks]
                top_indices = np.argsort(support_strengths)[-num_rungs:]
                depths = np.sort(support_levels[top_indices])
            
            print(f"Generated {num_rungs} rungs using support/resistance clustering")
            return depths
            
        except Exception as e:
            print(f"Warning: Support/resistance calculation failed: {e}, using linear")
            return np.linspace(d_min, d_max, num_rungs)
    
    def _volume_profile_positioning(self, num_rungs: int, d_min: float, d_max: float,
                                   current_price: float) -> np.ndarray:
        """Position rungs weighted by volume profile"""
        if self.historical_data is None or 'volume' not in self.historical_data.columns:
            print("Warning: No volume data available, using linear")
            return np.linspace(d_min, d_max, num_rungs)
        
        try:
            recent_data = self.historical_data.tail(2000)
            prices = recent_data['close'].values
            volumes = recent_data['volume'].values
            
            # Calculate depth from current price
            depths = (1 - prices / current_price) * 100
            valid_mask = (depths >= d_min) & (depths <= d_max)
            
            if not np.any(valid_mask):
                return np.linspace(d_min, d_max, num_rungs)
            
            valid_depths = depths[valid_mask]
            valid_volumes = volumes[valid_mask]
            
            # Create volume-weighted distribution
            hist, bin_edges = np.histogram(valid_depths, bins=num_rungs * 2, weights=valid_volumes)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Select top volume bins
            top_indices = np.argsort(hist)[-num_rungs:]
            selected_depths = np.sort(bin_centers[top_indices])
            
            print(f"Generated {num_rungs} rungs using volume profile weighting")
            return selected_depths
            
        except Exception as e:
            print(f"Warning: Volume profile calculation failed: {e}, using linear")
            return np.linspace(d_min, d_max, num_rungs)
    
    def _touch_pattern_positioning(self, num_rungs: int, d_min: float, d_max: float,
                                   theta: float, p: float, current_price: float) -> np.ndarray:
        """Position rungs based on historical touch patterns"""
        if self.historical_data is None or len(self.historical_data) < 500:
            print("Warning: Insufficient data for touch pattern analysis, using linear")
            return np.linspace(d_min, d_max, num_rungs)
        
        try:
            # Analyze where price actually touched historically
            lows = self.historical_data['low'].values[-2000:]
            current_prices_hist = self.historical_data['close'].values[-2000:]
            
            # Calculate actual touches (where price went down and came back)
            touch_depths = []
            for i in range(1, len(lows) - 1):
                if lows[i] < current_prices_hist[i]:
                    depth_pct = (1 - lows[i] / current_prices_hist[i]) * 100
                    if d_min <= depth_pct <= d_max:
                        touch_depths.append(depth_pct)
            
            if len(touch_depths) < 20:
                return np.linspace(d_min, d_max, num_rungs)
            
            # Cluster touch points
            touch_depths = np.array(touch_depths)
            hist, bin_edges = np.histogram(touch_depths, bins=num_rungs * 3)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Weight by frequency and Weibull probability
            weibull_probs = np.array([weibull_touch_probability(d, theta, p) for d in bin_centers])
            combined_score = hist * weibull_probs
            
            # Select top scoring depths
            top_indices = np.argsort(combined_score)[-num_rungs:]
            depths = np.sort(bin_centers[top_indices])
            
            print(f"Generated {num_rungs} rungs using touch pattern analysis")
            return depths
            
        except Exception as e:
            print(f"Warning: Touch pattern analysis failed: {e}, using linear")
            return np.linspace(d_min, d_max, num_rungs)
    
    def _adaptive_probability_positioning(self, num_rungs: int, d_min: float, d_max: float,
                                         theta: float, p: float) -> np.ndarray:
        """Position rungs to maintain equal probability spacing"""
        # Generate probability quantiles
        prob_quantiles = np.linspace(0.05, 0.95, num_rungs)
        
        # Convert to depths (inverse of CDF sampling)
        depths = []
        for q in prob_quantiles:
            # Find depth that gives this touch probability
            # Use binary search
            d_test = np.linspace(d_min, d_max, 1000)
            probs = np.array([weibull_touch_probability(d, theta, p) for d in d_test])
            idx = np.argmin(np.abs(probs - q))
            depths.append(d_test[idx])
        
        depths = np.array(depths)
        print(f"Generated {num_rungs} rungs with adaptive probability spacing")
        return depths
    
    def _fibonacci_positioning(self, num_rungs: int, d_min: float, d_max: float,
                              current_price: float) -> np.ndarray:
        """Position rungs at Fibonacci retracement levels"""
        # Standard Fibonacci ratios
        fib_ratios = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618])
        
        # Map to depth range
        depth_range = d_max - d_min
        fib_depths = d_min + fib_ratios * depth_range / fib_ratios.max()
        fib_depths = fib_depths[fib_depths <= d_max]
        
        if len(fib_depths) >= num_rungs:
            # Use subset of Fibonacci levels
            indices = np.linspace(0, len(fib_depths) - 1, num_rungs, dtype=int)
            depths = fib_depths[indices]
        else:
            # Supplement with linear spacing
            extra_needed = num_rungs - len(fib_depths)
            linear_depths = np.linspace(d_min, d_max, extra_needed)
            depths = np.sort(np.concatenate([fib_depths, linear_depths]))[:num_rungs]
        
        print(f"Generated {num_rungs} rungs using Fibonacci levels")
        return depths
    
    def _dynamic_density_positioning(self, num_rungs: int, d_min: float, d_max: float,
                                    theta: float, p: float) -> np.ndarray:
        """Position rungs with density proportional to touch probability"""
        # Generate fine grid
        fine_grid = np.linspace(d_min, d_max, 1000)
        touch_probs = np.array([weibull_touch_probability(d, theta, p) for d in fine_grid])
        
        # Cumulative probability
        cumulative = np.cumsum(touch_probs)
        cumulative /= cumulative[-1]
        
        # Select depths at equal cumulative intervals
        target_cumulative = np.linspace(0, 1, num_rungs + 2)[1:-1]  # Exclude endpoints
        depths = np.interp(target_cumulative, cumulative, fine_grid)
        
        print(f"Generated {num_rungs} rungs with dynamic density positioning")
        return depths

    def _get_positioning_method(self, rung_positioning: str) -> str:
        """Map rung positioning UI option to ladder_depths method or handle custom methods"""
        # Methods handled by ladder_depths.py
        standard_methods = {
            'quantile': 'quantile',
            'expected_value': 'expected_value',
            'linear': 'linear',
            'exponential': 'exponential',
            'logarithmic': 'logarithmic',
            'risk_weighted': 'risk_weighted'
        }
        
        # Advanced methods handled here in calculator
        if rung_positioning in ['support_resistance', 'volume_profile', 'touch_pattern', 
                               'adaptive_probability', 'fibonacci', 'dynamic_density']:
            return rung_positioning  # Will be handled by custom logic
        
        return standard_methods.get(rung_positioning, 'linear')

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
