"""
Strategy pattern implementation for quantity distribution and rung positioning methods.
Consolidates calculation logic and makes it more maintainable and extensible.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from weibull_fit import weibull_tail


class QuantityDistributionStrategy(ABC):
    """Abstract base class for quantity distribution strategies"""
    
    @abstractmethod
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        """Calculate quantity allocations for given parameters"""
        pass


class EqualQuantityStrategy(QuantityDistributionStrategy):
    """Equal quantity allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        prices = current_price * (1 - depths / 100)
        return (np.full(num_rungs, budget / num_rungs / prices.mean()) * prices) * (budget / (np.full(num_rungs, budget / num_rungs / prices.mean()) * prices).sum())


class EqualNotionalStrategy(QuantityDistributionStrategy):
    """Equal notional allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        return np.full(num_rungs, budget / num_rungs)


class LinearIncreaseStrategy(QuantityDistributionStrategy):
    """Linear increase allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        return np.linspace(0.5, 2.0, num_rungs) / np.linspace(0.5, 2.0, num_rungs).sum() * budget


class ExponentialIncreaseStrategy(QuantityDistributionStrategy):
    """Exponential increase allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        return np.exp(np.linspace(-1, 1, num_rungs)) / np.exp(np.linspace(-1, 1, num_rungs)).sum() * budget


class PriceWeightedStrategy(QuantityDistributionStrategy):
    """Price-weighted allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        return np.full(num_rungs, budget / num_rungs)


class RiskParityStrategy(QuantityDistributionStrategy):
    """Risk parity allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        return (1.0 / (depths + 1)) / (1.0 / (depths + 1)).sum() * budget


class AdaptiveKellyStrategy(QuantityDistributionStrategy):
    """Adaptive Kelly criterion allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        if not weibull_params:
            # Fallback to equal allocation
            return np.full(len(depths), budget / len(depths))
        
        theta = weibull_params['buy']['theta']
        p = weibull_params['buy']['p']
        touch_probs = np.array([weibull_tail(d, theta, p) for d in depths])
        
        # Expected returns based on depth (more depth = more potential return)
        expected_returns = depths / 10.0  # Simplified return model
        
        # Kelly fraction with safety factor
        kelly_fractions = (touch_probs * expected_returns) / (expected_returns + 1)
        kelly_fractions = np.clip(kelly_fractions, 0.05, 0.25)  # Safety limits
        
        # Normalize to budget
        allocations = kelly_fractions / kelly_fractions.sum() * budget
        return allocations


class VolatilityWeightedStrategy(QuantityDistributionStrategy):
    """Volatility-weighted allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        # Use depth as proxy for volatility risk
        volatility_proxy = depths
        inverse_vol = 1.0 / (volatility_proxy + 1)
        weights = inverse_vol / inverse_vol.sum()
        return weights * budget


class SharpeMaximizingStrategy(QuantityDistributionStrategy):
    """Sharpe ratio maximizing allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        if not weibull_params:
            # Fallback to equal allocation
            return np.full(len(depths), budget / len(depths))
        
        theta = weibull_params['buy']['theta']
        p = weibull_params['buy']['p']
        touch_probs = np.array([weibull_tail(d, theta, p) for d in depths])
        
        expected_returns = depths / 10.0  # Potential return
        risks = 1.0 / (touch_probs + 0.01)  # Risk as inverse of touch probability
        
        sharpe_ratios = expected_returns / risks
        sharpe_ratios = np.clip(sharpe_ratios, 0, np.percentile(sharpe_ratios, 90))
        
        weights = sharpe_ratios / sharpe_ratios.sum()
        return weights * budget


class FibonacciWeightedStrategy(QuantityDistributionStrategy):
    """Fibonacci-weighted allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        num_rungs = len(depths)
        
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
        
        # Normalize to budget
        allocations = (fib_weights / fib_weights.sum()) * budget
        
        print(f"Fibonacci allocation: Min=${allocations.min():.2f}, Max=${allocations.max():.2f}, Ratio={allocations.max()/allocations.min():.2f}x")
        return allocations


class ProbabilityWeightedStrategy(QuantityDistributionStrategy):
    """Probability-weighted allocation strategy"""
    
    def calculate(self, depths: np.ndarray, budget: float, current_price: float, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        if not weibull_params:
            # Fallback to equal allocation
            return np.full(len(depths), budget / len(depths))
        
        theta = weibull_params['buy']['theta']
        p = weibull_params['buy']['p']
        touch_probs = np.array([weibull_tail(d, theta, p) for d in depths])
        
        return (touch_probs / touch_probs.sum()) * budget


class QuantityDistributionFactory:
    """Factory for creating quantity distribution strategies"""
    
    _strategies = {
        'equal_quantity': EqualQuantityStrategy,
        'equal_notional': EqualNotionalStrategy,
        'linear_increase': LinearIncreaseStrategy,
        'exponential_increase': ExponentialIncreaseStrategy,
        'price_weighted': PriceWeightedStrategy,
        'risk_parity': RiskParityStrategy,
        'adaptive_kelly': AdaptiveKellyStrategy,
        'volatility_weighted': VolatilityWeightedStrategy,
        'sharpe_maximizing': SharpeMaximizingStrategy,
        'fibonacci_weighted': FibonacciWeightedStrategy,
        'probability_weighted': ProbabilityWeightedStrategy,
        'kelly_optimized': AdaptiveKellyStrategy,  # Alias for adaptive_kelly
    }
    
    @classmethod
    def create_strategy(cls, method: str) -> QuantityDistributionStrategy:
        """Create a quantity distribution strategy instance"""
        strategy_class = cls._strategies.get(method)
        if not strategy_class:
            print(f"Warning: Unknown quantity distribution method '{method}', using equal_notional")
            strategy_class = EqualNotionalStrategy
        
        return strategy_class()
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available quantity distribution methods"""
        return list(cls._strategies.keys())


class RungPositioningStrategy(ABC):
    """Abstract base class for rung positioning strategies"""
    
    @abstractmethod
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        """Calculate rung positions for given parameters"""
        pass


class LinearPositioningStrategy(RungPositioningStrategy):
    """Linear spacing positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        return np.linspace(d_min, d_max, num_rungs)


class ExponentialPositioningStrategy(RungPositioningStrategy):
    """Exponential spacing positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        # Exponential spacing - closer to current price
        depths = np.linspace(0, 1, num_rungs)
        depths = d_min + (d_max - d_min) * (np.exp(depths) - 1) / (np.e - 1)
        return depths


class LogarithmicPositioningStrategy(RungPositioningStrategy):
    """Logarithmic spacing positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        # Logarithmic spacing
        depths = np.linspace(0, 1, num_rungs)
        depths = d_min + (d_max - d_min) * np.log(1 + depths * (np.e - 1)) / np.log(np.e)
        return depths


class FibonacciPositioningStrategy(RungPositioningStrategy):
    """Fibonacci levels positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        # Fibonacci retracement levels
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        depths = []
        
        for i in range(num_rungs):
            if i < len(fib_levels):
                depth = d_min + (d_max - d_min) * fib_levels[i]
            else:
                # Extend beyond standard Fibonacci levels
                depth = d_min + (d_max - d_min) * (0.786 + (i - len(fib_levels) + 1) * 0.1)
            depths.append(depth)
        
        return np.array(depths)


class QuantilePositioningStrategy(RungPositioningStrategy):
    """Quantile-based positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        # Use quantiles for positioning
        quantiles = np.linspace(0.1, 0.9, num_rungs)
        depths = d_min + (d_max - d_min) * quantiles
        return depths


class DynamicDensityPositioningStrategy(RungPositioningStrategy):
    """Dynamic density positioning strategy"""
    
    def calculate(self, num_rungs: int, d_min: float, d_max: float, 
                 current_price: float, timeframe_hours: int, 
                 weibull_params: Optional[Dict] = None) -> np.ndarray:
        if not weibull_params:
            # Fallback to linear spacing
            return np.linspace(d_min, d_max, num_rungs)
        
        theta = weibull_params['buy']['theta']
        p = weibull_params['buy']['p']
        
        # Generate more points for density calculation
        fine_depths = np.linspace(d_min, d_max, num_rungs * 10)
        densities = np.array([weibull_touch_probability(d, theta, p) for d in fine_depths])
        
        # Find peaks in density
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(densities, height=np.mean(densities))
        
        if len(peaks) >= num_rungs:
            # Use the highest density peaks
            peak_densities = densities[peaks]
            sorted_indices = np.argsort(peak_densities)[::-1]
            selected_peaks = peaks[sorted_indices[:num_rungs]]
            depths = fine_depths[selected_peaks]
        else:
            # Combine peaks with uniform spacing
            depths = list(fine_depths[peaks])
            remaining = num_rungs - len(depths)
            if remaining > 0:
                uniform_depths = np.linspace(d_min, d_max, remaining)
                depths.extend(uniform_depths)
            depths = np.array(depths[:num_rungs])
        
        return np.sort(depths)


class RungPositioningFactory:
    """Factory for creating rung positioning strategies"""
    
    _strategies = {
        'linear': LinearPositioningStrategy,
        'exponential': ExponentialPositioningStrategy,
        'logarithmic': LogarithmicPositioningStrategy,
        'fibonacci': FibonacciPositioningStrategy,
        'quantile': QuantilePositioningStrategy,
        'dynamic_density': DynamicDensityPositioningStrategy,
    }
    
    @classmethod
    def create_strategy(cls, method: str) -> RungPositioningStrategy:
        """Create a rung positioning strategy instance"""
        strategy_class = cls._strategies.get(method)
        if not strategy_class:
            print(f"Warning: Unknown rung positioning method '{method}', using linear")
            strategy_class = LinearPositioningStrategy
        
        return strategy_class()
    
    @classmethod
    def get_available_methods(cls) -> list:
        """Get list of available rung positioning methods"""
        return list(cls._strategies.keys())
