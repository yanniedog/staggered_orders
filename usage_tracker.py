"""
Usage Tracking System
Tracks configuration usage patterns for the staggered order ladder GUI.
"""
import json
import os
import time
from typing import Dict, List, Tuple, Any


class UsageTracker:
    """Tracks and analyzes configuration usage patterns"""
    
    def __init__(self, storage_file='usage_stats.json'):
        """
        Initialize the usage tracker
        
        Args:
            storage_file: Path to the JSON file for persistent storage
        """
        self.storage_file = storage_file
        self.usage_stats = {}
        self.load_stats()
    
    def load_stats(self):
        """Load usage statistics from persistent storage"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    self.usage_stats = json.load(f)
                print(f"Loaded usage stats: {len(self.usage_stats)} configurations tracked")
            else:
                self.usage_stats = {}
                print("No existing usage stats found, starting fresh")
        except Exception as e:
            print(f"Error loading usage stats: {e}")
            self.usage_stats = {}
    
    def save_stats(self):
        """Save usage statistics to persistent storage"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.usage_stats, f, indent=2)
            print(f"Saved usage stats: {len(self.usage_stats)} configurations")
        except Exception as e:
            print(f"Error saving usage stats: {e}")
    
    def generate_cache_key(self, config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Cache key string
        """
        return (f"{config['crypto_symbol']}_{config['aggression_level']}_"
                f"{config['num_rungs']}_{config['timeframe_hours']}_"
                f"{config['budget']}_{config['quantity_distribution']}_"
                f"{config['rung_positioning']}")
    
    def track_usage(self, aggression_level: int, num_rungs: int, timeframe_hours: int,
                    budget: float, quantity_distribution: str, crypto_symbol: str,
                    rung_positioning: str):
        """
        Track usage of a configuration
        
        Args:
            aggression_level: Aggression level (1-5)
            num_rungs: Number of ladder rungs
            timeframe_hours: Analysis timeframe in hours
            budget: Budget in USD
            quantity_distribution: Quantity distribution method
            crypto_symbol: Cryptocurrency symbol
            rung_positioning: Rung positioning method
        """
        config = {
            'aggression_level': aggression_level,
            'num_rungs': num_rungs,
            'timeframe_hours': timeframe_hours,
            'budget': budget,
            'quantity_distribution': quantity_distribution,
            'crypto_symbol': crypto_symbol,
            'rung_positioning': rung_positioning
        }
        
        config_key = self.generate_cache_key(config)
        
        # Increment usage count
        if config_key in self.usage_stats:
            self.usage_stats[config_key]['count'] += 1
            self.usage_stats[config_key]['last_used'] = time.time()
        else:
            self.usage_stats[config_key] = {
                'count': 1,
                'first_used': time.time(),
                'last_used': time.time(),
                'config': config
            }
        
        # Save stats periodically (every 10 uses)
        if self.usage_stats[config_key]['count'] % 10 == 0:
            self.save_stats()
    
    def get_most_used_configurations(self, limit: int = 20) -> List[Tuple[str, Dict]]:
        """
        Get the most frequently used configurations
        
        Args:
            limit: Maximum number of configurations to return
            
        Returns:
            List of (config_key, stats) tuples sorted by usage count
        """
        if not self.usage_stats:
            return []
        
        # Sort by usage count (descending)
        sorted_configs = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        # Return top configurations
        return sorted_configs[:limit]
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get usage analytics summary
        
        Returns:
            Dictionary with analytics metrics
        """
        if not self.usage_stats:
            return {
                'total_configurations': 0,
                'total_uses': 0,
                'most_popular_crypto': 'N/A',
                'most_popular_qty_method': 'N/A',
                'most_popular_pos_method': 'N/A',
                'average_uses_per_config': 0,
                'crypto_distribution': {},
                'method_distribution': {'quantity': {}, 'positioning': {}}
            }
        
        total_configs = len(self.usage_stats)
        total_uses = sum(stats['count'] for stats in self.usage_stats.values())
        
        # Analyze crypto usage
        crypto_usage = {}
        method_usage = {'quantity': {}, 'positioning': {}}
        
        for config_key, stats in self.usage_stats.items():
            config = stats['config']
            
            # Track crypto usage
            crypto = config['crypto_symbol']
            crypto_usage[crypto] = crypto_usage.get(crypto, 0) + stats['count']
            
            # Track method usage
            qty_method = config['quantity_distribution']
            pos_method = config['rung_positioning']
            
            method_usage['quantity'][qty_method] = method_usage['quantity'].get(qty_method, 0) + stats['count']
            method_usage['positioning'][pos_method] = method_usage['positioning'].get(pos_method, 0) + stats['count']
        
        most_popular_crypto = max(crypto_usage.items(), key=lambda x: x[1])[0] if crypto_usage else 'N/A'
        most_popular_qty = max(method_usage['quantity'].items(), key=lambda x: x[1])[0] if method_usage['quantity'] else 'N/A'
        most_popular_pos = max(method_usage['positioning'].items(), key=lambda x: x[1])[0] if method_usage['positioning'] else 'N/A'
        
        return {
            'total_configurations': total_configs,
            'total_uses': total_uses,
            'most_popular_crypto': most_popular_crypto,
            'most_popular_qty_method': most_popular_qty,
            'most_popular_pos_method': most_popular_pos,
            'average_uses_per_config': total_uses / total_configs if total_configs > 0 else 0,
            'crypto_distribution': crypto_usage,
            'method_distribution': method_usage
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get raw usage statistics dictionary"""
        return self.usage_stats

