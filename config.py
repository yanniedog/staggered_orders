"""
Centralized configuration loader.
Single source of truth for all configuration loading.
"""
import yaml
from logger import log_info, log_warning, log_error, log_debug

_CONFIG_CACHE = None

def _get_default_config():
    """Get default configuration"""
    return {
        'budget_usd': 100000.0,
        'symbol': 'SOLUSDT',
        'lookback_days': 1095,
        'max_analysis_hours': 720,
        'cache_data': True,
        'cache_hours': 24,
        'num_rungs': 30,
        'min_notional': 10.0,
        'min_fit_quality': 0.90,
        'risk_adjustment_factor': 1.5,
        'total_cost_pct': 0.25
    }

def load_config():
    """Load configuration from config.yaml with defaults fallback"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            defaults = _get_default_config()
            # Only update non-null values
            for key, value in config.items():
                if value is not None:
                    defaults[key] = value
            _CONFIG_CACHE = defaults
            return _CONFIG_CACHE
    except FileNotFoundError:
        log_warning("Warning: config.yaml not found, using defaults")
        _CONFIG_CACHE = _get_default_config()
        return _CONFIG_CACHE
    except yaml.YAMLError as e:
        log_warning(f"Warning: Error parsing config.yaml: {e}")
        _CONFIG_CACHE = _get_default_config()
        return _CONFIG_CACHE

def get_config():
    """Get cached configuration"""
    return load_config()

def clear_config_cache():
    """Clear configuration cache"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
