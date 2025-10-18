"""
Configuration loader for GUI
Simple config loading utility.
"""
import yaml

def load_config():
    """Load configuration from config.yaml"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: config.yaml not found, using defaults")
        return {
            'budget_usd': 100000.0,
            'symbol': 'SOLUSDT',
            'lookback_days': 1095,
            'max_analysis_hours': 720
        }
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing config.yaml: {e}")
        return {
            'budget_usd': 100000.0,
            'symbol': 'SOLUSDT', 
            'lookback_days': 1095,
            'max_analysis_hours': 720
        }
