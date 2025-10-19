"""
Comprehensive logging system for staggered order ladder analysis.
Captures all problems, warnings, and analysis details for each run.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import traceback
import warnings
import sys


class AnalysisLogger:
    """
    Comprehensive logger that captures all analysis details, problems, and warnings.
    """
    
    def __init__(self, output_dir: str = "output", symbol: str = "SOLUSDT"):
        """
        Initialize the analysis logger.
        
        Args:
            output_dir: Directory to save log files
            symbol: Trading symbol for log file naming
        """
        self.output_dir = output_dir
        self.symbol = symbol
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_data = {
            "run_id": self.run_id,
            "symbol": symbol,
            "start_time": datetime.now().isoformat(),
            "problems": [],
            "warnings": [],
            "analysis_steps": [],
            "configuration": {},
            "data_summary": {},
            "validation_results": {},
            "performance_metrics": {},
            "errors": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup file logging
        self.log_file = os.path.join(output_dir, f"analysis_log_{symbol}_{self.run_id}.log")
        self.json_file = os.path.join(output_dir, f"analysis_data_{symbol}_{self.run_id}.json")
        
        # Configure logging
        self._setup_logging()
        
        # Capture warnings
        self._setup_warning_capture()
        
    def _setup_logging(self):
        """Setup file and console logging with stdout/stderr capture."""
        # Create logger
        self.logger = logging.getLogger(f"analysis_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Capture stdout and stderr
        self._setup_output_capture()
        
    def _setup_output_capture(self):
        """Capture stdout and stderr to ensure all output goes to logfile."""
        import io
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create custom stream that logs to file
        class LoggingStream(io.TextIOWrapper):
            def __init__(self, logger, level, original_stream):
                self.logger = logger
                self.level = level
                self.original_stream = original_stream
                
            def write(self, message):
                if message.strip():  # Only log non-empty messages
                    self.logger.log(self.level, message.strip())
                self.original_stream.write(message)
                return len(message)
                
            def flush(self):
                self.original_stream.flush()
        
        # Redirect stdout and stderr
        sys.stdout = LoggingStream(self.logger, logging.INFO, self.original_stdout)
        sys.stderr = LoggingStream(self.logger, logging.ERROR, self.original_stderr)
        
    def _setup_warning_capture(self):
        """Capture warnings and add them to log data."""
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            warning_info = {
                "message": str(message),
                "category": category.__name__,
                "filename": filename,
                "lineno": lineno,
                "timestamp": datetime.now().isoformat()
            }
            self.log_data["warnings"].append(warning_info)
            self.logger.warning(f"{category.__name__}: {message}")
        
        # Capture warnings
        warnings.showwarning = warning_handler
        
    def log_problem(self, problem: str, severity: str = "WARNING", details: Dict = None):
        """
        Log a problem encountered during analysis.
        
        Args:
            problem: Description of the problem
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            details: Additional details about the problem
        """
        problem_info = {
            "problem": problem,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.log_data["problems"].append(problem_info)
        
        if severity == "ERROR":
            self.logger.error(f"PROBLEM: {problem}")
        elif severity == "WARNING":
            self.logger.warning(f"PROBLEM: {problem}")
        else:
            self.logger.info(f"PROBLEM: {problem}")
            
    def log_analysis_step(self, step: str, status: str = "SUCCESS", details: Dict = None):
        """
        Log an analysis step.
        
        Args:
            step: Description of the analysis step
            status: Status of the step (SUCCESS, FAILED, WARNING)
            details: Additional details about the step
        """
        step_info = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.log_data["analysis_steps"].append(step_info)
        
        if status == "SUCCESS":
            self.logger.info(f"STEP: {step}")
        elif status == "FAILED":
            self.logger.error(f"STEP FAILED: {step}")
        else:
            self.logger.warning(f"STEP WARNING: {step}")
            
    def log_configuration(self, config: Dict):
        """Log the configuration used for this run."""
        self.log_data["configuration"] = config
        self.logger.info("Configuration logged")
        
    def log_data_summary(self, summary: Dict):
        """Log data summary information."""
        self.log_data["data_summary"] = summary
        self.logger.info("Data summary logged")
        
    def log_validation_results(self, results: Dict):
        """Log validation results."""
        self.log_data["validation_results"] = results
        self.logger.info("Validation results logged")
        
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics."""
        self.log_data["performance_metrics"] = metrics
        self.logger.info("Performance metrics logged")
        
    def log_error(self, error: Exception, context: str = ""):
        """
        Log an error with full traceback.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_data["errors"].append(error_info)
        self.logger.error(f"ERROR in {context}: {error}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
    def log_weibull_fit(self, side: str, fit_results: Dict):
        """Log Weibull fit results."""
        fit_info = {
            "side": side,
            "theta": fit_results.get("theta", 0),
            "p": fit_results.get("p", 0),
            "r_squared": fit_results.get("r_squared", 0),
            "rmse": fit_results.get("rmse", 0),
            "fit_quality": fit_results.get("fit_quality", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
        
        if "weibull_fits" not in self.log_data:
            self.log_data["weibull_fits"] = {}
        self.log_data["weibull_fits"][side] = fit_info
        
        self.logger.info(f"Weibull fit ({side}): R²={fit_results.get('r_squared', 0):.4f}, "
                        f"RMSE={fit_results.get('rmse', 0):.4f}, "
                        f"Quality={fit_results.get('fit_quality', 'unknown')}")
        
    def log_scenario_results(self, scenarios: List[Dict]):
        """Log scenario analysis results."""
        self.log_data["scenarios"] = scenarios
        
        # Log top scenarios
        if scenarios:
            top_scenario = scenarios[0]
            self.logger.info(f"Top scenario: {top_scenario.get('profit_target', 0):.1f}% profit, "
                           f"{top_scenario.get('rungs', 0)} rungs, "
                           f"{top_scenario.get('expected_return', 0):.3f} expected return")
        
    def log_order_results(self, orders: Dict):
        """Log order generation results."""
        self.log_data["orders"] = orders
        
        if "paired_orders" in orders:
            pairs = orders["paired_orders"]
            self.logger.info(f"Generated {len(pairs)} paired orders")
            if pairs:
                total_profit = sum(pair.get("expected_profit", 0) for pair in pairs)
                self.logger.info(f"Total expected profit: ${total_profit:.2f}")
        
    def log_sensitivity_results(self, sensitivity: Dict):
        """Log sensitivity analysis results."""
        self.log_data["sensitivity"] = sensitivity
        self.logger.info("Sensitivity analysis completed")
        
    def cleanup(self):
        """Restore original stdout/stderr streams."""
        if hasattr(self, 'original_stdout'):
            sys.stdout = self.original_stdout
        if hasattr(self, 'original_stderr'):
            sys.stderr = self.original_stderr
            
    def finalize(self):
        """Finalize the log and save all data."""
        self.log_data["end_time"] = datetime.now().isoformat()
        
        # Calculate run duration
        start_time = datetime.fromisoformat(self.log_data["start_time"])
        end_time = datetime.fromisoformat(self.log_data["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.log_data["duration_seconds"] = duration
        
        # Save JSON data
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Analysis data saved to {self.json_file}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON data: {e}")
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("ANALYSIS RUN SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Duration: {duration:.1f} seconds")
        self.logger.info(f"Problems: {len(self.log_data['problems'])}")
        self.logger.info(f"Warnings: {len(self.log_data['warnings'])}")
        self.logger.info(f"Errors: {len(self.log_data['errors'])}")
        self.logger.info(f"Analysis Steps: {len(self.log_data['analysis_steps'])}")
        
        # Log critical problems
        critical_problems = [p for p in self.log_data['problems'] if p['severity'] == 'CRITICAL']
        if critical_problems:
            self.logger.error("CRITICAL PROBLEMS FOUND:")
            for problem in critical_problems:
                self.logger.error(f"  - {problem['problem']}")
        
        # Log errors
        if self.log_data['errors']:
            self.logger.error("ERRORS ENCOUNTERED:")
            for error in self.log_data['errors']:
                self.logger.error(f"  - {error['error_type']}: {error['error_message']}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Data file: {self.json_file}")
        self.logger.info("=" * 60)
        
        # Cleanup stdout/stderr redirection
        self.cleanup()
        
        return self.log_data


# Global logger context for easy access from any module
_global_logger = None

def get_global_logger():
    """Get the current global logger instance."""
    return _global_logger

def set_global_logger(logger):
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger

def log_info(message):
    """Log info message using global logger."""
    if _global_logger:
        _global_logger.logger.info(message)
    else:
        print(message)

def log_warning(message):
    """Log warning message using global logger."""
    if _global_logger:
        _global_logger.logger.warning(message)
    else:
        print(f"WARNING: {message}")

def log_error(message):
    """Log error message using global logger."""
    if _global_logger:
        _global_logger.logger.error(message)
    else:
        print(f"ERROR: {message}")

def log_debug(message):
    """Log debug message using global logger."""
    if _global_logger:
        _global_logger.logger.debug(message)
    else:
        print(f"DEBUG: {message}")


def create_analysis_logger(output_dir: str = "output", symbol: str = "SOLUSDT") -> AnalysisLogger:
    """
    Create a new analysis logger instance.
    
    Args:
        output_dir: Directory to save log files
        symbol: Trading symbol for log file naming
        
    Returns:
        AnalysisLogger instance
    """
    return AnalysisLogger(output_dir, symbol)


# Context manager for automatic cleanup
class LoggingContext:
    """Context manager for automatic logger cleanup."""
    
    def __init__(self, output_dir: str = "output", symbol: str = "SOLUSDT"):
        self.logger = None
        self.output_dir = output_dir
        self.symbol = symbol
        
    def __enter__(self):
        self.logger = create_analysis_logger(self.output_dir, self.symbol)
        set_global_logger(self.logger)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            if exc_type:
                self.logger.log_error(exc_val, "Context manager exit")
            self.logger.finalize()
        set_global_logger(None)


if __name__ == "__main__":
    # Test the logger
    with LoggingContext() as logger:
        logger.log_analysis_step("Testing logger", "SUCCESS")
        logger.log_problem("Test problem", "WARNING")
        logger.log_configuration({"test": "config"})
        logger.log_data_summary({"candles": 1000})
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.log_error(e, "Test context")
