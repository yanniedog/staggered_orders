"""
Rule-based validation system for staggered order analysis results.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationRule:
    name: str
    level: ValidationLevel
    check_func: callable
    message: str
    details: str = ""


class ValidationEngine:
    """Rule-based validation engine for analysis results."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.rules = self._initialize_rules()
        self.results = []
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize all validation rules."""
        return [
            # Critical rules
            ValidationRule(
                "nan_values", ValidationLevel.CRITICAL,
                self._check_nan_values, "No NaN/Inf values found", "Critical data integrity check"
            ),
            ValidationRule(
                "weibull_fit_quality", ValidationLevel.CRITICAL,
                self._check_weibull_fit_quality, "Weibull fit quality acceptable", "Model reliability check"
            ),
            ValidationRule(
                "paired_orders_profitable", ValidationLevel.CRITICAL,
                self._check_paired_orders_profitable, "All paired orders profitable", "Strategy viability check"
            ),
            ValidationRule(
                "probability_bounds", ValidationLevel.CRITICAL,
                self._check_probability_bounds, "Touch probabilities within bounds", "Mathematical validity check"
            ),
            
            # Warning rules
            ValidationRule(
                "weibull_parameters", ValidationLevel.WARNING,
                self._check_weibull_parameters, "Weibull parameters reasonable", "Model parameter validation"
            ),
            ValidationRule(
                "profit_targets", ValidationLevel.WARNING,
                self._check_profit_targets, "Profit targets realistic", "Strategy feasibility check"
            ),
            ValidationRule(
                "time_horizon", ValidationLevel.WARNING,
                self._check_time_horizon, "Time horizon reasonable", "Strategy practicality check"
            ),
            ValidationRule(
                "order_sizes", ValidationLevel.WARNING,
                self._check_order_sizes, "Order sizes practical", "Trading constraints check"
            ),
            ValidationRule(
                "market_conditions", ValidationLevel.WARNING,
                self._check_market_conditions, "Market conditions normal", "Market environment check"
            ),
            
            # Info rules
            ValidationRule(
                "data_consistency", ValidationLevel.INFO,
                self._check_data_consistency, "Data consistency verified", "Cross-validation check"
            ),
            ValidationRule(
                "scenario_analysis", ValidationLevel.INFO,
                self._check_scenario_analysis, "Scenario analysis complete", "Analysis completeness check"
            )
        ]
    
    def validate(self, scenarios_df: pd.DataFrame, optimal_scenario: Dict,
                fit_metrics: Dict, fit_metrics_sell: Dict,
                paired_orders_df: pd.DataFrame) -> bool:
        """Run all validation rules and return overall result."""
        print("\n" + "="*50)
        print("VALIDATION CHECKS")
        print("="*50)
        
        self.results = []
        critical_failures = 0
        
        # Store data for rule checks
        self.scenarios_df = scenarios_df
        self.optimal_scenario = optimal_scenario
        self.fit_metrics = fit_metrics
        self.fit_metrics_sell = fit_metrics_sell
        self.paired_orders_df = paired_orders_df
        
        # Run all rules
        for rule in self.rules:
            result = self._run_rule(rule)
            self.results.append(result)
            
            if result['level'] == ValidationLevel.CRITICAL and not result['passed']:
                critical_failures += 1
        
        # Print summary
        self._print_summary(critical_failures)
        
        return critical_failures == 0
    
    def _run_rule(self, rule: ValidationRule) -> Dict:
        """Run a single validation rule."""
        try:
            passed = rule.check_func()
            result = {
                'rule': rule.name,
                'level': rule.level,
                'passed': passed,
                'message': rule.message,
                'details': rule.details
            }
            
            # Print result
            status = "[PASS]" if passed else "[FAIL]" if rule.level == ValidationLevel.CRITICAL else "[WARN]"
            print(f"\n{rule.name.upper().replace('_', ' ')}")
            print("-" * 30)
            print(f"{status} {rule.message}")
            
            # Log problems
            if not passed and self.logger:
                self.logger.log_problem(f"Validation failed: {rule.name}", 
                                      rule.level.value, {"rule": rule.name})
            
            return result
            
        except Exception as e:
            print(f"\n{rule.name.upper().replace('_', ' ')}")
            print("-" * 30)
            print(f"[ERROR] Rule execution failed: {e}")
            return {
                'rule': rule.name,
                'level': rule.level,
                'passed': False,
                'message': f"Rule execution failed: {e}",
                'details': rule.details
            }
    
    def _print_summary(self, critical_failures: int):
        """Print validation summary."""
        print("\n" + "="*50)
        if critical_failures == 0:
            print("[SUCCESS] All critical validations passed")
        else:
            print(f"[FAILURE] {critical_failures} critical validation failures found")
        print("="*50)
    
    # Rule check methods
    def _check_nan_values(self) -> bool:
        """Check for NaN/Inf values in critical data."""
        nan_checks = []
        
        if self.scenarios_df is not None and not self.scenarios_df.empty:
            numeric_cols = self.scenarios_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.scenarios_df[col].isna().any():
                    nan_checks.append(f"Scenarios: {col} has NaN values")
                if np.isinf(self.scenarios_df[col]).any():
                    nan_checks.append(f"Scenarios: {col} has Inf values")
        
        if not self.paired_orders_df.empty:
            numeric_cols = self.paired_orders_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.paired_orders_df[col].isna().any():
                    nan_checks.append(f"Paired orders: {col} has NaN values")
                if np.isinf(self.paired_orders_df[col]).any():
                    nan_checks.append(f"Paired orders: {col} has Inf values")
        
        if nan_checks:
            for check in nan_checks:
                print(f"  - {check}")
            return False
        return True
    
    def _check_weibull_fit_quality(self) -> bool:
        """Check Weibull fit quality."""
        buy_quality = self.fit_metrics.get('fit_quality', 'unknown')
        sell_quality = self.fit_metrics_sell.get('fit_quality', 'unknown')
        
        if buy_quality in ['poor', 'unknown']:
            print(f"  - Buy-side fit quality: {buy_quality}")
            return False
        
        if sell_quality in ['poor', 'unknown']:
            print(f"  - Sell-side fit quality: {sell_quality}")
            return False
        
        print(f"  - Buy-side: {buy_quality}")
        print(f"  - Sell-side: {sell_quality}")
        return True
    
    def _check_paired_orders_profitable(self) -> bool:
        """Check that all paired orders are profitable."""
        if self.paired_orders_df.empty:
            print("  - No paired orders generated")
            return False
        
        unprofitable_pairs = self.paired_orders_df[self.paired_orders_df['profit_pct'] <= 0]
        if len(unprofitable_pairs) > 0:
            print(f"  - {len(unprofitable_pairs)} unprofitable pairs found")
            return False
        
        print(f"  - All {len(self.paired_orders_df)} pairs are profitable")
        return True
    
    def _check_probability_bounds(self) -> bool:
        """Check that touch probabilities are within valid bounds."""
        if 'avg_buy_touch_prob' in self.optimal_scenario:
            buy_prob = self.optimal_scenario['avg_buy_touch_prob']
            if not (0 <= buy_prob <= 1):
                print(f"  - Buy touch probability out of bounds: {buy_prob:.3f}")
                return False
            print(f"  - Buy touch probability: {buy_prob:.3f}")
        
        if 'avg_sell_touch_prob' in self.optimal_scenario:
            sell_prob = self.optimal_scenario['avg_sell_touch_prob']
            if not (0 <= sell_prob <= 1):
                print(f"  - Sell touch probability out of bounds: {sell_prob:.3f}")
                return False
            print(f"  - Sell touch probability: {sell_prob:.3f}")
        
        return True
    
    def _check_weibull_parameters(self) -> bool:
        """Check Weibull parameters are reasonable."""
        theta = self.fit_metrics.get('theta', None)
        p = self.fit_metrics.get('p', None)
        theta_sell = self.fit_metrics_sell.get('theta', None)
        p_sell = self.fit_metrics_sell.get('p', None)
        
        issues = []
        
        if theta is not None and p is not None:
            if theta <= 0 or theta > 50:
                issues.append(f"Buy-side theta out of bounds: {theta:.3f}")
            if p <= 0 or p > 5:
                issues.append(f"Buy-side p out of bounds: {p:.3f}")
            if theta > 20:
                issues.append(f"Buy-side theta very high: {theta:.3f}")
            if p > 3:
                issues.append(f"Buy-side p very high: {p:.3f}")
        
        if theta_sell is not None and p_sell is not None:
            if theta_sell <= 0 or theta_sell > 50:
                issues.append(f"Sell-side theta out of bounds: {theta_sell:.3f}")
            if p_sell <= 0 or p_sell > 5:
                issues.append(f"Sell-side p out of bounds: {p_sell:.3f}")
            if theta_sell > 20:
                issues.append(f"Sell-side theta very high: {theta_sell:.3f}")
            if p_sell > 3:
                issues.append(f"Sell-side p very high: {p_sell:.3f}")
        
        if issues:
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("  - All parameters within reasonable bounds")
        return True
    
    def _check_profit_targets(self) -> bool:
        """Check profit targets are realistic."""
        if self.paired_orders_df.empty:
            return True
        
        max_profit_pct = self.paired_orders_df['profit_pct'].max()
        min_profit_pct = self.paired_orders_df['profit_pct'].min()
        avg_profit_pct = self.paired_orders_df['profit_pct'].mean()
        
        issues = []
        
        if max_profit_pct > 50:
            issues.append(f"Maximum profit very high: {max_profit_pct:.1f}%")
        
        if min_profit_pct < 0.5:
            issues.append(f"Minimum profit very low: {min_profit_pct:.1f}%")
        
        if issues:
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print(f"  - Profit range: {min_profit_pct:.1f}% - {max_profit_pct:.1f}%")
        print(f"  - Average profit: {avg_profit_pct:.1f}%")
        return True
    
    def _check_time_horizon(self) -> bool:
        """Check time horizon is reasonable."""
        if 'expected_timeframe_hours' not in self.optimal_scenario:
            print("  - Timeframe data not available")
            return True
        
        timeframe = self.optimal_scenario['expected_timeframe_hours']
        
        if timeframe > 365 * 24:
            print(f"  - Very long timeframe: {timeframe:.0f} hours ({timeframe/24/30:.1f} months)")
            return False
        elif timeframe < 1:
            print(f"  - Very short timeframe: {timeframe:.2f} hours")
            return False
        
        print(f"  - Timeframe reasonable: {timeframe:.1f} hours ({timeframe/24:.1f} days)")
        return True
    
    def _check_order_sizes(self) -> bool:
        """Check order sizes are practical."""
        if self.paired_orders_df.empty or 'buy_notional' not in self.paired_orders_df.columns:
            return True
        
        max_order_size = self.paired_orders_df['buy_notional'].max()
        min_order_size = self.paired_orders_df['buy_notional'].min()
        
        issues = []
        
        if max_order_size > 10000:
            issues.append(f"Very large order size: ${max_order_size:.0f}")
        
        if min_order_size < 10:
            issues.append(f"Very small order size: ${min_order_size:.2f}")
        
        if issues:
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print(f"  - Order size range: ${min_order_size:.2f} - ${max_order_size:.0f}")
        return True
    
    def _check_market_conditions(self) -> bool:
        """Check market conditions are normal."""
        issues = []
        
        if 'current_price' in self.optimal_scenario:
            current_price = self.optimal_scenario['current_price']
            if current_price < 1.0 or current_price > 1000:
                issues.append(f"Unusual current price: ${current_price:.2f}")
            else:
                print(f"  - Current price reasonable: ${current_price:.2f}")
        
        if 'total_allocation' in self.optimal_scenario:
            total_allocation = self.optimal_scenario['total_allocation']
            budget = self.optimal_scenario.get('budget_usd', 10000)
            allocation_ratio = total_allocation / budget if budget > 0 else 0
            
            if allocation_ratio > 1.0:
                issues.append(f"Total allocation exceeds budget: {allocation_ratio:.2f}")
            elif allocation_ratio < 0.1:
                issues.append(f"Very low budget utilization: {allocation_ratio:.2f}")
            else:
                print(f"  - Budget utilization: {allocation_ratio:.2f}")
        
        if 'sharpe_ratio' in self.optimal_scenario:
            sharpe = self.optimal_scenario['sharpe_ratio']
            if sharpe > 5.0:
                issues.append(f"Unrealistic Sharpe ratio: {sharpe:.2f}")
            elif sharpe < 0:
                issues.append(f"Negative Sharpe ratio: {sharpe:.2f}")
            else:
                print(f"  - Sharpe ratio reasonable: {sharpe:.2f}")
        
        if issues:
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True
    
    def _check_data_consistency(self) -> bool:
        """Check data consistency across components."""
        if self.paired_orders_df.empty or self.scenarios_df is None:
            return True
        
        paired_profit_range = (self.paired_orders_df['profit_pct'].min(), 
                             self.paired_orders_df['profit_pct'].max())
        scenario_profit_range = (self.scenarios_df['profit_target_pct'].min(), 
                               self.scenarios_df['profit_target_pct'].max())
        
        print(f"  - Paired orders profit range: {paired_profit_range[0]:.1f}% - {paired_profit_range[1]:.1f}%")
        print(f"  - Scenario profit range: {scenario_profit_range[0]:.1f}% - {scenario_profit_range[1]:.1f}%")
        
        if (paired_profit_range[1] < scenario_profit_range[0] or 
            paired_profit_range[0] > scenario_profit_range[1]):
            print("  - Profit ranges don't overlap")
            return False
        
        print("  - Profit ranges overlap correctly")
        return True
    
    def _check_scenario_analysis(self) -> bool:
        """Check scenario analysis completeness."""
        if self.scenarios_df is None or self.scenarios_df.empty:
            print("  - No scenario analysis data available")
            return False
        
        max_scenario_profit = self.scenarios_df['profit_target_pct'].max()
        negative_returns = self.scenarios_df[self.scenarios_df['expected_profit_per_dollar'] <= 0]
        
        print(f"  - Maximum scenario profit: {max_scenario_profit:.1f}%")
        print(f"  - Scenarios with negative returns: {len(negative_returns)}")
        
        if max_scenario_profit > 50:
            print("  - Maximum profit target very high")
            return False
        
        if len(negative_returns) > 0:
            print("  - Some scenarios have negative returns")
            return False
        
        print("  - All scenarios have positive expected returns")
        return True


# Legacy function for backward compatibility
def validate_analysis_results(scenarios_df: pd.DataFrame, optimal_scenario: Dict,
                             fit_metrics: Dict, fit_metrics_sell: Dict,
                             paired_orders_df: pd.DataFrame, 
                             logger=None) -> bool:
    """Legacy validation function using new rule-based engine."""
    engine = ValidationEngine(logger)
    return engine.validate(scenarios_df, optimal_scenario, fit_metrics, fit_metrics_sell, paired_orders_df)


def validate_weibull_fit(fit_metrics: Dict, min_quality: float = 0.90) -> bool:
    """Validate Weibull fit quality."""
    r_squared = fit_metrics.get('r_squared', 0)
    return r_squared >= min_quality


def validate_paired_orders(paired_orders_df: pd.DataFrame) -> bool:
    """Validate paired orders specifications."""
    if paired_orders_df.empty:
        return False
    
    # Check profitability
    unprofitable = paired_orders_df['profit_pct'] <= 0
    if unprofitable.any():
        return False
    
    # Check quantity matching (should be close)
    qty_mismatch = ~np.isclose(paired_orders_df['buy_qty'], paired_orders_df['sell_qty'], rtol=1e-6)
    if qty_mismatch.any():
        return False
    
    return True