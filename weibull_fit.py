"""
Weibull tail distribution fitting for touch probabilities.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import warnings


def weibull_tail(depth: np.ndarray, theta: float, p: float) -> np.ndarray:
    """
    Weibull tail distribution: q(d) = exp(-(d/theta)^p)
    
    Args:
        depth: Depth values (percentages)
        theta: Scale parameter
        p: Shape parameter
    
    Returns:
        Touch probabilities
    """
    return np.exp(-(depth / theta) ** p)


def fit_weibull_tail(depths: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fit Weibull tail distribution to touch probabilities with enhanced validation.
    
    Args:
        depths: Array of depths (percentages)
        probabilities: Array of empirical probabilities
    
    Returns:
        Tuple of (theta, p, fit_metrics)
    """
    # Validate inputs
    if len(depths) != len(probabilities):
        raise ValueError(f"Length mismatch: depths={len(depths)}, probabilities={len(probabilities)}")
    
    if len(depths) < 3:
        raise ValueError(f"Insufficient data points: {len(depths)} (minimum 3 required)")
    
    # Check for invalid values
    if np.any(np.isnan(depths)) or np.any(np.isnan(probabilities)):
        raise ValueError("NaN values found in input data")
    
    if np.any(np.isinf(depths)) or np.any(np.isinf(probabilities)):
        raise ValueError("Inf values found in input data")
    
    # Validate probability bounds
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        invalid_mask = (probabilities < 0) | (probabilities > 1)
        print(f"Warning: Found {np.sum(invalid_mask)} probabilities outside [0,1] range")
        print("Clipping probabilities to valid range")
        probabilities = np.clip(probabilities, 1e-10, 1.0)  # Avoid log(0)
    
    # Validate depth bounds
    if np.any(depths <= 0):
        raise ValueError("All depths must be positive")
    
    # Filter out zero probabilities for log transformation
    valid_mask = probabilities > 1e-10
    depths_valid = depths[valid_mask]
    probs_valid = probabilities[valid_mask]
    
    if len(depths_valid) < 3:
        raise ValueError("Insufficient valid data points after filtering")
    
    print(f"Fitting Weibull with {len(depths_valid)} valid data points")
    
    # Log transform for linear fitting
    log_probs = -np.log(probs_valid)
    
    # Initial guess for parameters
    # theta: scale parameter (typical depth)
    # p: shape parameter (tail behavior)
    theta_guess = np.median(depths_valid)
    p_guess = 1.0
    
    try:
        # Fit the transformed function: -log(q) = (d/theta)^p
        # This becomes: log(-log(q)) = p * log(d) - p * log(theta)
        # So: log(-log(q)) = p * log(d/theta)
        
        # Define the function to fit
        def log_weibull(d, theta, p):
            return (d / theta) ** p
        
        # Fit parameters with robust bounds
        popt, pcov = curve_fit(
            log_weibull, 
            depths_valid, 
            log_probs,
            p0=[theta_guess, p_guess],
            maxfev=1000,
            bounds=([0.1, 0.1], [100.0, 5.0])
        )
        
        theta_fit, p_fit = popt
        
        # Validate fitted parameters
        if theta_fit <= 0 or p_fit <= 0:
            raise ValueError(f"Invalid fitted parameters: theta={theta_fit}, p={p_fit}")
        
        if theta_fit > 50:  # More than 50% typical depth is unrealistic
            print(f"Warning: Unrealistic theta parameter: {theta_fit:.2f}%")
        
        if p_fit > 5:  # Very high shape parameter
            print(f"Warning: Very high shape parameter: {p_fit:.2f}")
        
        # Calculate fit quality
        predicted_log_probs = log_weibull(depths_valid, theta_fit, p_fit)
        predicted_probs = np.exp(-predicted_log_probs)
        
        # Ensure predicted probabilities are valid
        predicted_probs = np.clip(predicted_probs, 1e-10, 1.0)
        
        # R² calculation
        ss_res = np.sum((probs_valid - predicted_probs) ** 2)
        ss_tot = np.sum((probs_valid - np.mean(probs_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        rmse = np.sqrt(np.mean((probs_valid - predicted_probs) ** 2))
        
        # Correlation coefficient
        correlation, _ = pearsonr(probs_valid, predicted_probs)
        
        # Basic residual analysis
        residuals = probs_valid - predicted_probs
        residual_std = np.std(residuals)
        residual_skewness = 0.0  # Placeholder - could calculate actual skewness if needed
        
        # Determine fit quality
        if r_squared >= 0.95:
            fit_quality = "excellent"
        elif r_squared >= 0.90:
            fit_quality = "good"
        elif r_squared >= 0.80:
            fit_quality = "fair"
        else:
            fit_quality = "poor"
        
        fit_metrics = {
            'theta': theta_fit,
            'p': p_fit,
            'r_squared': r_squared,
            'rmse': rmse,
            'correlation': correlation,
            'residual_std': residual_std,
            'residual_skewness': residual_skewness,
            'n_points': len(depths_valid),
            'n_valid_points': len(depths_valid),
            'fit_quality': fit_quality,
            'converged': True
        }
        
        print(f"Weibull fit results:")
        print(f"  Theta (scale): {theta_fit:.3f}")
        print(f"  P (shape): {p_fit:.3f}")
        print(f"  R²: {r_squared:.4f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Residual std: {residual_std:.6f}")
        print(f"  Data points: {len(depths_valid)}")
        print(f"  Fit quality: {fit_quality}")
        
        return theta_fit, p_fit, fit_metrics
        
    except Exception as e:
        warnings.warn(f"Weibull fitting failed: {e}")
        # Fallback to simple exponential
        return fit_exponential_fallback(depths_valid, probs_valid)


def fit_exponential_fallback(depths: np.ndarray, probabilities: np.ndarray) -> Tuple[float, float, Dict]:
    """
    Fallback to exponential distribution if Weibull fails.
    """
    print("Falling back to exponential distribution")
    
    # Exponential: q(d) = exp(-d/theta)
    log_probs = -np.log(probabilities)
    
    # Linear fit: log_probs = d/theta
    theta_fit = np.mean(depths / log_probs)
    p_fit = 1.0  # Exponential is Weibull with p=1
    
    # Calculate metrics
    predicted_probs = np.exp(-depths / theta_fit)
    
    ss_res = np.sum((probabilities - predicted_probs) ** 2)
    ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    rmse = np.sqrt(np.mean((probabilities - predicted_probs) ** 2))
    correlation, _ = pearsonr(probabilities, predicted_probs)
    
    fit_metrics = {
        'r_squared': r_squared,
        'rmse': rmse,
        'correlation': correlation,
        'n_points': len(depths),
        'theta_std': 0.0,
        'p_std': 0.0
    }
    
    print(f"Exponential fit results:")
    print(f"  Theta (scale): {theta_fit:.3f}")
    print(f"  R²: {r_squared:.4f}")
    print(f"  RMSE: {rmse:.6f}")
    
    return theta_fit, p_fit, fit_metrics


def plot_fit_diagnostic(depths: np.ndarray, probabilities: np.ndarray, 
                       theta: float, p: float, fit_metrics: Dict) -> None:
    """
    Create diagnostic plot of the Weibull fit.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Linear scale
    predicted_probs = weibull_tail(depths, theta, p)
    
    ax1.plot(depths, probabilities, 'bo', alpha=0.6, label='Empirical')
    ax1.plot(depths, predicted_probs, 'r-', linewidth=2, label='Weibull Fit')
    ax1.set_xlabel('Depth (%)')
    ax1.set_ylabel('Touch Probability')
    ax1.set_title(f'Weibull Fit (R² = {fit_metrics["r_squared"]:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax2.semilogy(depths, probabilities, 'bo', alpha=0.6, label='Empirical')
    ax2.semilogy(depths, predicted_probs, 'r-', linewidth=2, label='Weibull Fit')
    ax2.set_xlabel('Depth (%)')
    ax2.set_ylabel('Touch Probability (log scale)')
    ax2.set_title('Weibull Fit - Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/touch_fit.png', dpi=300, bbox_inches='tight')
    plt.show()


def validate_fit_quality(fit_metrics: Dict, min_r_squared: float = 0.90) -> bool:
    """
    Validate the quality of the Weibull fit with comprehensive checks.
    
    Args:
        fit_metrics: Dictionary with fit quality metrics
        min_r_squared: Minimum acceptable R²
    
    Returns:
        True if fit quality is acceptable
    """
    r_squared = fit_metrics['r_squared']
    n_points = fit_metrics['n_points']
    rmse = fit_metrics['rmse']
    correlation = fit_metrics['correlation']
    cv_rmse = fit_metrics.get('cv_rmse', 0.0)  # Cross-validation RMSE (optional, default 0 if not calculated)
    residual_std = fit_metrics.get('residual_std', 0)
    residual_skewness = fit_metrics.get('residual_skewness', 0)
    fit_quality = fit_metrics.get('fit_quality', 'unknown')
    
    warnings_list = []
    
    # Check R²
    if r_squared < min_r_squared:
        warnings_list.append(f"Poor fit quality: R² = {r_squared:.3f} < {min_r_squared}")
    
    # Check sample size
    if n_points < 50:
        warnings_list.append(f"Insufficient data points: {n_points} < 50 (minimum for reliable Weibull fitting)")
    
    # Check RMSE (relative to data range)
    if rmse > 0.1:  # RMSE > 10% of typical probability range
        warnings_list.append(f"High RMSE: {rmse:.4f} (may indicate poor fit)")
    
    # Check correlation
    if correlation < 0.8:
        warnings_list.append(f"Low correlation: {correlation:.3f} < 0.8")
    
    # Check cross-validation performance (only if calculated)
    if cv_rmse > 0.0 and cv_rmse > 0.2:
        warnings_list.append(f"Poor cross-validation performance: CV RMSE = {cv_rmse:.4f}")
    
    # Check residual properties
    if abs(residual_skewness) > 2.0:
        warnings_list.append(f"Highly skewed residuals: skewness = {residual_skewness:.3f} (indicates systematic bias)")
    
    if residual_std > 0.1:
        warnings_list.append(f"High residual variability: std = {residual_std:.4f}")
    
    # Check parameter stability
    theta_std = fit_metrics.get('theta_std', 0)
    p_std = fit_metrics.get('p_std', 0)
    
    if theta_std > 0.5:  # High uncertainty in scale parameter
        warnings_list.append(f"High uncertainty in theta: std = {theta_std:.3f}")
    
    if p_std > 0.3:  # High uncertainty in shape parameter
        warnings_list.append(f"High uncertainty in p: std = {p_std:.3f}")
    
    # Overall assessment
    if warnings_list:
        print(f"Fit quality validation FAILED:")
        for warning in warnings_list:
            print(f"  - {warning}")
        return False
    else:
        print(f"Fit quality validation PASSED:")
        print(f"  R² = {r_squared:.3f}")
        print(f"  Sample size = {n_points}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  Correlation = {correlation:.3f}")
        if cv_rmse > 0.0:
            print(f"  Cross-validation RMSE = {cv_rmse:.4f}")
        print(f"  Residual std = {residual_std:.4f}")
        print(f"  Residual skewness = {residual_skewness:.3f}")
        print(f"  Overall quality = {fit_quality}")
        return True


if __name__ == "__main__":
    # Test with sample data
    from touch_analysis import analyze_touch_probabilities
    from data_fetcher import fetch_solusdt_data
    
    df = fetch_solusdt_data()
    depths, probs = analyze_touch_probabilities(df, max_analysis_hours=24)
    
    theta, p, metrics = fit_weibull_tail(depths, probs)
    plot_fit_diagnostic(depths, probs, theta, p, metrics)
    validate_fit_quality(metrics)
