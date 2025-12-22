from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
from scipy.signal import savgol_filter

def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)
    equation = f"COP = {slope:.4f} × OAT + {intercept:.4f}"
    return y_pred, slope, intercept, r2, equation

def smoothed_linear_regression(X, y, window_length=5, polyorder=2):
    X = np.array(X).reshape(-1, 1) if X.ndim == 1 else np.array(X)
    y = np.array(y).flatten()
    
    # Remove any NaN or inf values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(y))
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) < 2:
        # Not enough data for regression
        return y, y, None, None
    
    # Ensure window_length is appropriate for Savitzky-Golay
    if window_length >= len(y_clean):
        window_length = min(5, len(y_clean))
    if window_length % 2 == 0:  # Must be odd
        window_length = max(3, window_length - 1)
    if window_length < 3:  # Too small
        window_length = 3
    
    try:
        # Apply Savitzky-Golay filter for smoothing
        y_smooth = savgol_filter(y_clean, window_length=window_length, polyorder=min(polyorder, window_length-1))
    except:
        # If smoothing fails, use original data
        y_smooth = y_clean
    
    # Perform linear regression
    try:
        model = LinearRegression()
        model.fit(X_clean, y_smooth)
        y_pred = model.predict(X_clean)
        r2 = model.score(X_clean, y_smooth)
        slope = model.coef_[0] if hasattr(model.coef_, '__len__') else model.coef_
    except:
        # If regression fails, return basic values
        y_pred = y_smooth
        r2 = None
        slope = None
    
    return y_pred, y_smooth, r2, slope