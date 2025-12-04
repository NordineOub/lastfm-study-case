"""
Time series analysis functions including stationarity tests and SARIMAX modeling.
"""

from typing import Dict, Tuple
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


def check_stationarity(timeseries: pd.Series) -> Dict[str, float]:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.
    
    Args:
        timeseries: Time series data to test
        
    Returns:
        Dictionary with test results including adf_statistic, p_value, and is_stationary
    """
    result = adfuller(timeseries, autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05
    }


def fit_sarimax_model(
    df_ts: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
):
    """
    Fit SARIMAX model to time series data.
    
    Args:
        df_ts: DataFrame with datetime index and target column
        order: (p, d, q) for ARIMA component
        seasonal_order: (P, D, Q, s) for seasonal component
    
    Returns:
        Fitted SARIMAX results object
    """
    model = SARIMAX(df_ts, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    return results


def evaluate_forecast(observed: pd.DataFrame, forecast: pd.Series) -> Dict[str, float]:
    """
    Calculate evaluation metrics for forecast.
    
    Args:
        observed: Actual observed values
        forecast: Forecasted values
        
    Returns:
        Dictionary with MAE and MSE metrics
    """
    mae = mean_absolute_error(observed, forecast)
    mse = mean_squared_error(observed, forecast)
    return {"mae": mae, "mse": mse}


def forecast_with_sarimax(
    df_ts: pd.DataFrame,
    forecast_periods: int = 3,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)
) -> Tuple:
    """
    Complete SARIMAX forecasting pipeline.
    
    Args:
        df_ts: Time series DataFrame with datetime index
        forecast_periods: Number of periods to forecast
        order: ARIMA order
        seasonal_order: Seasonal ARIMA order
        
    Returns:
        Tuple of (fitted_model, forecast, metrics)
    """
    results = fit_sarimax_model(df_ts, order, seasonal_order)
    
    forecast = results.forecast(steps=forecast_periods)
    
    observed = df_ts[-forecast_periods:]
    metrics = evaluate_forecast(observed, forecast[:forecast_periods])
    
    
    return results, forecast, metrics
