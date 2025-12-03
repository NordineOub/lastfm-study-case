"""
Visualization functions for time series and forecast analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_time_series(
    df_ts: pd.DataFrame,
    column: str,
    title: str = "Time Series",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: tuple = (10, 6),
    color: str = "deeppink"
):
    """
    Plot a time series.
    
    Args:
        df_ts: DataFrame with datetime index
        column: Column name to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        color: Line color
    """
    plt.figure(figsize=figsize)
    plt.plot(df_ts[column], linewidth=3, c=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_acf_pacf(df_ts: pd.DataFrame):
    """
    Plot ACF and PACF for time series analysis.
    
    Args:
        df_ts: Time series DataFrame
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df_ts, ax=axes[0])
    plot_pacf(df_ts, ax=axes[1])
    plt.tight_layout()
    plt.show()


def plot_forecast(
    df_ts: pd.DataFrame,
    forecast: pd.Series,
    title: str = "Forecast",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: tuple = (10, 6)
):
    """
    Plot observed data with forecast.
    
    Args:
        df_ts: Historical time series data
        forecast: Forecast series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.plot(df_ts, label="Observed", linewidth=2)
    plt.plot(forecast, label="Forecast", color="red", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, figsize: tuple = (10, 6)):
    """
    Plot comparison of model performance.
    
    Args:
        results_df: DataFrame with model results
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # RMSE comparison
    axes[0].barh(results_df["model"], results_df["rmse"])
    axes[0].set_xlabel("RMSE")
    axes[0].set_title("Model RMSE Comparison")
    axes[0].invert_yaxis()
    
    # R² comparison
    axes[1].barh(results_df["model"], results_df["r2"])
    axes[1].set_xlabel("R² Score")
    axes[1].set_title("Model R² Comparison")
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
