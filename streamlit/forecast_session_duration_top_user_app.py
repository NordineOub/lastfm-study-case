import sys
import streamlit as st
import pandas as pd
import numpy as np

# ensure project src is importable
sys.path.append('..')

# Notebook helper imports (reuse existing project code)
from src.common.definition import (
    create_spark_session,
    load_track_data,
    add_sessions_id_columns,
)
from src.data_preparation import prepare_top_user_monthly_sessions_for_forecasting
from src.time_series_analysis import (
    check_stationarity,
    forecast_with_sarimax,
)
from src.ml_models import (
    prepare_train_test_split,
    get_default_models_config,
    train_multiple_models,
    forecast_future_periods,
    create_forecast_dataframe,
)
from src.visualization import (
    plot_forecast,
    plot_model_comparison,
)

from sklearn.linear_model import Ridge


# Configuration
DATA_PATH = "./userid-timestamp-artid-artname-traid-traname.tsv"
SESSION_GAP_SEC = 20 * 60  # 20 minutes
DEFAULT_FORECAST_PERIODS = 3  # Next 3 months


def run():
    st.set_page_config(page_title="Forecast Avg Session Duration - Top User", layout="wide")
    st.title("Forecast Average Session Duration for Top User")
    st.write("Select the top user and forecast the next months of average session duration, starting from the last available record.")

    # Cached Spark session
    @st.cache_resource
    def get_spark():
        return create_spark_session("exercise_2_forecasting")

    # Cached data loading & preparation
    @st.cache_data
    def load_and_prepare(data_path, session_gap_sec):
        # Obtain the cached Spark session inside the cached function so we don't
        # pass an unhashable SparkSession object as an argument to st.cache_data.
        spark = get_spark()
        track_list = load_track_data(spark, data_path)
        df_sessions = add_sessions_id_columns(track_list, session_gap_sec)
        df_pandas, top_user, n_sessions = prepare_top_user_monthly_sessions_for_forecasting(df_sessions)
        return df_pandas, top_user, n_sessions

    with st.sidebar:
        st.header("Settings")
        forecast_periods = st.number_input("Forecast periods (months)", min_value=1, max_value=12, value=DEFAULT_FORECAST_PERIODS)
        run_train = st.button("Train ML Models")
        run_sarimax = st.button("Run SARIMAX Forecast")
        cleanup = st.button("Stop Spark (cleanup)")

    # Data load
    with st.spinner("Loading data and preparing top user..."):
        df_pandas, top_user, n_sessions = load_and_prepare(DATA_PATH, SESSION_GAP_SEC)

    st.subheader("Top User Summary")
    st.write(f"Top User: {top_user} — Number of Sessions: {n_sessions}")
    st.subheader("Monthly Aggregated Data (tail)")
    st.dataframe(df_pandas.tail(10))

    # Train/Test split preview
    X_train, y_train, X_test, y_test = prepare_train_test_split(df_pandas)
    st.write(f"Training set size: {len(X_train)}  —  Test set size: {len(X_test)}")

    # Train ML models (on demand)
    if run_train:
        with st.spinner("Training ML models..."):
            models_config = get_default_models_config()
            results_df = train_multiple_models(X_train, y_train, X_test, y_test, models_config)
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)
        # show model comparison plot if available
        try:
            plot_model_comparison(results_df)
            st.pyplot()  # many project plotting helpers render to matplotlib
        except Exception:
            st.write("Model comparison plot could not be rendered by helper function.")

        # Best model metrics and ML forecast
        best_model_name = results_df.iloc[0]["model"]
        best_params = results_df.iloc[0]["best_params"]
        st.write(f"Best Model: {best_model_name}")
        st.write(f"Best Parameters: {best_params}")
        st.write(f"RMSE: {results_df.iloc[0]['rmse']:.2f}")
        st.write(f"R²: {results_df.iloc[0]['r2']:.4f}")

        # Fit best model and forecast future
        best_model = Ridge(random_state=42, **best_params)
        last_month = int(df_pandas["count_months"].max())
        future_t, predictions = forecast_future_periods(best_model, X_train, y_train, last_month, forecast_periods)

        # Create forecast dataframe
        last_date = df_pandas["timestamp_month"].max()
        forecast_df_ml = create_forecast_dataframe(last_date, future_t, predictions)
        st.subheader("ML Model Forecast")
        st.dataframe(forecast_df_ml)
        st.write("Predicted Average Session Duration (seconds -> minutes):")
        for i, pred in enumerate(predictions, 1):
            st.write(f"Month +{i}: {pred:.2f} sec ({pred/60:.2f} min)")

    # SARIMAX (on demand)
    if run_sarimax:
        with st.spinner("Running SARIMAX forecast and evaluation..."):
            df_ts = df_pandas[["timestamp_month", "avg_session_duration_sec"]].set_index("timestamp_month")
            stationarity = check_stationarity(df_ts["avg_session_duration_sec"])
            results_sarimax, forecast_sarimax, metrics_sarimax = forecast_with_sarimax(
                df_ts,
                forecast_periods=forecast_periods,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
            )

        st.subheader("SARIMAX Results")
        st.write(f"ADF Statistic: {stationarity['adf_statistic']:.4f}")
        st.write(f"p-value: {stationarity['p_value']:.4f}")
        st.write(f"Stationary: {'Yes' if stationarity['is_stationary'] else 'No'}")
        st.write(f"MAE: {metrics_sarimax['mae']:.2f}  —  MSE: {metrics_sarimax['mse']:.2f}  —  RMSE: {np.sqrt(metrics_sarimax['mse']):.2f}")
        st.subheader("SARIMAX - Future Predictions")
        for i, (date, pred) in enumerate(forecast_sarimax.items(), 1):
            st.write(f"Month +{i} ({date.strftime('%Y-%m')}): {pred:.2f} sec ({pred/60:.2f} min)")

        # Try to plot forecast using project helper
        try:
            plot_forecast(df_ts, forecast_sarimax, title="SARIMAX Forecast - Average Session Duration", ylabel="Duration (seconds)")
            st.pyplot()
        except Exception:
            st.write("Forecast plot could not be rendered by helper function.")

    # Cleanup
    if cleanup:
        try:
            spark = get_spark()
            spark.stop()
            st.success("Spark stopped.")
        except Exception as e:
            st.error(f"Error stopping Spark: {e}")

    # Minimal hint for local run
    st.info("Run this app with: streamlit run notebooks/forecast_session_duration_top_user_app.py")


if __name__ == '__main__':
    run()