
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


def prepare_train_test_split(
    df: pd.DataFrame,
    feature_col: str = "count_months",
    target_col: str = "avg_session_duration_sec",
    train_ratio: float = 0.67
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        feature_col: Name of feature column
        target_col: Name of target column
        train_ratio: Ratio of training data (0-1)
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    split_idx = int(train_ratio * len(df))
    train_set, test_set = np.split(df, [split_idx])
    
    X_train = train_set[[feature_col]]
    y_train = train_set[target_col]
    X_test = test_set[[feature_col]]
    y_test = test_set[target_col]
    
    return X_train, y_train, X_test, y_test


def get_default_models_config() -> Dict:
    """
    Return default configuration for ML models with hyperparameter grids.
    
    Returns:
        Dictionary mapping model names to (model, param_grid) tuples
    """
    return {
        "LinearRegression": (
            LinearRegression(),
            {"fit_intercept": [True, False], "copy_X": [True]},
        ),
        "Ridge": (
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]}
        ),
        "Lasso": (
            Lasso(max_iter=5000),
            {"alpha": [0.001, 0.01, 0.1, 1.0]}
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3]
            },
        ),
        "SVR": (
            SVR(),
            {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"]
            },
        ),
    }


def train_multiple_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_and_params: Dict
) -> pd.DataFrame:
    """
    Train multiple ML models with GridSearchCV and evaluate them.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models_and_params: Dict of {model_name: (model, param_grid)}
    
    Returns:
        DataFrame with results sorted by RMSE (best first)
    """
    results = []
    
    for name, (model, param_grid) in models_and_params.items():
        print(f"Training {name}...")
        
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "model": name,
            "best_params": gs.best_params_,
            "rmse": rmse,
            "r2": r2
        })
    
    results_df = pd.DataFrame(results).sort_values(by="rmse")
    return results_df


def forecast_future_periods(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    last_month_value: int,
    n_periods: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train model and forecast future periods.
    
    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training target
        last_month_value: Last month counter value
        n_periods: Number of periods to forecast
        
    Returns:
        Tuple of (future_t, predictions)
    """
    model.fit(X_train, y_train)
    future_t = np.array([[last_month_value + i] for i in range(1, n_periods + 1)])
    predictions = model.predict(future_t)
    return future_t, predictions


def create_forecast_dataframe(
    last_date: pd.Timestamp,
    future_t: np.ndarray,
    predictions: np.ndarray
) -> pd.DataFrame:
    """
    Create a formatted forecast DataFrame.
    
    Args:
        last_date: Last date in the original series
        future_t: Future time values
        predictions: Predicted values
        
    Returns:
        DataFrame with month, t, and forecast columns
    """
    future_months = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=len(predictions),
        freq="MS"
    )
    
    forecast_df = pd.DataFrame({
        "month": future_months,
        "t": future_t.ravel(),
        "forecast_avg_session_duration_sec": predictions
    })
    
    return forecast_df
