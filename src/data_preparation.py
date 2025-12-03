"""
Data preparation functions for LastFM dataset analysis.
Handles PySpark transformations for session-based analysis.
"""

from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def get_top_user_and_session_count(df_sessions: DataFrame) -> Tuple[str, int]:
    """
    Find the user with the most sessions.
    
    Args:
        df_sessions: DataFrame with userid and session_id columns
        
    Returns:
        Tuple of (userid, session_count)
    """
    user_session_counts = (
        df_sessions.select("userid", "session_id")
        .distinct()
        .groupBy("userid")
        .agg(F.count("session_id").alias("n_sessions"))
    )
    
    top_user_row = user_session_counts.orderBy(F.col("n_sessions").desc()).first()
    return top_user_row["userid"], int(top_user_row["n_sessions"])


def compute_user_sessions_with_duration(df_sessions: DataFrame, userid: str) -> DataFrame:
    """
    Compute session start times and durations for a specific user.
    
    Args:
        df_sessions: DataFrame with session data
        userid: User ID to filter for
        
    Returns:
        DataFrame with session_start and session_duration_sec columns
    """
    user_sessions = (
        df_sessions.filter(F.col("userid") == userid)
        .groupBy("userid", "session_id")
        .agg(
            F.min("timestamp").alias("session_start"),
            (F.max("timestamp").cast("long") - F.min("timestamp").cast("long")).alias(
                "session_duration_sec"
            ),
        )
    )
    return user_sessions


def aggregate_sessions_by_month(user_sessions: DataFrame) -> DataFrame:
    """
    Aggregate user sessions by month with average duration.
    
    Args:
        user_sessions: DataFrame with session data
        
    Returns:
        DataFrame with monthly aggregated session durations
    """
    user_sessions_monthly = (
        user_sessions.withColumn("timestamp_month", F.date_trunc("month", "session_start"))
        .groupBy("timestamp_month")
        .agg(F.avg("session_duration_sec").alias("avg_session_duration_sec"))
        .orderBy("timestamp_month")
    )
    return user_sessions_monthly


def add_month_counter(user_sessions_monthly: DataFrame) -> DataFrame:
    """
    Add a sequential month counter column (0, 1, 2, ...).
    
    Args:
        user_sessions_monthly: DataFrame with monthly session data
        
    Returns:
        DataFrame with added count_months column
    """
    w = Window.orderBy("timestamp_month")
    user_sessions_monthly_count = user_sessions_monthly.withColumn(
        "count_months", F.row_number().over(w) - 1
    )
    return user_sessions_monthly_count


def prepare_top_user_monthly_sessions_for_forecasting(df_sessions: DataFrame) -> Tuple:
    """
    Prepare monthly aggregated session data for the top user (by session count) for forecasting.
    
    This function supports Exercise 2: Forecast the next 3 months of average session duration
    for the user with the highest number of sessions.
    
    Args:
        df_sessions: Raw session DataFrame with userid, session_id, and timestamp columns
        
    Returns:
        Tuple of (pandas_df, top_user_id, n_sessions) where:
            - pandas_df: Monthly aggregated data with columns [timestamp_month, count_months, avg_session_duration_sec]
            - top_user_id: User ID with the most sessions
            - n_sessions: Total number of sessions for the top user
    """
    # Find top user
    top_user, top_user_n_sessions = get_top_user_and_session_count(df_sessions)
    
    # Compute sessions with duration
    user_sessions = compute_user_sessions_with_duration(df_sessions, top_user)
    
    # Aggregate by month
    user_sessions_monthly = aggregate_sessions_by_month(user_sessions)
    
    # Add month counter
    user_sessions_monthly_count = add_month_counter(user_sessions_monthly)
    
    # Convert to Pandas
    df_pandas = user_sessions_monthly_count.select(
        "timestamp_month", "count_months", "avg_session_duration_sec"
    ).toPandas()
    
    return df_pandas, top_user, top_user_n_sessions
