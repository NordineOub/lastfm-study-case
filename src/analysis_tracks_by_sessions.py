from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def compute_session_duration(df_sessions: DataFrame) -> DataFrame:
    """Return one row per (userid, session_id) with session duration in seconds."""
    return df_sessions.groupBy("userid", "session_id").agg(
        (F.unix_timestamp(F.max("timestamp")) - F.unix_timestamp(F.min("timestamp"))).alias(
            "session_duration_sec"
        )
    )


def top_tracks_from_longest_sessions(
    df_sessions: DataFrame,
    top_n_sessions: int,
    top_n_tracks: int 
) -> DataFrame:
    """Compute top N tracks from the longest N sessions.
    Based on userid, session_id, timestamp, and track_col.
    """
    session_duration = compute_session_duration(df_sessions)

    top_sessions = (
        session_duration.orderBy(F.col("session_duration_sec").desc())
        .limit(top_n_sessions)
        .select("userid", "session_id")
    )

    top_tracks = (
        df_sessions.join(top_sessions, on=["userid", "session_id"], how="inner")
        .groupBy("track_name")
        .agg(F.count("*").alias("play_count"))
        .orderBy(F.col("play_count").desc())
        .limit(top_n_tracks)
    )
    return top_tracks
