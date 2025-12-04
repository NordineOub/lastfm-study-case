import sys
import streamlit as st
import os
# ensure project src is importable
sys.path.append('..')

from src.common.definition import (
    create_spark_session,
    load_track_data,
    add_sessions_id_columns,
)
from src.analysis_tracks_by_sessions import (
    top_tracks_from_longest_sessions,
    compute_session_duration,
)

# Configuration
DATA_PATH = os.getenv("DATA_PATH")
DEFAULT_SESSION_GAP_MIN = 20
DEFAULT_TOP_N_SESSIONS = 50
DEFAULT_TOP_N_TRACKS = 10

st.set_page_config(page_title="Top Songs - Longest Sessions", layout="wide")
st.title("Top 10 Songs in Top 50 Longest Sessions")
st.markdown("""
What are the top 10 songs played in the top 50 longest sessions by tracks count?

A user session is defined as consecutive plays where each play starts within
`SESSION_GAP_MIN` minutes of the previous play.
""")

@st.cache_resource
def get_spark():
    return create_spark_session("exercise_1_top_songs")

@st.cache_data
def prepare_data(data_path, session_gap_min, top_n_sessions, top_n_tracks):
    # Create/get spark inside cached function to avoid passing unhashable objects
    spark = get_spark()

    track_list = load_track_data(spark, data_path)
    total_records = int(track_list.count())

    df_sessions = add_sessions_id_columns(track_list, int(session_gap_min * 60))
    df_sessions = df_sessions.select("userid", "timestamp", "track_name", "session_id")

    # Session durations (Spark DF -> pandas)
    session_durations = compute_session_duration(df_sessions)
    session_durations_pd = session_durations.orderBy("session_duration_sec", ascending=False).toPandas()

    # Top tracks from longest sessions
    top_tracks = top_tracks_from_longest_sessions(df_sessions, top_n_sessions, top_n_tracks)
    top_tracks_pd = top_tracks.toPandas()

    # A small sample of sessions for preview
    sample_sessions_pd = df_sessions.limit(10).toPandas()

    return {
        "total_records": total_records,
        "sample_sessions": sample_sessions_pd,
        "session_durations": session_durations_pd,
        "top_tracks": top_tracks_pd,
    }

def main():
    with st.sidebar:
        st.header("Settings")
        session_gap_min = st.number_input("Session gap (minutes)", min_value=1, max_value=120, value=DEFAULT_SESSION_GAP_MIN)
        top_n_sessions = st.number_input("Top N sessions to consider", min_value=1, max_value=500, value=DEFAULT_TOP_N_SESSIONS)
        top_n_tracks = st.number_input("Top N tracks to return", min_value=1, max_value=100, value=DEFAULT_TOP_N_TRACKS)
        run_analysis = st.button("Run Analysis")
        cleanup = st.button("Stop Spark (cleanup)")

    if run_analysis:
        with st.spinner("Loading data and computing results..."):
            results = prepare_data(DATA_PATH, session_gap_min, top_n_sessions, top_n_tracks)

        st.subheader("Dataset Summary")
        st.write(f"Total records loaded: {results['total_records']:,}")

        st.subheader("Sample Sessions")
        st.dataframe(results["sample_sessions"])

        st.subheader("Session Durations (Top)")
        st.write("Showing sessions ordered by duration (seconds).")
        st.dataframe(results["session_durations"].head(50))

        st.subheader(f"Top {top_n_tracks} Songs in Top {top_n_sessions} Longest Sessions")
        st.dataframe(results["top_tracks"])

    else:
        st.info("Set parameters in the sidebar and click 'Run Analysis' to compute results.")

    if cleanup:
        try:
            spark = get_spark()
            spark.stop()
            st.success("Spark stopped.")
        except Exception as e:
            st.error(f"Error stopping Spark: {e}")

    st.markdown("---")
    st.info("Run this page locally with: `streamlit run notebooks/top_songs_longest_sessions_app.py`")

if __name__ == '__main__':
    main()
