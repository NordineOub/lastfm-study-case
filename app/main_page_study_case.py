import streamlit as st
import importlib.util
from pathlib import Path

P = Path(__file__).parent

PAGES = {
    "Forecast — Top User (Exercise 2)": P / "forecast_session_duration_top_user_app.py",
    "Top Songs — Longest Sessions (Exercise 1)": P / "top_songs_longest_sessions_app.py",
}

st.set_page_config(page_title="LastFM Exercises", layout="wide")
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose a page", list(PAGES.keys()))

page_path = PAGES[choice]

# Load module from path and execute its `run()` or `main()` function
spec = importlib.util.spec_from_file_location(page_path.stem, str(page_path))
module = importlib.util.module_from_spec(spec)
loader = spec.loader
if loader is None:
    st.error(f"Could not load page: {page_path}")
else:
    try:
        loader.exec_module(module)
        # Prefer `run`, then `main` if available
        if hasattr(module, "run"):
            module.run()
        elif hasattr(module, "main"):
            module.main()
        else:
            st.error("Page module does not expose run() or main()")
    except Exception as e:
        st.error(f"Error while running page: {e}")
