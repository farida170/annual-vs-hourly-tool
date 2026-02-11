import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Health Check", layout="wide")
st.title("✅ Health Check — Annual vs Hourly Tool")

DATA_DIR = Path(__file__).parent / "data"
st.write("DATA_DIR:", str(DATA_DIR))
st.write("Exists:", DATA_DIR.exists())

if DATA_DIR.exists():
    files = sorted([p.name for p in DATA_DIR.glob("*")])
    st.write("Files in /data:", files)

    # Try reading a few files to confirm they exist + parse
    for name in files:
        p = DATA_DIR / name
        st.write("---")
        st.write("Reading:", name, "size (MB):", round(p.stat().st_size / 1e6, 2))
        if name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(p)
                st.write("Shape:", df.shape)
                st.dataframe(df.head(5))
            except Exception as e:
                st.error(f"Failed to read {name}: {e}")
else:
    st.error("data/ folder not found in repo deployment.")
