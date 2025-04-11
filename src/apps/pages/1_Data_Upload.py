# 1_Data_Upload.py

import sys
import os
import streamlit as st
import pandas as pd

# 🔧 Add project root to sys.path for reliable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 📦 Import from your processor module
from helper_functions.data_pre_processing.data_upload_processor.processor import run_question_data_pipeline

# 🚀 Streamlit page setup
st.set_page_config(page_title="Upload Data")
st.title("Upload Raw Data")

# 🔐 API key check
if "api_key" not in st.session_state:
    st.warning("No API key found.")
    st.stop()

# 📁 Upload raw data
raw_data = st.file_uploader("Choose a raw data Excel file", type="xlsx", key="raw_data_uploader")
if raw_data:
    df = pd.read_excel(raw_data)
    st.session_state.raw_data = df
    st.success("Raw data uploaded successfully!")
    st.dataframe(df)

# 📁 Upload data map
data_map = st.file_uploader("Choose a data map Excel file", type="xlsx", key="data_map_uploader")
if data_map:
    df = pd.read_excel(data_map)
    st.session_state.data_map = df
    st.success("Data map uploaded successfully!")
    st.dataframe(df)

# ✅ Run pipeline once everything is uploaded
required_keys = ["raw_data", "data_map", "api_key"]
if all(k in st.session_state for k in required_keys):
    with st.spinner("Processing data..."):
        try:
            data_dict = run_question_data_pipeline(
                st.session_state.data_map,
                st.session_state.raw_data,
                st.session_state.api_key,
                source="wave_10"
            )
            st.success("Data processed successfully!")
        except Exception as e:
            st.error(f"Error in data pipeline: {e}")
            st.stop()

    # 🔍 Show contents of data_dict
    for key, value in data_dict.items():
        st.header(key)

        if isinstance(value, pd.DataFrame):
            if not value.empty:
                st.dataframe(value.head())
                st.caption(f"{value.shape[0]} rows × {value.shape[1]} columns")
            else:
                st.warning("This section is empty.")
        elif isinstance(value, list):
            if value:
                st.json(value[:3])  # Preview first 3 items
                st.caption(f"{len(value)} items in list")
            else:
                st.warning("This list is empty.")
        else:
            st.write(value)
