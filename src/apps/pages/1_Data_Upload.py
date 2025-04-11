# 1_Data_Upload.py

import sys
import os
import streamlit as st
import pandas as pd

#  Add project root to sys.path for reliable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

#  Import from your processor module
from helper_functions.data_pre_processing.data_upload_processor.processor import run_question_data_pipeline
from helper_functions.database_upload.postgres_uploader import *

#  Streamlit page setup
st.set_page_config(page_title="Upload Data")
st.title("Upload Raw Data")

#  Input source name via form
with st.form("wave_source_form"):
    wave_input = st.text_input("Enter a source name (e.g., wave_10):", value="wave_10")
    submitted = st.form_submit_button("Submit Source")

#  Store in session state if submitted
if submitted:
    if not wave_input.strip():
        st.warning("Please enter a valid wave/source name.")
        st.stop()
    st.session_state.wave_source = wave_input.strip()
    st.success(f"Source set to: {st.session_state.wave_source}")

#  Block further execution if source isn't set yet
if "wave_source" not in st.session_state:
    st.info("Please enter a source name to continue.")
    st.stop()

#  API key check
if "api_key" not in st.session_state:
    st.warning("No API key found.")
    st.stop()

#  Upload raw data
raw_data = st.file_uploader("Choose a raw data Excel file", type="xlsx", key="raw_data_uploader")
if raw_data:
    df = pd.read_excel(raw_data)
    st.session_state.raw_data = df
    st.success("Raw data uploaded successfully!")
    st.dataframe(df)

#  Upload data map
data_map = st.file_uploader("Choose a data map Excel file", type="xlsx", key="data_map_uploader")
if data_map:
    df = pd.read_excel(data_map)
    st.session_state.data_map = df
    st.success("Data map uploaded successfully!")
    st.dataframe(df)

#  Run pipeline once everything is uploaded
required_keys = ["raw_data", "data_map", "api_key", "wave_source"]
if all(k in st.session_state for k in required_keys):
    with st.spinner("Processing data..."):
        try:
            data_dict = run_question_data_pipeline(
                st.session_state.data_map,
                st.session_state.raw_data,
                st.session_state.api_key,
                source=st.session_state.wave_source  # 👈 use submitted value
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
                st.json(value[:3])
                st.caption(f"{len(value)} items in list")
            else:
                st.warning("This list is empty.")
        else:
            st.write(value)


# -------------------- Enhanced PostgreSQL Upload Section --------------------

if all(k in st.session_state for k in required_keys):


# ✅ Step 1: Create engine with feedback
    try:
        engine = create_postgres_engine('postgres', 'postgres', 'localhost', 5432, 'cl_survey_data')
        st.success("✅ Connected to PostgreSQL database.")
    except Exception as e:
        st.error(f"❌ Failed to connect to PostgreSQL: {e}")
        st.stop()

    # ✅ Step 2: Upload standard tables with checks and feedback
    table_map = {
        'embeddings_metadata_df': f"{st.session_state.wave_source}_embeddings_metadata",
        'type_subtype': f"{st.session_state.wave_source}_type_subtype",
        'question_guide': f"{st.session_state.wave_source}_question_guide"
    }

    for key, table_name in table_map.items():
        try:
            df = data_dict.get(key)
            if df is None:
                st.warning(f"⚠️ `{key}` is missing in processed data. Skipping `{table_name}` upload.")
                continue
            if df.empty:
                st.warning(f"⚠️ `{key}` is empty. Skipping `{table_name}` upload.")
                continue

            push_dataframe_to_postgres_db(df, table_name, engine)
            st.success(f"✅ Uploaded `{key}` to table `{table_name}`.")

        except Exception as e:
            st.error(f"❌ Failed to upload `{key}` to `{table_name}`: {e}")

    # ✅ Step 3: Upload raw data chunks (if available)
    try:
        mapped_df = data_dict.get("mapped_data")
        if mapped_df is None:
            st.error("❌ `mapped_data` missing. Cannot upload raw survey chunks.")
        elif mapped_df.empty:
            st.warning("⚠️ `mapped_data` is empty. Skipping raw survey upload.")
        else:
            process_raw_data(
                file=mapped_df,
                wave_name=st.session_state.wave_source,
                engine=engine,
                response_id_col='Respondent',
                chunk_size=60
            )
            st.success(f"✅ Raw data chunks uploaded for `{st.session_state.wave_source}`.")
    except Exception as e:
        st.error(f"❌ Failed to process raw data chunks: {e}")
