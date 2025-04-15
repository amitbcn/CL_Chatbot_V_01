import streamlit as st
import sys
import os

# Add the project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required functions
from helper_functions.database_upload.postgres_uploader import *

# --- Configuration ---
# Correct path to YAML file (in the root folder)
yaml_file_path = os.path.join(project_root,"..", "testing_mapping.yaml")


# Create a connection engine
engine = create_postgres_engine('postgres', 'postgres', 'localhost', 5432, 'cl_survey_data')

# Fetch wave list
wave_list = get_wave_n_list(engine)

# Dropdown to select wave
wave_name = st.selectbox("Select a wave:", ["Select a wave..."] + wave_list)

if wave_name == "Select a wave...":
    st.warning("Please select a wave to continue.")
    st.stop()


# Load the dataset using the YAML mapping
data_dict = load_full_survey_dataset(engine, wave_name, yaml_file_path, max_chunks=100)
for key, df in data_dict.items():
    st.session_state[key] = df
    print(key)

# Display keys
st.subheader("Available Data Sections")
st.write(list(data_dict.keys()))

# Display each DataFrame
st.subheader("Loaded DataFrames")
for key, df in data_dict.items():
    st.markdown(f"### {key}")
    st.dataframe(df)
