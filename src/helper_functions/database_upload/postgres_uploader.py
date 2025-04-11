import pandas as pd
from typing import Tuple, Dict, List
import streamlit as st
import os
from sqlalchemy import inspect
import yaml

import re
from typing import List
from sqlalchemy.engine import Engine
from sqlalchemy import inspect
yaml_file_path =  'testing_mapping.yaml'
#os.path.join(PROJECT_DIR, "config", "mapping.yaml")


from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Optional

# -------------------- Database Connection --------------------

def create_postgres_engine(
    username: str = os.getenv("POSTGRES_USER"),
    password: str = os.getenv("POSTGRES_PASSWORD"),
    host: str = 'localhost',
    port: int = 5432,
    database: str = os.getenv("POSTGRES_DB"),
    echo: Optional[bool] = False
) -> Engine:
    """
    Create a SQLAlchemy engine for a PostgreSQL database.

    Parameters:
        username (str): The username used to authenticate with the PostgreSQL database.
        password (str): The password used to authenticate with the PostgreSQL database.
        host (str): The hostname or IP address of the PostgreSQL server.
        port (int): The port number on which the PostgreSQL server is listening.
        database (str): The name of the PostgreSQL database to connect to.
        echo (bool, optional): If True, the engine will log all statements. Default is False.

    Returns:
        Engine: A SQLAlchemy Engine object that can be used to interact with the database.

    Example:
        engine = create_postgres_engine(
            username="myuser",
            password="mypassword",
            host="localhost",
            port=5432,
            database="mydatabase"
        )
    """
    # Format the PostgreSQL connection URL
    connection_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"

    # Create the SQLAlchemy engine
    engine = create_engine(connection_url, echo=echo)

    return engine

# -------------------- Wave List Extraction --------------------

def get_wave_n_list(engine: Engine) -> List[str]:
    """
    Retrieves a sorted list of unique wave identifiers from the database tables.

    A wave identifier is expected to match the pattern 'wave_<number>_' at the beginning of the table name.
    For example, from 'wave_1_survey' or 'wave_2_data', the function extracts 'wave_1', 'wave_2', etc.

    Parameters:
    -----------
    engine : sqlalchemy.engine.Engine
        A SQLAlchemy engine connected to the target database.

    Returns:
    --------
    List[str]
        A sorted list of unique wave identifiers found in the table names.
    """
    # Create an inspector to introspect the database schema
    inspector = inspect(engine)

    # Retrieve all table names in the connected database
    table_names = inspector.get_table_names()

    # Regex pattern to match 'wave_<number>_' at the start of table names
    wave_pattern = re.compile(r'^(wave_\d+)_')

    wave_set = set()

    # Iterate through table names and extract wave identifiers
    for table in table_names:
        match = wave_pattern.match(table)
        if match:
            wave_set.add(match.group(1))  # Add the matched wave identifier to the set

    # Return the sorted list of unique wave identifiers
    return sorted(wave_set)

# -------------------- Column Sanitization --------------------

def sanitize_all_columns(
    df: pd.DataFrame,
    max_pg_col_length: int = 63,
    upper_limit: int = 1000
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Sanitizes all column names in a DataFrame to ensure they do not exceed the maximum allowed length.
    If a column name is too long, it is shortened using an encoding scheme, and a mapping is created.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame whose column names need to be sanitized.
    max_pg_col_length : int, optional
        The maximum allowed length for PostgreSQL column names (default is 63).
    upper_limit : int, optional
        The maximum number of unique encodings allowed for shortening (default is 1000).

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, str]]
        - A new DataFrame with sanitized column names.
        - A dictionary mapping encoded column names to their original names.
    """
    encoded_mapping: Dict[str, str] = {}  # Stores mapping from new to original names
    used_encoding = 0  # Counter for generating unique codes

    new_columns = []
    for col in df.columns:
        # Sanitize each column name and update encoding reference
        sanitized_col, used_encoding = sanitize_column(
            col,
            used_encoding,
            max_pg_col_length,
            upper_limit,
            encoded_mapping
        )
        new_columns.append(sanitized_col)

    # Apply the new column names to a copy of the DataFrame
    sanitized_df = df.copy()
    sanitized_df.columns = new_columns

    return sanitized_df, encoded_mapping

def generate_code(n: int, upper_limit: int) -> str:
    """
    Generates a zero-padded 3-digit string code from an integer,
    ensuring it does not exceed a specified upper limit.

    Parameters:
    -----------
    n : int
        The current encoding index.
    upper_limit : int
        The maximum allowed value for the encoding index.

    Returns:
    --------
    str
        A zero-padded string representing the code (e.g., '001', '045').

    Raises:
    -------
    ValueError:
        If the encoding index exceeds the specified upper limit.
    """
    if n >= upper_limit:
        raise ValueError(f"Exceeded maximum of {upper_limit} unique encodings.")
    
    # Format the number as a zero-padded 3-digit string
    return f"{n:03}"

def sanitize_column(
    col: str,
    used_encoding_ref: int,
    max_pg_col_length: int,
    upper_limit: int,
    encoded_mapping: Dict[str, str]
) -> Tuple[str, int]:
    """
    Shortens a column name if it exceeds the maximum allowed length, using a consistent encoding scheme.

    Parameters:
    -----------
    col : str
        The original column name.
    used_encoding_ref : int
        A counter/reference number used to generate unique short codes.
    max_pg_col_length : int
        The maximum allowed length for PostgreSQL column names.
    upper_limit : int
        The upper limit for the encoding reference; used in code generation logic.
    encoded_mapping : Dict[str, str]
        A dictionary to store the mapping of new column names to original column names.

    Returns:
    --------
    Tuple[str, int]
        A tuple containing:
            - The sanitized (or original) column name.
            - The updated encoding reference counter.

    Raises:
    -------
    ValueError:
        If the column does not match the expected pattern for encoding (e.g., not starting with 'Q<number>').
    """
    # If column length is within the allowed limit, return as is
    if len(col) <= max_pg_col_length:
        return col, used_encoding_ref

    # Match a pattern like 'Q1', 'Q123', etc. at the start of the column name
    match = re.match(r"(Q\d+)", col)
    if not match:
        raise ValueError(f"Column '{col}' does not match expected format for encoding.")

    # Extract question number and generate a unique short code
    question_number = match.group(1)
    code = generate_code(used_encoding_ref, upper_limit)

    # Construct new column name and store mapping
    new_col = f"{question_number}_{code}"
    encoded_mapping[new_col] = col

    # Return new column name and incremented encoding reference
    return new_col, used_encoding_ref + 1

# -------------------- Survey Data Chunking --------------------

def get_chunks(
    df: pd.DataFrame,
    response_id_col: str,
    chunk_size: int = 60
) -> List[pd.DataFrame]:
    """
    Splits a DataFrame into smaller chunks based on a specified number of data columns,
    while retaining the respondent identifier column in each chunk.

    Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        response_id_col (str): The name of the column that uniquely identifies each respondent.
        chunk_size (int, optional): Maximum number of data columns per chunk (excluding the ID column). Defaults to 60.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each containing the ID column and a subset of the data columns.

    Raises:
        ValueError: If the response ID column is not found in the input DataFrame.
    """
    # Check if the response ID column exists in the DataFrame
    if response_id_col not in df.columns:
        raise ValueError(f"{response_id_col} column not found in DataFrame")

    # Exclude the ID column from data columns to be chunked
    data_cols = [col for col in df.columns if col != response_id_col]

    # If the number of data columns is small enough, return the DataFrame as a single chunk
    if len(data_cols) <= chunk_size:
        print("[DEBUG] Data columns are fewer than or equal to chunk size. No splitting needed.")
        return [df[[response_id_col] + data_cols]]

    chunks = []
    # Loop through the data columns in steps of chunk_size
    for i in range(0, len(data_cols), chunk_size):
        # Get a subset of columns for the current chunk
        chunk_cols = data_cols[i:i + chunk_size]
        # Create a new DataFrame with the ID column and chunk columns
        chunk_df = df[[response_id_col] + chunk_cols]
        chunks.append(chunk_df)

    return chunks

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Parameters:
    -----------
    file_path : str
        Path to the YAML file.

    Returns:
    --------
    dict
        The contents of the YAML file as a dictionary.
    
    Raises:
    -------
    FileNotFoundError: If the file does not exist.
    yaml.YAMLError: If there is an error parsing the YAML.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"[ERROR] Failed to parse YAML file: {e}")
    
def update_yaml_mapping(wave_key: str, new_mapping: dict, yaml_path: str = yaml_file_path):
    """
    Updates a master YAML configuration file with a new column mapping for a specific data wave.

    Parameters:
        wave_key (str): The key representing the data wave (e.g., "wave_1", "wave_2").
                        Must match the pattern "wave_<number>".
        new_mapping (dict): A dictionary representing the mapping of original to sanitized column names.
        yaml_path (str, optional): File path to the YAML configuration file. Defaults to `yaml_file_path`.

    Returns:
        None. Writes updated mappings directly to the YAML file and provides Streamlit feedback.
    """
    try:
        # Validate wave key format using regex (e.g., must be "wave_1", "wave_2", etc.)
        if not re.match(r'^wave_\d+$', wave_key):
            st.error("Wave name must follow the format: wave_1, wave_2, etc.")
            return

		
        # Load existing YAML configuration if the file exists, otherwise start with an empty dictionary
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
                
        else:
            config_data = {}

        # Update or insert the new mapping under the specified wave key
        config_data[wave_key] = new_mapping

        # Write the updated configuration back to the YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)

        # Provide confirmation to the user via Streamlit
        st.success(f"Mapping successfully saved for {wave_key}")
    
    except Exception as e:
        # Show an error message in case of any failure during reading or writing
        st.error(f"Failed to update YAML: {e}")

# -------------------- Data Upload Orchestration --------------------

def process_raw_data(file: pd.DataFrame, wave_name: str, engine: Engine, response_id_col: str = "Respondent", chunk_size: int = 60, yaml_path: str = yaml_file_path):
    """
    Processes raw survey data from a DataFrame, sanitizes column names, updates column mappings, 
    splits data into chunks (if necessary), and stores each chunk in a SQL database.

    Parameters:
        file (pd.DataFrame): The raw input data.
        wave_name (str): The identifier for the current data wave, used for naming database tables.
        engine (Engine): SQLAlchemy engine used to connect to the database.
        response_id_col (str, optional): The name of the column that uniquely identifies respondents. Defaults to "Respondent".
        chunk_size (int, optional): Maximum number of columns per chunk to store in the database. Defaults to 60.

    Returns:
        None. Stores processed data directly to the database and logs progress to Streamlit UI.
    """
    try:
        # Create a copy of the original DataFrame to avoid modifying it in-place
        df = file.copy()
        st.info(f"Raw data loaded with shape {df.shape}")

        # Sanitize column names and retrieve a mapping of old to new names
        sanitized_df, mapping = sanitize_all_columns(df)

        # Save the column mapping to a YAML config file for future reference
        update_yaml_mapping(wave_name, mapping, yaml_file_path)

        # Prepare the base name for the SQL table
        base_table_name = wave_name + "_raw_data"

        # Split the DataFrame into smaller chunks based on respondent ID
        chunks = get_chunks(sanitized_df, response_id_col, chunk_size)
        is_single_chunk = len(chunks) == 1

        # Loop through each chunk and store it in the SQL database
        for idx, chunk_df in enumerate(chunks, start=1):
            # Use base name for single chunk; append index for multiple chunks
            table_name = base_table_name if is_single_chunk else f"{base_table_name}_{idx}"

            # Write chunk to the database, replacing any existing table with the same name
            chunk_df.to_sql(table_name, con=engine, index=False, if_exists='replace')

        # Inform the user of successful completion
        st.success(f"Raw data processed and stored under table: {base_table_name}")
    
    except Exception as e:
        # Display an error message in case anything goes wrong during the process
        st.error(f"Error processing raw data: {e}")

def push_dataframe_to_postgres_db(file: pd.DataFrame, file_name: str, engine: Engine):
    """
    Processes a pandas DataFrame and stores it in the PostgreSQL database using the provided engine.

    Parameters:
        file (pd.DataFrame): The DataFrame containing the data to be stored.
        file_name (str): The name used as the base for the SQL table name.
        engine (Engine): A SQLAlchemy Engine connected to the target PostgreSQL database.

    Returns:
        None

    Side Effects:
        - Displays messages in the Streamlit app.
        - Writes the DataFrame to a SQL table in the database.
    """
    try:
        # Make a copy of the DataFrame to avoid modifying the original
        df = file.copy()

        # Display info message in Streamlit with the shape of the DataFrame
        st.info(f"Data loaded with shape {df.shape}")

        # Define the table name using the provided file_name
        table_name = f"{file_name}"

        # Save the DataFrame to the PostgreSQL database
        df.to_sql(table_name, con=engine, index=False, if_exists='replace')

        # Confirm success to the user
        st.success(f"Data saved under table: {table_name}")

    except Exception as e:
        # Display error message in Streamlit if something goes wrong
        st.error(f"Error processing data: {e}")

# -------------------- Data Download Orchestration --------------------

def combine_chunks(
    engine: Engine,
    base_table_name: str,
    max_chunks: int = 100
) -> pd.DataFrame:
    """
    Loads and combines chunked tables from a SQL database into a single DataFrame.

    Parameters:
        engine (Engine): SQLAlchemy engine for database connection.
        base_table_name (str): The base name of the tables to combine (e.g., "wave_1_raw_data").
                               Assumes chunked tables follow the pattern base_table_name_1, base_table_name_2, etc.
        max_chunks (int, optional): The maximum number of chunks to attempt to load. Defaults to 100.

    Returns:
        pd.DataFrame: The combined DataFrame containing all concatenated chunks.

    Raises:
        ValueError: If the base table (first chunk) cannot be read from the database.
    """
    combined_df = None
    st.write("Loading Survey Data!")
    base_table_name = base_table_name + "_raw_data"

    for idx in range(1, max_chunks + 1):
        # Construct the table name for the current chunk
        table_name = f"{base_table_name}_{idx}" if idx >= 1 else base_table_name

        try:
            # Attempt to load the chunk from the database
            chunk_df = pd.read_sql_table(table_name, con=engine)

            if combined_df is None:
                # Initialize combined_df with the first chunk
                combined_df = chunk_df
            else:
                # Drop the first column of the current chunk to avoid duplicate ID/index columns
                chunk_df = chunk_df.drop(columns=[combined_df.columns[0]])
                # Concatenate the current chunk horizontally
                combined_df = pd.concat([combined_df, chunk_df], axis=1)

        except Exception as e:
            if idx == 1:
                # If the first chunk fails to load, raise an error
                raise ValueError(f"[ERROR] Could not read base table '{table_name}': {e}")
            else:
                # If subsequent chunks are missing, stop the loop
                st.write("All data loaded.")
                break

    return combined_df

def restore_original_columns(
    df: pd.DataFrame,
    mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Restore the original column names of a DataFrame using a provided mapping.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame whose columns need to be renamed.
    mapping : Dict[str, str]
        A dictionary mapping current column names to their original names.
        Keys are the current names, and values are the original names.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with columns renamed to their original names where applicable.
    """
    # Generate the list of restored column names using the mapping
    restored_columns = [mapping.get(col, col) for col in df.columns]
    
    # Create a copy of the original DataFrame to avoid modifying it in-place
    restored_df = df.copy()
    
    # Rename the columns with the restored names
    restored_df.columns = restored_columns

    return restored_df

def load_related_tables(
    engine: Engine,
    base_table_name: str,
    suffixes: List[str] = ["_type_subtype", "_question_guide", "_embeddings_metadata"]
) -> Dict[str, pd.DataFrame]:
    """
    Loads multiple related tables by suffixing predefined strings to a base table name.

    Parameters:
    -----------
    engine : Engine
        SQLAlchemy engine connected to the target database.
    base_table_name : str
        The base name to which suffixes will be appended to form full table names.
    suffixes : List[str], optional
        List of suffixes to append to the base table name to form full table names.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary mapping each suffix (without underscore) to its loaded DataFrame.

    Raises:
    -------
    ValueError: If any of the tables fail to load.
    """
    tables = {}
    for suffix in suffixes:
        full_table_name = f"{base_table_name}{suffix}"
        try:
            df = pd.read_sql_table(full_table_name, con=engine)
            # Optional: use suffix without underscore as the key
            tables[suffix.lstrip('_')] = df
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load table '{full_table_name}': {e}")
    
    return tables

def load_full_survey_dataset(
    engine: Engine,
    base_table_name: str,
    mapping_path: str,
    max_chunks: int = 100,
    suffixes: list = ["_type_subtype", "_question_guide", "_embeddings_metadata"]
) -> Dict[str, pd.DataFrame]:
    """
    Loads a full survey dataset including combined chunked tables, restored column names,
    and additional related tables based on predefined suffixes.

    Parameters:
    -----------
    engine : Engine
        SQLAlchemy engine for database connection.
    base_table_name : str
        The base name of the chunked data tables.
    mapping_path : str
        File path to the YAML file containing column name mappings.
    max_chunks : int, optional
        Maximum number of chunked tables to combine. Default is 100.
    suffixes : list, optional
        List of suffixes to append to the base name for loading additional tables.

    Returns:
    --------
    Dict[str, pd.DataFrame]
        A dictionary containing:
            - 'data': Combined main survey data with restored column names.
            - Additional related tables as DataFrames (e.g., 'type_subtype', 'question_guide', etc.).
    """
    # Step 1: Read column mapping from YAML
    column_mapping = read_yaml_file(mapping_path)[base_table_name]

    # Step 2: Load and combine chunked tables
    combined_df = combine_chunks(engine, base_table_name, max_chunks=max_chunks)

    # Step 3: Restore original column names
    restored_df = restore_original_columns(combined_df, column_mapping)

    # Step 4: Load related tables
    related_tables = load_related_tables(engine, base_table_name, suffixes=suffixes)

    # Step 5: Combine all results into one dictionary
    return {
        "mapped_raw_data": restored_df,
        **related_tables
    }
