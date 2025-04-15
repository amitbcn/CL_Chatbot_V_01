import pandas as pd
import yaml



def read_yaml(file_path):
    """Reads a YAML file and returns the data as a Python dictionary."""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        #st.warning(f"File not found: {file_path}")
        return {}
    except yaml.YAMLError as e:
        #st.warning(f"YAML parsing error: {e}")
        return {}
    

def categorize(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Applies categorization filters to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to apply filters to.
        metadata (dict): A dictionary containing bucketing metadata.

    Returns:
        pd.DataFrame: A new DataFrame with bucketing filters applied.

    Raises:
        ValueError: If `df` is not a valid DataFrame or `metadata` is not a dictionary.
    """
    #Basic checks, raise error if not correct object type
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not isinstance(metadata, dict):
        raise ValueError("Input `metadata` must be a dictionary.")

    df = df.copy()
    
    survey_question = metadata.get("survey_question")
    filter_name = metadata.get("filter_name")
    if 'mapping' in metadata.keys():
        mapping = metadata.get("mapping")

        if survey_question not in df.columns:
            raise KeyError(f"Column '{survey_question}' not found in DataFrame.")

        if not isinstance(mapping, dict):
            raise ValueError("`mappings` must be a dictionary.")

        # Initialize new column with original values

        # Apply numeric range mapping
        for key, value in mapping.items():
            if (isinstance(value, list) and
                len(value) == 2 and
                all(isinstance(v, str) and v.isdigit() for v in value)):
                min_val = float(value[0])
                max_val = float(value[1])
                df.loc[df[survey_question].between(min_val, max_val), filter_name] = key

            elif (isinstance(value, list) and
                len(value) == 1 and
                all(isinstance(v, (str, int)) and v.isdigit() for v in value)):
                numeric_input = int(value[0])
                df.loc[df[survey_question] == numeric_input, filter_name] = key

            elif isinstance(value, list):
                df.loc[df[survey_question].isin(value), filter_name] = key

    return df

def recreate(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """recreates a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        metadata (dict): A dictionary containing metadata with keys:
            - 'survey_question': The column name to duplicate.
            - 'filter_name': The name of the new duplicated column.

    Returns:
        pd.DataFrame: A new DataFrame with the duplicated column.

    Raises:
        ValueError: If `df` is not a valid DataFrame or `metadata` is not a dictionary.
        KeyError: If `survey_question` is not found in DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not isinstance(metadata, dict):
        raise ValueError("Input `metadata` must be a dictionary.")
    
    survey_question = metadata.get("survey_question")
    filter_name = metadata.get("filter_name")
    
    if survey_question not in df.columns:
        raise KeyError(f"Column '{survey_question}' not found in DataFrame.")
    
    df = df.copy()
    df[filter_name] = df[survey_question]
    
    return df

def add_filtering_columns(df: pd.DataFrame, default_filters: dict) -> pd.DataFrame:
    """Adds filtering columns based on configuration."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input `df` must be a pandas DataFrame.")
    if not isinstance(default_filters, dict):
        raise ValueError("Input `categorization_configs` must be a dictionary.")
    
    for config in default_filters.values():
        if config["filter_type"] == "categorize":
            df = categorize(df, config)
        elif config["filter_type"] == "recreate":
            df = recreate(df, config)
    
    return df

