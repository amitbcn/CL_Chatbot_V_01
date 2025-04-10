import pandas as pd
import re

def question_guide_formatting(question_guide: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a question guide DataFrame by extracting relevant columns, renaming them,
    filtering based on question number format, and removing missing values.
    Args:
        question_guide (pd.DataFrame): Raw question guide DataFrame.
    Returns:
        pd.DataFrame: Processed question guide DataFrame.
    """
    question_guide = question_guide.iloc[:,0:2]  # Select relevant columns
    question_guide.columns = ['Options', 'Question']
    question_guide = question_guide.dropna(how = 'any') # Remove missing values
    return question_guide

def flatten_data_map(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and associates question identifiers with answers.
    Creates new columns for Question_code and Question_string. 
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with question codes and question strings filled.
    """
    df['Question_code'] = df['Options'].apply(lambda x: x if isinstance(x, str) and x.startswith('Q') and x[1:].isdigit() else None)
    df['Question_code'] = df['Question_code'].ffill()  # Forward fill question codes 
    df['Question_string'] = df.apply(lambda row: row['Question'] if row['Question_code'] == row['Options'] else None, axis=1)
    df['Question_string'] = df['Question_string'].ffill()  # Forward fill question strings
    return df

def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where the value is the same as the question string,
    except when the question_code occurs only once in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Count the occurrences of each question_code
    counts = df['Question_code'].value_counts()
    
    # Split into two groups
    single_row_qs = df[df['Question_code'].isin(counts[counts == 1].index)]
    multi_row_qs = df[df['Question_code'].isin(counts[counts > 1].index)]

    single_row_qs.loc[:, 'Options'] = "None"
    
    # Filter multi-row question_codes
    filtered_multi = multi_row_qs[multi_row_qs['Options'] != multi_row_qs['Question_code']]
    
    # Combine both parts
    return pd.concat([single_row_qs, filtered_multi], ignore_index=True)

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of the DataFrame for clarity.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df = df.rename(columns={'Options': 'answer_code', 'Question': 'answer_string'})
    return df[['Question_code', 'Question_string', 'answer_code', 'answer_string']]

#Orhcestrate entire question_guide creation
def process_question_guide(question_guide_map_df : pd.DataFrame) -> pd.DataFrame:
    """
    A wrapper function that processes a question guide DataFrame through multiple transformation steps.
    
    Args:
        question_guide_map_df (pd.DataFrame): The input DataFrame containing the question guide map.
    
    Returns:
        pd.DataFrame: The transformed DataFrame after applying all processing steps.
    """

    # Step 1: Format question guide
    question_guide = question_guide_formatting(question_guide_map_df)
    
    # Step 2: Extract question information
    flatten_data_map_df = flatten_data_map(question_guide)
    
    # Step 3: Filter rows
    filter_rows_df = filter_rows(flatten_data_map_df)
    
    # Step 4: Rename columns
    renamed_df = rename_columns(filter_rows_df)
    
    return renamed_df

# Mapping answer key to raw data
def mapping_answer_key_to_raw_data(raw_df: pd.DataFrame, question_guide : pd.DataFrame) -> pd.DataFrame:
        """Replaces encoded values in a DataFrame using the mapping DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing encoded survey responses.
            df (pd.Dataframe): The dataframe with the survey question mappings


        Returns:
            pd.DataFrame: A new DataFrame with replaced values.

        Raises:
            ValueError: If `df` is not a valid DataFrame.
        """
        if not isinstance(raw_df, pd.DataFrame):
            raise ValueError("Input `raw_df` must be a pandas DataFrame.")
        
        raw_df = raw_df.copy()
        raw_df.dropna(how='all')
        question_guide = question_guide[question_guide['answer_code'] != 'None']

        for col in raw_df.columns:
            if col.startswith("Q"):  # Process only relevant survey columns
                if raw_df[col].nunique(dropna=True) <= 2:  # Skip binary columns
                    continue

                match = re.match(r"(Q\d+)", col)
                question_code = match.group(1) if match else None
                if not question_code:
                    continue
                # Filter mapping_df for the relevant question
                relevant_mapping = question_guide[question_guide["Question_code"] == question_code].copy()


                if not relevant_mapping.empty:
                    # Ensure consistent dtype between mapping keys and raw_df[col]
                    target_dtype = raw_df[col].dtype
                    relevant_mapping["answer_code"] = relevant_mapping["answer_code"].astype(target_dtype)

                    # Build the mapping dictionary
                    mapping_dict = (
                        relevant_mapping.drop_duplicates("answer_code")
                        .set_index("answer_code")["answer_string"]
                        .to_dict()
                    )


                    # Map values using the dictionary (preserving unmapped values)
                    raw_df[col] = raw_df[col].map(mapping_dict).fillna(raw_df[col])



        return raw_df
