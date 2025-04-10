import pandas as pd
import re


def type_and_sub_type_generator(question_guide,raw_data):
    ##step1 : Validate the raw data
    validating_data = validate_dataframe(raw_data)

    #step2 : Cleaning question guide data map
    question_guide = question_guide_data_processing(question_guide)

    #step3: Processing survey Raw data
    question_type = []
    question_sub_type = []

    #step4: Mapping question type to subtype i.e identifying each question
    for question_no in question_guide['Question_no.']:
        data_type = survey_question_codeentifier_type(question_number=question_no,raw_data=raw_data)
        question_type.append(data_type[0])
        question_sub_type.append(data_type[1])
    question_guide["Type"] = question_type
    question_guide["Sub-type"] = question_sub_type

    return question_guide.drop_duplicates()

def survey_question_codeentifier_type(question_number,raw_data):
    """
    Gives the type of question.
    Returns: Types and subtype

    """
    n = 3 # Rank 
    list_of_questions = identify_list_of_questions_from_question_number(question_number,raw_data)
    list_of_questions = removing_extra_remove(list_of_questions)
    list_of_questions = removing_extra_extra_input(list_of_questions)
    unique_values_with_negative_one = identify_unique_values(list_of_questions,raw_data)
    unique_values_without_negative_one = remove_number(unique_values_with_negative_one,-1)
    unique_values_without_negative_one = remove_number(unique_values_with_negative_one,-99)

    # if ("." not in question_number):
    #     return("Multiple choice","single-select")
    if len(list_of_questions) == 1:
        return("Multiple choice","single-select")
    elif len(unique_values_without_negative_one) == 2:
        return("Multiple choice","multi-select")
    elif -99 in unique_values_with_negative_one and len(unique_values_without_negative_one) == n:
        return("Rank","Rank")
    elif len(unique_values_without_negative_one) > 3:
        return("Matrix","single-select")
    else:
        return("Others","Others")
    
def identify_list_of_questions_from_question_number(question_number, df):
    """
    Returns a list of column names that start with the given question number, ensuring only exact matches using regex.
    
    :param question_number: A string like "Q1", "Q2", etc.
    :param df: A pandas DataFrame.
    :return: A list of matching column names.
    """
    # pattern = re.compile(rf'^{re.escape(question_number)}([ ._]|$)')
    pattern = re.compile(rf'^{re.escape(question_number)}([ ._:]|$)')
    list_of_columns = [col for col in df.columns if pattern.match(col)]

    return list_of_columns

def identify_unique_values(list_of_questions,raw_data):
    unique_values = []
    for question in list_of_questions:
        unique_values = unique_values + list(set(raw_data[question].dropna()))
    return(list(set(unique_values)))

def remove_number(lst,n):
    """
    Returns a new list with all instances of -1 removed.
    """
    return [x for x in lst if x != n]

def removing_extra_remove(lst):
    return [x for x in lst if "removed" not in x]

def removing_extra_extra_input(lst):
    return [x for x in lst if (("_user_input" not in x) and (" user input" not in x))]

def validate_dataframe(df):
    """
    Validates a DataFrame for:
    - Blank rows at the top
    - Presence of a column containing "_user_input" or " user input"
    
    Parameters:
    df (pd.DataFrame): The DataFrame to validate
    
    Returns:
    dict: A dictionary with validation results
    """
    validation_results = {
        "blank_rows_at_top": 0,
        "user_input_column_present": False
    }
    
    # Check for blank rows at the top
    blank_rows_count = 0
    for index, row in df.iterrows():
        if row.isnull().all():  # If all values in the row are NaN
            blank_rows_count += 1
        else:
            break  # Stop counting when a non-blank row is found
    
    validation_results["blank_rows_at_top"] = blank_rows_count

    # Check for columns containing "_user_input" or " user input"
    for col in df.columns:
        if "_user_input" in col or " user input" in col:
            validation_results["user_input_column_present"] = True
            break  # No need to check further if found

    return validation_results

def question_guide_data_processing(question_guide):
    """
    Processes a question guide DataFrame by extracting relevant columns, renaming them,
    filtering based on question number format, and removing missing values.

    Parameters:
    question_guide (pd.DataFrame): A pandas DataFrame containing the question guide data.

    Returns:
    pd.DataFrame: A processed DataFrame with only valid question numbers and corresponding questions.
    """
    # Select only the first two columns (assuming they contain question numbers and questions)
    question_guide = question_guide.iloc[:,0:2]

    # Rename columns for better readability
    question_guide.columns =  ['Question_no.','Question']

    # Filter rows where 'Question_no.' starts with "Q" (to keep only valid question entries)
    question_guide = question_guide[question_guide['Question_no.'].str.startswith("Q") == True]

    # Remove any rows containing NaN values
    question_guide = question_guide.dropna()

    return question_guide  # Return the cleaned and processed DataFrame

def replace_text_with_negative_one(df: pd.DataFrame):
    """
    Returns a copy of the DataFrame where all string (text) values are replaced with -1.
    Uses DataFrame.map instead of applymap to avoid deprecation warning.
    """
    df.map(lambda x: -1 if isinstance(x, str) else x)

    return df

def fill_nan_with_value(df, value=-99):
    """
    Replaces all NaN values in a DataFrame with a specified value.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    value (any, optional): Value to replace NaNs with. Default is -99.

    Returns:
    pd.DataFrame: DataFrame with NaNs replaced.
    """
    df.fillna(value)

    return df

def raw_data_processing_for_type_sub_type(df):
    df = replace_text_with_negative_one(df)
    df = fill_nan_with_value(df)

    return (df)


