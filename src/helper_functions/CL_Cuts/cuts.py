import pandas as pd
import re
import numpy as np

# Create view for single select questions
def pivot_count(df, groupby_col, column_col, total_column_name = "Total"):
    """
    Creates a pivot table from a DataFrame where one column is used for grouping,
    another column's unique values become the columns, and the values are counts.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    groupby_col (str): The column name to group by (rows).
    column_col (str): The column name whose unique values will be used as columns.

    Returns:
    pd.DataFrame: A pivot table with counts and a total row at the bottom.
    """
    pivot_table = df.pivot_table(index=groupby_col, columns=column_col, aggfunc='size', fill_value=0)

    # Add total row at the bottom
    pivot_table.loc[total_column_name] = pivot_table.sum()

    return pivot_table

# Filter the dataframe based on filter from the view_creation yaml
def filter_dataframe(data_df, filter_dictionary):
    """
    Filters a pandas DataFrame based on the provided filter dictionary.

    Parameters:
    data_df (pd.DataFrame): The input DataFrame to be filtered.
    filter_dictionary (dict): A dictionary where keys are column names and values are the filtering criteria.
                              If a value is "All", that column is not filtered.

    Returns:
    pd.DataFrame: A filtered DataFrame based on the specified criteria.
    """
    # Iterate through the filter dictionary
    for key, value in filter_dictionary.items():
        # Apply filter only if the value is not "All"
        if "ALL" not in value:
                data_df = data_df[data_df[key].isin(value)]
    
    # Return the filtered DataFrame
    return data_df
# Create percentage view form single and multi select pivot

def pivot_percentage(pivot_table, total_column_name="Total"):
    """
    Converts a crosstab/pivot table of counts into percentages with '%' signs,
    adds a final row labeled 'Total' with 100% for each column.

    Parameters:
    pivot_table (pd.DataFrame): Crosstab or pivot table with a total row at the bottom.
    total_column_name (str): The name of the row that contains total values for percentage calculation.

    Returns:
    pd.DataFrame: Percentage table with '%' signs and a final row showing 100% in each column.
    """

    # Get the total counts for each column (ensure it's numeric)
    total_values = pivot_table.loc[total_column_name].astype(int).round(0)

    # Drop the total row before percentage calculation
    data_only = pivot_table.drop(index=total_column_name)

    # Calculate percentage values (safe rounding before string formatting)
    percentage_df = (np.ceil(data_only.div(total_values, axis=1) * 1000) / 10).astype(float)

    # Convert to string and append '%' sign
    percentage_df = percentage_df.astype(str) + '%'

    # Create a final "Total" row filled with '100%' strings
    total_row = pd.Series(['100%' for _ in percentage_df.columns],
                          index=percentage_df.columns,
                          name='Total')

    # Append the new total row
    percentage_df = pd.concat([percentage_df, total_row.to_frame().T])

    return percentage_df



def total_multi_select(data_df, survey_question, demographics, total_column_name="Total"):
    """
    Creates a pivot table from a multi-select survey question, aggregating counts 
    based on demographic groups and adding a total row.
    """
    pattern = re.compile(rf"^{re.escape(survey_question)}(?!\d)")

    
    # Select columns related to the survey question and demographics
    exclude_pattern = re.compile(r"user[\s_-]*input", re.IGNORECASE)

    relevant_columns = [
        col for col in data_df.columns 
        if pattern.match(col) and not exclude_pattern.search(col)
    ] + [demographics]

    columns_to_calculate_sum = [col for col in data_df.columns if pattern.match(col) and not exclude_pattern.search(col)]


    sum_vals = data_df[columns_to_calculate_sum].sum(axis=1)
    sum_vals = [1 if x > 0 else 0 for x in sum_vals]
    data_df = data_df[relevant_columns].copy()
    data_df.loc[:, "unique_total"] = sum_vals

    output_df = data_df.melt(id_vars=[demographics, "unique_total"], var_name=survey_question, value_name='Count')
    output_df = output_df.groupby([survey_question, demographics])['unique_total'].sum().reset_index()
    
    output_df = output_df[[demographics, 'unique_total']].drop_duplicates().T
    output_df.columns = output_df.iloc[0]
    output_df = output_df[1:].reset_index(drop=True)
    output_df.index = [total_column_name]
    
    return output_df


def multi_select_pivot(data_df, survey_question, demographics, total_column_name="Total"):
    """
    Creates a pivot table from a multi-select survey question, aggregating counts 
    based on demographic groups and adding a total row.
    """
    pattern = re.compile(rf"^{re.escape(survey_question)}(?!\d)")


    exclude_pattern = re.compile(r"user[\s_-]*input", re.IGNORECASE)

    relevant_columns = [
        col for col in data_df.columns 
        if pattern.match(col) and not exclude_pattern.search(col)
    ] + [demographics]

    output_df = data_df[relevant_columns].melt(id_vars=[demographics], var_name=survey_question, value_name='Count')
    

    output_df = output_df.groupby([survey_question, demographics])['Count'].sum().reset_index()
    output_df = output_df.pivot(index=survey_question, columns=demographics, values='Count').fillna(0)

    output_df = pd.concat([output_df, total_multi_select(data_df, survey_question, demographics)])
    
    return output_df
