from io import StringIO
import pandas as pd

def markdown_table_to_df(md_str):
    """
    Convert a Markdown table (GitHub-style) into a pandas DataFrame.
    
    Args:
        md_str (str): The Markdown table as a string.
    
    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    # Split the input into lines, stripping away extra surrounding whitespace and '|'.
    lines = md_str.strip().split('\n')
    lines = [line.strip().strip('|') for line in lines if line.strip()]

    # Build a list of lines excluding the 'separator' line (---|--- etc.)
    clean_lines = []
    for line in lines:
        # Remove all spaces for the check
        check = line.replace(' ', '')
        # If all characters are '-', ':', or '|', consider this a separator line and skip it
        if all(ch in '-:|' for ch in check):
            continue
        clean_lines.append(line)

    # Join the cleaned lines into a single string
    text_for_csv = '\n'.join(clean_lines)

    # Parse with pandas.read_csv, using '|' as the delimiter
    df = pd.read_csv(StringIO(text_for_csv), sep='|', engine='python')

    # Clean up column names and strip any extra whitespace in data cells
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    df = df.dropna()
    return df

def output_formating_custom_agent(response,answer_break = "Insights:"):
    final_response = response.split(answer_break)
    output_dict = {"Data":markdown_table_to_df(final_response[0]),
                "Insights":final_response[1]}
    return output_dict

def custom_agent_layer(query, custom_agent):
    response = custom_agent.run(query)
    
    try:
        final_response = response.split("Insights:")
        output_dict = {
            "Data": markdown_table_to_df(final_response[0]),
            "Insights": final_response[1].strip()
        }
    except Exception as e:
        output_dict = {
            "Data": markdown_table_to_df(response),  # still try parsing the table
            "Insights": f" Unable to extract 'Insights:' section. Error: {e}\n\nFull response:\n{response}"
        }
    
    return output_dict