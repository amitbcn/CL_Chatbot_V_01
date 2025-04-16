
import sys
import os

# Add project root to sys.path in Jupyter or interactive session
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from src.helper_functions.query_mapping.query_mapping_functions import *
from src.helper_functions.CL_Cuts.cuts import *
import json

####################################################################################################################

def extract_filters(query: str, filter_dict: dict) -> dict:
    """
    Returns a dictionary:
        {
           "Generation Filter": [matched_values...],
           "Political Consumption Filter": [matched_values...],
           ...
        }
    where each key corresponds to a filter category, and the value is
    a list of all recognized filter values found in `query`.
    """
    FILTERS = {}
    for key, value in filter_dict.items():
        # Split each string by commas and strip whitespace
        FILTERS[key] = [item.strip() for item in value.split(',')]
    filter_dict = FILTERS
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()

    # Prepare an output dictionary with empty lists
    detected_filters = {f: [] for f in filter_dict}

    for filter_name, filter_values in filter_dict.items():
        for val in filter_values:
            # Simple substring check
            # Lowercase the filter value for matching
            val_lower = val.lower()

            # If the filter value is multiple words or has punctuation,
            # you might do a more careful match. Here we keep it simple:
            if val_lower in query_lower:
                detected_filters[filter_name].append(val)

    # Remove filter categories that had no matches
    detected_filters = {k: v for k, v in detected_filters.items() if v}
    return detected_filters



def extract_demographics(query: str, filter_dict: dict) -> list:
    """
    Given a user query (string) and the FILTERS dictionary,
    return a list of all filter "demographics" (keys) 
    found in the query via case-insensitive substring match.
    """
    FILTERS = {}
    for key, value in filter_dict.items():
        # Split each string by commas and strip whitespace
        FILTERS[key] = [item.strip() for item in value.split(',')]
    filter_dict = FILTERS
    
    query_lower = query.lower()
    demographics_found = []
    
    for demographic_key in filter_dict.keys():
        demographic_key_modified = demographic_key.replace(" Filter","")
        if demographic_key_modified.lower() in query_lower:
            demographics_found.append(demographic_key)
    
    return demographics_found



###########################################################################################################################


def make_get_question_info_tool(question_dict,prompt_for_sorting,model,type_subtype_df,api_key):
    def _get_best_question_info_tool(query: str) -> dict:
        return get_best_question_info(
            query=query,
            question_dict_with_embedding=question_dict,
            prompt_for_llm_sorting=prompt_for_sorting,
            model=model,
            type_sub_type_df=type_subtype_df,
            api_key = api_key
        )
    return _get_best_question_info_tool



def make_multi_select_pivot_tool(df,filters):
    def _multi_select_pivot_tool(tool_input: str) -> str:
        print(tool_input)
        try:
            # Handles both actual dict input and stringified JSON input
            tool_args = ast.literal_eval(tool_input)#tool_input if isinstance(tool_input, dict) else json.loads(tool_input)

            query = tool_args["query"]
            question_id = tool_args["question_id"]

            demographics = extract_demographics(query,filters)
            if not isinstance(demographics, list):
                demographics = [demographics]

            print(demographics)

        except Exception as e:
            return f"Tool failed to parse input: {e}"

        try:
            local_df = df.copy()

            # Apply any filters detected from the user query
            extra_filters = extract_filters(query, filters)
            print(extra_filters)
            filtered_dict = {k: v for k, v in extra_filters.items() if k not in demographics}
            local_df = filter_dataframe(local_df,filtered_dict)

            # Proceed with pivot
            question_cols = [col for col in local_df.columns if col.startswith(question_id)]
            if not question_cols:
                raise ValueError("No columns found for the question ID")
            
            

            melted = local_df[question_cols + demographics].melt(
                id_vars=demographics,
                var_name="Question_Option",
                value_name="Selected"
            )
            filtered = melted[melted["Selected"] == 1]
            grouped = filtered.groupby(["Question_Option"] + demographics).size().reset_index(name="Count")
            pivot_table = grouped.pivot_table(index="Question_Option", columns=demographics, values="Count", fill_value=0)

            pivot_table.loc["Total"] = pivot_table.sum()
            return pivot_table.to_markdown()
        except Exception as e:
            return f"Pivot generation failed: {e}"
    return _multi_select_pivot_tool




def make_pivot_count_tool(df,filters):
    def _pivot_count_tool(tool_input: str) -> str:
        try:
            tool_args = tool_input if isinstance(tool_input, dict) else ast.literal_eval(tool_input)
            query = tool_args["query"]
            question_id = tool_args["question_id"]


            demographics = extract_demographics(query=query,filter_dict=filters)

            if not isinstance(demographics, list):
                demographics = [demographics]

                

        except Exception as e:
            return f"Tool failed to parse input: {e}"

        try:
            local_df = df.copy()

            # Extra filters from input (e.g., income=Low)
            extra_filters = extract_filters(query, filters)
            print(extra_filters)
            filtered_dict = {k: v for k, v in extra_filters.items() if k not in demographics}
            local_df = filter_dataframe(local_df,filtered_dict)

            # Validate that the column exists
            if question_id not in local_df.columns:
                return f"Column '{question_id}' not found."

            # Build pivot
            pivot = local_df.pivot_table(index=question_id, columns=demographics, aggfunc="size", fill_value=0)
            pivot.loc["Total"] = pivot.sum()
            return pivot.to_markdown()
        except Exception as e:
            return f"Pivot generation failed: {e}"
    return _pivot_count_tool