import pandas as pd

from .data_processing import *
from .type_subtype_categorizer import *
from .question_embedder import *

def run_question_data_pipeline(data_map: pd.DataFrame, raw_data: pd.DataFrame, api_key: str, source : str):
    """
    Orchestrates the full question data pipeline:
    1. Processes the question guide from the data map.
    2. Maps the raw data to the question guide.
    3. Generates type and subtype columns.
    4. Converts the enriched guide into a question dictionary with embeddings.
    5. Splits the question dictionary into metadata and embedding tables.

    Args:
        data_map (pd.DataFrame): DataFrame used to build the question guide.
        raw_data (pd.DataFrame): Raw responses or data to be mapped to questions.
        api_key (str): API key for OpenAI services used in embedding.

    Returns:
        dict: A dictionary with all major data artifacts:
              - 'question_guide': enriched question guide DataFrame
              - 'mapped_data': raw data mapped with answer keys
              - 'type_subtype_df': type and subtype annotations
              - 'question_dict': final list of embedded question dictionaries
              - 'metadata_df': DataFrame excluding chunk_embedding
              - 'embedding_df': DataFrame with chunk_embedding only and primary key
    """
    # Step 1: Process question guide
    question_guide = process_question_guide(data_map)

    # Step 2: Map raw data to the question guide
    mapped_data = mapping_answer_key_to_raw_data(raw_data, question_guide)

    # Step 3: Generate type/subtype info within the guide
    type_subtype_df = type_and_sub_type_generator(question_guide, mapped_data)

    # Step 4: Generate final question dictionary with embeddings
    question_dict = process_question_list_pipeline(question_guide, api_key, source = source)

    # Step 5: Split question_dict into metadata and embeddings
    metadata_df, embedding_df = split_question_dict_to_tables(question_dict)

    return {
        'question_guide': question_guide,
        'mapped_data': mapped_data,
        'type_subtype': type_subtype_df,
        'question_dict': question_dict,
        'embeddings_metadata_df': metadata_df,
        'embedding_df': embedding_df
    }
