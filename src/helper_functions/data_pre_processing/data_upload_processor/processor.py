import pandas as pd
import os
import sys

from .data_processing import *
from .type_subtype_categorizer import *
from .question_embedder import *

def run_question_data_pipeline(data_map: pd.DataFrame, raw_data: pd.DataFrame, api_key: str, source: str, encoding_type: str = "OpenAI"):
    """
    Executes the end-to-end pipeline for transforming raw survey or questionnaire data into a structured, 
    enriched format with embeddings for further processing or storage.

    The pipeline consists of the following steps:
    1. Parses the question guide from a mapping DataFrame.
    2. Aligns and maps raw response data to the questions defined in the guide.
    3. Annotates each question with type and subtype metadata.
    4. Generates a dictionary of processed questions, including vector embeddings.
    5. Splits the question dictionary into two structured tables:
       - One for metadata (question text, types, etc.)
       - One for embedding vectors (for ML or search indexing)

    Args:
        data_map (pd.DataFrame): A DataFrame containing question metadata used to build the guide.
        raw_data (pd.DataFrame): The raw dataset containing responses or source data to be aligned with the guide.
        api_key (str): API key used for generating embeddings (typically with OpenAI).
        source (str): Identifier for the data source, used in embedding generation and tagging.
        encoding_type (str, optional): Encoding model or method to use for embeddings (default is "OpenAI").

    Returns:
        dict: A dictionary containing the following processed outputs:
            - 'question_guide' (pd.DataFrame): Parsed and structured question guide.
            - 'mapped_data' (pd.DataFrame): Raw data aligned with the question structure.
            - 'type_subtype' (pd.DataFrame): DataFrame with type and subtype annotations for each question.
            - 'question_dict' (list[dict]): List of enriched question entries, each with text, metadata, and embeddings.
            - 'embeddings_metadata_df' (pd.DataFrame): Question metadata excluding embeddings.
            - 'embedding_df' (pd.DataFrame): Table of question embeddings with a reference key.
    """

    # Step 1: Process question guide
    question_guide = process_question_guide(data_map)

    # Step 2: Map raw data to the question guide
    mapped_data = mapping_answer_key_to_raw_data(raw_data, question_guide)

    # Step 3: Generate type/subtype info within the guide
    type_subtype_df = type_and_sub_type_generator(question_guide, mapped_data)

    # Step 4: Generate final question dictionary with embeddings
    question_dict = process_question_list_pipeline(question_guide, api_key,encoding_type = encoding_type ,source = source)

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
