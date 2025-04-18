import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings



def question_df_to_dict(question_guide: pd.DataFrame, source: str = None):
    """
    Converts a DataFrame of questions and answers into a list of dictionaries with structured metadata.

    Args:
        question_guide (pd.DataFrame): A DataFrame with columns:
            - 'Question_code': Unique code for each question.
            - 'Question_string': The question text.
            - 'answer_string': Text for possible answers.
        source (str, optional): Optional string to tag the source of the data.

    Returns:
        list[dict]: A list of dictionaries with structured question data.
    """
    # Validate input
    required_columns = {'Question_code', 'Question_string', 'answer_string'}
    if not isinstance(question_guide, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not required_columns.issubset(question_guide.columns):
        raise ValueError(f"DataFrame must include columns: {required_columns}")

    # Group answers by question and combine answer strings
    grouped = question_guide.groupby(['Question_code', 'Question_string'])['answer_string'] \
                            .agg('|| '.join).reset_index()

    # Build the structured list without modifying the DataFrame
    question_list = []
    for _, row in grouped.iterrows():
        question_dict = {
            "question_code": row['Question_code'],
            "question_text": row['Question_string'],
            "answer_options": row['answer_string'],
            "question_with_answers": f"{row['Question_string']} {row['answer_string']}",
            "source": source
        }
        question_list.append(question_dict)

    return question_list


def semantic_chunking_wrapper(text, embedding_type="OpenAI", api_key=None):
    """
    Splits the given text into semantically meaningful chunks using the SemanticChunker.

    Args:
        text (str): The input text to be chunked.
        embedding_type (str): The type of embedding to use ("openai" or "huggingface").
        api_key (str, optional): API key for OpenAI embeddings.

    Returns:
        list: A list of text chunks obtained from semantic chunking.
    """

    if embedding_type == "OpenAI":
        if not api_key:
            raise ValueError("API key is required for OpenAI embeddings.")
        embedding_model = OpenAIEmbeddings(api_key=api_key)
    elif embedding_type == "HuggingFace":
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Invalid embedding_type. Choose 'openai' or 'huggingface'.")

    text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="standard_deviation")
    chunks = text_splitter.split_text(text)
    
    return chunks


def chunk_questions(question_list: list, api_key: str, embedding_type = "OpenAI"):
    """
    Applies semantic chunking to the 'question_with_answers' field for each question dictionary.

    Args:
        question_list (list): A list of dictionaries, each containing question metadata including 
                              'question_with_answers'.

        api_key (str): API key used by the semantic chunking function.

    Returns:
        list: A new list of dictionaries where each entry contains a chunk of the original question data.
    """
    chunked_questions = []  # Initialize list for output dictionaries

    for question in question_list:
        # Generate semantic chunks from the full question + answers string
        chunks = semantic_chunking_wrapper(question['question_with_answers'], embedding_type,api_key)

        # Create a new dictionary for each chunk
        for chunk in chunks:
            chunked_questions.append({
                'question_code': question['question_code'],
                'question_text': question['question_text'],
                'question_with_answers': question['question_with_answers'],
                'answer_options': question['answer_options'],
                'source': question['source'],
                'chunk_text': chunk
            })

    return chunked_questions


def encoding(text, api_key=None, type_of_encoding="OpenAI"):
    """
    Encodes the given text into an embedding vector using the specified encoding method.

    Args:
        text (str): The input text to be encoded.
        api_key (str, optional): API key for authentication with the embedding model (required for OpenAI).
        type_of_encoding (str, optional): Specifies the encoding type. "OpenAI" or "HuggingFace". Defaults to "OpenAI".

    Returns:
        list or None: A list representing the embedding vector if encoding is successful,
                      otherwise None if the encoding type is unsupported.
    """
    if type_of_encoding == "OpenAI":
        from langchain.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
        return embeddings.embed_query(text)

    elif type_of_encoding == "HuggingFace":
        model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight and free
        return model.encode(text).tolist()  # Convert to list to match return type

    return None  # Unsupported encoding type



def add_embeddings_to_chunks(chunked_questions: list, api_key: str, encoding_type: str = "OpenAI"):
    """
    Adds embedding vectors to each chunked question dictionary.

    Args:
        chunked_questions (list): A list of dictionaries. Each dictionary contains metadata and a 'chunk_text' field.
        api_key (str): API key used to access the embedding service.
        encoding_type (str): The type of embedding model to use (default: "OpenAI").

    Returns:
        list: A list of dictionaries, each with an added 'chunk_embedding' field.
    """
    updated_chunks = []

    for chunk in chunked_questions:
        # Generate an embedding for the current chunk of text
        chunk['chunk_embedding'] = encoding(chunk['chunk_text'], api_key, encoding_type)

        # Append the updated chunk dictionary to the result list
        updated_chunks.append(chunk)

    return updated_chunks


def process_question_list_pipeline(question_df: pd.DataFrame, api_key: str, encoding_type: str = "OpenAI", source: str = None):
    """
    Orchestrates the full pipeline to process questions:
    1. Convert DataFrame to structured question dictionaries.
    2. Apply semantic chunking to questions and answers.
    3. Add embeddings to each chunked text.

    Args:
        question_df (pd.DataFrame): DataFrame containing questions and answers.
        api_key (str): API key for OpenAI services.
        encoding_type (str): The encoding model to use. Default is "OpenAI".
        source (str, optional): Optional tag to include as the data source.

    Returns:
        list: Final list of dictionaries with questions, chunks, and embeddings.
    """
    # Step 1: Convert the DataFrame into structured question dictionaries
    question_list = question_df_to_dict(question_df, source=source)

    # Step 2: Chunk the combined question + answer text using semantic chunking
    chunked_question_list = chunk_questions(question_list, api_key, embedding_type = encoding_type)

    # Step 3: Add embeddings to each chunk
    embedded_chunked_questions = add_embeddings_to_chunks(chunked_question_list, api_key, encoding_type)

    return embedded_chunked_questions


def split_question_dict_to_tables(question_dict: list):
    """
    Splits a list of question dictionaries into metadata and embedding DataFrames,
    using a generated primary key.

    Args:
        question_dict (list): List of dictionaries, each representing a question.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: metadata_df, embedding_df
    """
    import pandas as pd

    metadata_records = []
    embedding_records = []

    for idx, item in enumerate(question_dict):
        primary_key = f"{item['source']}_{idx}"

        # Metadata (excluding embedding)
        metadata = {k: v for k, v in item.items() if k != 'chunk_embedding'}
        metadata['primary_key'] = primary_key
        metadata_records.append(metadata)

        # Embedding (only embedding and primary key)
        embedding_records.append({
            'primary_key': primary_key,
            'chunk_embedding': item['chunk_embedding']
        })

    metadata_df = pd.DataFrame(metadata_records)
    embedding_df = pd.DataFrame(embedding_records)

    return metadata_df, embedding_df