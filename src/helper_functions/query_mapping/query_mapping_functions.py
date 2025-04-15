import pandas as pd
import numpy as np
import ast
from typing import List, Dict

from langchain.embeddings import OpenAIEmbeddings  # adjust import if needed
from langchain.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

"""
==============================================================================
Question Retrieval & Ranking System Using OpenAI Embeddings and LLMs
==============================================================================

This script allows ranking and classification of questions based on a user query
by using a two-step approach:
1. Semantic similarity using OpenAI embeddings + cosine similarity.
2. Final reranking using a Language Model (LLM) prompt.

==============================================================================
SECTION: Imports
------------------------------------------------------------------------------
Essential libraries include:
- pandas, numpy: for data manipulation.
- ast: for safe evaluation of LLM responses.
- langchain: for using OpenAI LLM and embedding models.
==============================================================================
"""

"""
==============================================================================
SECTION: Utilities
==============================================================================

1. initializing_open_ai_model(api_key, model_name, temperature)
   - Initializes and returns an OpenAI chat model using LangChain's ChatOpenAI.
   - Parameters: API key, model name (default "gpt-4o"), temperature.
   - Returns: A ChatOpenAI model instance.

2. encoding(text, api_key, type_of_encoding="OpenAI")
   - Generates embedding vector for the given text using OpenAI embeddings.
   - Parameters: input text, API key.
   - Returns: A list of floats (embedding vector).

3. cosine_similarity(vec1, vec2)
   - Computes cosine similarity between two embedding vectors.
   - Returns: A float similarity score.

4. promt_to_sort_llm(prompt_dict)
   - Creates a structured LangChain prompt template from system/user messages.
   - Returns: A ChatPromptTemplate instance.
==============================================================================
"""

"""
==============================================================================
SECTION: Rankers
==============================================================================

1. top_ranker_by_cosine_similarity(api_key, question_list, query, top_n)
   - Computes query embeddings and calculates cosine similarity between the query and pre-embedded questions.
   - Returns: Top N questions with the highest similarity scores.

2. rank_questions_by_llm(model, prompt, question_list, query, top_n)
   - Uses an LLM to rerank the most similar questions based on query context.
   - Accepts a formatted prompt and question list.
   - Returns: Top N LLM-ranked questions.
==============================================================================
"""

"""
==============================================================================
SECTION: Process Wrapper
==============================================================================

get_best_question_info(query, api_key, question_dict_with_embedding, 
                       prompt_for_llm_sorting, model, type_sub_type_df,
                       top_n=5, top_q=3)

- Orchestrates the entire pipeline:
    1. Retrieves top-N semantically similar questions via cosine similarity.
    2. Reranks those using an LLM with prompt guidance.
    3. Retrieves type and subtype info from a mapping DataFrame.

- Returns:
    {
        "question_id": str,
        "question_text": str,
        "type": str,
        "subtype": str,
        "full_row": pd.Series
    }

==============================================================================
"""

#------------------------------- Utilities -------------------------------#
def initializing_open_ai_model( api_key: str, model_name: str = "gpt-4o", temperature: float = 1.0):
    """
    Initialize and return an OpenAI chat model.

    Args:
		api_key (str): OpenAI API key.
        model_name (str): Name of the model (e.g., 'gpt-4'. This is also the default).
        temperature (float): Sampling temperature (default is 1.0).

    Returns:
        ChatOpenAI: An instance of the chat model.
    """
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature
        )
        return model
    except Exception as e:
        raise ValueError(f"Failed to initialize model: {e}")

def encoding(text: str, api_key: str, type_of_encoding: str = "OpenAI") -> list | None:
    """
    Encodes the given text into an embedding vector using the specified encoding method.

    Args:
        text (str): The input text to be encoded.
        api_key (str): API key for authentication with the embedding model.
        type_of_encoding (str, optional): Specifies the encoding type. Defaults to "OpenAI".

    Returns:
        list or None: A list representing the embedding vector if encoding is successful,
                      otherwise None if the encoding type is unsupported or fails.
    """
    if type_of_encoding == "OpenAI":
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=api_key
                # dimensions=1024  # Optionally uncomment if you want custom dimensions
            )
            return embeddings.embed_query(text)
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {e}")

    return None

def cosine_similarity(vec1: list, vec2: list) -> float:
        """
        Computes the cosine similarity between two vectors.
 
        Args:
            vec1 (list): The first vector.
            vec2 (list): The second vector.
 
        Returns:
            float: The cosine similarity between the two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  # For computing cosine similarity

def promt_to_sort_llm(prompt_dict):
    prompt = ChatPromptTemplate.from_messages([
         ('system', prompt_dict['system_prompt']),
         ('user', prompt_dict['user_prompt'])
    ])
    return prompt

#------------------------------- Rankers -------------------------------#
def top_ranker_by_cosine_similarity(api_key,
        question_list: list[dict], query: str, top_n: int = 5
    , ) -> list[dict]:
        """
        Ranks the questions based on their cosine similarity to the query.

        Args:
            question_list (List[Dict]): A list of dictionaries, each containing question metadata and chunk embeddings.
            query (str): The query string to compare against the question chunks.
            top_n (int, optional): The number of top results to return. Defaults to 5.

        Returns:
            List[Dict]: A list of the top N dictionaries from the question list, 
                        sorted by cosine similarity score in descending order.
        """
        # Generate embeddings for the query text
        text_embeddings = encoding(query, api_key)
        # Compute cosine similarity scores for each question
        for question in question_list:
            question["cosine_similarity_score"] = cosine_similarity(
                text_embeddings, question["chunk_embedding"]
            )

        # Sort the question list by cosine similarity score in descending order
        sorted_questions = sorted(
            question_list, key=lambda x: x["cosine_similarity_score"], reverse=True
        )

        return sorted_questions[:top_n]

def rank_questions_by_llm(model, prompt, question_list: List[Dict], query: str, top_n: int = 5) -> List[Dict]:
    """
    Sorts and ranks questions using an LLM, returning the top N most relevant ones.
    """
    if not question_list:
        return []

    formatted_questions = "\n".join([f"{i+1}. {q['question_text']}" for i, q in enumerate(question_list)])
    chain = prompt | model

    try:
        response = chain.invoke({'query': query, 'formatted_questions': formatted_questions}).content
        response = ast.literal_eval(response.strip())
        ranked_indices = [int(num) - 1 for num in response]
        ranked_questions = [question_list[i] for i in ranked_indices if 0 <= i < len(question_list)][:top_n]
        
        print([i['question_code'] for i in ranked_questions])
        return ranked_questions

    except Exception as e:
        raise ValueError(f"[ERROR] Failed to rank questions: {e}")
    
#------------------------------- Process Wrappers -------------------------------#
def get_best_question_info(query, api_key ,question_dict_with_embedding, prompt_for_llm_sorting, model, type_sub_type_df, top_n=5, top_q=3):
    # Step 1: Find top-N similar by embedding
    similarity_score = top_ranker_by_cosine_similarity(
        api_key, question_dict_with_embedding, query=query, top_n=top_n
    )

    # Step 2: Rerank with LLM
    ranked_questions = rank_questions_by_llm(
        model, prompt_for_llm_sorting, question_list=similarity_score, query=query, top_n=top_q
    )

    best_question = ranked_questions[0]
    question_id = best_question['question_code']
    question_text = best_question['question_text']

    # Step 3: Get type/subtype
    row = type_sub_type_df[type_sub_type_df["Question_no."] == question_id]
    if row.empty:
        raise ValueError(f"Type/Subtype not found for question {question_id}")

    q_type = row["Type"].values[0]
    q_subtype = row["Sub-type"].values[0]

    return {
        "question_id": question_id,
        "question_text": question_text,
        "type": q_type,
        "subtype": q_subtype,
        "full_row": row
    }