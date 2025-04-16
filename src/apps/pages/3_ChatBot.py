# chat_page.py

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


from src.helper_functions.CL_Cuts.cuts import *
from src.helper_functions.query_mapping.query_mapping_functions import *
from src.helper_functions.tools.tools import *
from src.helper_functions.tools.output_formatting import *
from src.helper_functions.data_pre_processing.data_filtering.filter_dataframe import *
from config.path_config import DATA_DIR, PROJECT_DIR, DEPENDENCIES_DIR, CONFIG_DIR
from src.helper_functions.query_typo_check.check_typo import *

if "api_key" in st.session_state:
    api_key = st.session_state.api_key
else:
    st.warning("No api key found.")
    st.stop()

if "question_guide" in st.session_state:
    question_guide = st.session_state.question_guide

if "embedding_metadata" in st.session_state:
    embedding_metadata = st.session_state.embedding_metadata

if "question_dict" in st.session_state:
    question_dict = st.session_state.question_dict

if "type_subtype" in st.session_state:
    type_subtype = st.session_state.type_subtype

if "mapped_data" in st.session_state:
    mapped_data = st.session_state.mapped_data

if "embedding_df" in st.session_state:
    embedding_df = st.session_state.embedding_df



# Setup LLM
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)

## to get propmt_for_llm_sorting
llm_ranking_prompt_path = CONFIG_DIR+"\\llm_ranking_prompt.yaml"

prompt_dict = read_yaml(llm_ranking_prompt_path)

prompt_for_llm_sorting = promt_to_sort_llm(prompt_dict)


## to get final_processed_raw_df_filter
default_filters_path = CONFIG_DIR+'\\default_filters\\wave_10_default_filters.yaml'
default_filters = read_yaml(default_filters_path)
final_processed_raw_df_filter = add_filtering_columns(mapped_data,default_filters)

## to get Filters
Filters = generate_filter_value_dict(final_processed_raw_df_filter,default_filters)

# tool functions
get_best_question_info_tool = make_get_question_info_tool(question_dict,prompt_for_llm_sorting,model = llm, type_subtype_df=type_subtype,api_key = api_key)
pivot_count_tool = make_pivot_count_tool(final_processed_raw_df_filter,Filters)
multi_select_pivot_tool = make_multi_select_pivot_tool(final_processed_raw_df_filter,Filters)

# create tools
tools = [
    Tool(
        name="GetBestQuestionInfo",
        func=get_best_question_info_tool,
        description="Given a user query, returns the best matching survey question and its type/subtype."
    ),
    Tool(
        name="MultiSelectPivot",
        func=multi_select_pivot_tool,
        description=(
    "Generate pivot table for a multi-select question. "
    "Input can be a Python dictionary with the following keys:\n"
    "- 'query': the full user question\n"
    "- 'question_id': the question code \n"
)
    ),
    Tool(
        name="SingleSelectPivot",
        func=pivot_count_tool,
        description=(
    "Generate pivot table for a single-select question. "
    "Input can be a Python dictionary with the following keys:\n"
    "- 'query': the full user question\n"
    "- 'question_id': the question code \n"
)
    )
]


# initialize custom agent
agents_prompt_file_name = CONFIG_DIR+"\\agent_prompt.yaml"
agents_prompt = read_yaml(agents_prompt_file_name)

custom_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs = {
    "prefix": agents_prompt['custom_agent_prefix']
}
)



# Streamlit UI
st.set_page_config(page_title="Survey Chatbot")
st.title("🤖 Survey Chatbot")


# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append({
        "role":"assistant",
        "content":"Hello!\nI'm your survey analysis assistant.I'll help you uncover trends, insights and more."
    })


user_input = st.chat_input("Ask me about your data...")


# Inject CSS styles (once)
# Inject custom CSS for layout
st.markdown("""
<style>
    .user-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .bot-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 10px;
    }
    .user-text, .bot-text {
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #ccc;
        max-width: 60%;
        white-space: pre-wrap;
    }
    .user-text {
        background-color: #DCF8C6;
        margin-left: 10px;
    }
    .bot-text {
        background-color: #ECECEC;
        margin-right: 10px;
    }
    .user-avatar, .bot-avatar {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Render chat history with conditional layout
for msg in st.session_state.chat_history:
    is_user = msg["role"] == "user"
    container_class = "user-container" if is_user else "bot-container"
    text_class = "user-text" if is_user else "bot-text"
    avatar_class = "user-avatar" if is_user else "bot-avatar"
    avatar = "🧑" if is_user else "🤖"

    # Build the HTML based on sender
    if is_user:
        html = f"""
        <div class="{container_class}">
            <div class="{text_class}">{msg["content"]}</div>
            <div class="{avatar_class}">{avatar}</div>
        </div>
        """
    else:
        html = f"""
        <div class="{container_class}">
            <div class="{avatar_class}">{avatar}</div>
            <div class="{text_class}">{msg["content"]}</div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)

    # Optional: display a dataframe if requested
    if msg["role"] == "assistant" and msg.get("show_df") and "cut_df" in msg:
        with st.expander("View Cut"):
            st.dataframe(msg["cut_df"])  # replace with your actual dataframe



# Handle new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    user_input = llm_layer_to_check_typos(user_input, agents_prompt['typo_layer'], Filters['Country Filter'], Filters, llm )
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            agent_output = custom_agent_layer(user_input, custom_agent)
            insights = agent_output["Insights"]
            cut_df = agent_output["Data"]

            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": insights,
                "show_df": True,
                "cut_df": cut_df
            })

            st.rerun()
