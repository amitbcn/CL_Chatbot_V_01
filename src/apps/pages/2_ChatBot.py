# chat_page.py

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

if "api_key" in st.session_state:
    api_key = st.session_state.api_key
else:
    st.warning("No api key found.")
    st.stop()

# Setup LLM
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)

# Check for uploaded data
if "raw_data" in st.session_state:
    raw_data = st.session_state.raw_data
else:
    st.warning("No dataset found.")
    st.stop()

if "data_map" in st.session_state:
    data_map = st.session_state.data_map
else:
    st.warning("No dataset found.")
    st.stop()


# Prompt setup
system_prompt = """
You are a highly skilled data analyst. Analyze the dataset provided and generate meaningful insights.
Focus on things like trends, distributions, anomalies, missing values, column types, and potential business takeaways.

Use clear, simple explanations and format key points as bullet points if needed.
Do not hallucinate or assume column meanings—base your analysis on visible data only.
"""


# Create LangChain prompt chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

# Streamlit UI
st.set_page_config(page_title="Survey Chatbot")
st.title("🤖 Survey Chatbot")


# Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me about your data...")

# Display conversation history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "show_df" in msg and msg["show_df"]:
            with st.expander("View Cut"):
                st.dataframe(raw_data.head())

# Handle new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Optionally inject data description into the prompt
            context = f"\nHere is the dataset:\n{raw_data.head(3).to_markdown()}" if raw_data is not None else ""
            response = chain.invoke({"question": user_input + context})
            st.markdown(response)

            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "show_df" : True})

            st.rerun()