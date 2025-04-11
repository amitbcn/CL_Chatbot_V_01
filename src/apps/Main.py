import streamlit as st

st.set_page_config(page_title="Data Analyst Chatbot")
st.title("Welcome to the Data Analyst Assistant")

st.markdown("""
This app allows you to:
- 📤 Upload a dataset (on the **Upload Data** page)
- 💬 Chat with an AI analyst about your data (on the **Survey Chatbot** page)

Use the sidebar to navigate between pages.
""")

# API key input form
st.markdown("### 🔐 Enter your OpenAI API Key")
with st.form("api_key_form"):
    api_key_input = st.text_input(
        "API Key (sk-...)", 
        type="password", 
        placeholder="Paste your OpenAI API key here..."
    )
    submit_button = st.form_submit_button("Submit")

if submit_button and api_key_input:
    st.session_state.api_key = api_key_input
    st.success("✅ API key saved securely in session.")
