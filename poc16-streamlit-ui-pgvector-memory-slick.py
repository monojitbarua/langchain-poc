import torch
torch.classes.__path__ = []

from dotenv import load_dotenv
import streamlit as st
from typing import Set
#from poc14_doc_assistant_pgvector_memory import retrieval_llm
from poc17_doc_assistant_pgvector_memory_ollama import retrieval_llm

load_dotenv()
st.set_page_config(page_title="Langchain POC", layout="wide")

# --- Dark Mode Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #121212 !important;
            color: white !important;
        }
        .stTextInput input {
            background-color: #2c2c2c !important;
            color: white !important;
        }
        .stChatMessage, .stMarkdown {
            background-color: #1f1f1f !important;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .stSidebar, .sidebar .sidebar-content {
            background-color: #1f1f1f !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #333 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar content ---
with st.sidebar:
    st.image("data/image.png", caption="Product Diagram", use_container_width=True)
    st.markdown("""
        ## Product Info
        - **Name**: Langchain POC  
        - **Version**: 1.0  

        More info or diagrams can go here.
    """)

# --- Header ---
st.markdown("<h1 style='color: white;'>Langchain POC</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# --- Session state setup ---
if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# --- Prompt input ---
prompt = st.text_input("Prompt", placeholder="Enter your question...")

# --- Source builder ---
def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string

# --- Query handler ---
if prompt:
    with st.spinner("Generating response ..."):
        generated_response = retrieval_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        sources = set([doc.metadata["source"] for doc in generated_response["context"]])
        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))

# --- Chat UI with latest messages at bottom ---
if st.session_state["chat_answers_history"]:
    chat_container = st.container()
    with chat_container:
        # Reverse to show latest messages at bottom (like ChatGPT)
        for generated_response, user_query in zip(
            reversed(st.session_state["chat_answers_history"]),
            reversed(st.session_state["user_prompt_history"]),
        ):
            st.chat_message("user").write(user_query)
            st.chat_message("assistant").write(generated_response)
