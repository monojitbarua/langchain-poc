import torch

torch.classes.__path__ = []

from dotenv import load_dotenv
import streamlit as st
from poc10_doc_assistant import retrieval_llm
from typing import Set

# RUN THIS FILE WITH  "streamlit run 11-streamlit-ui.py"


st.header("Langchain POC")
prompt = st.text_input("Prompt", placeholder="Enter your question...")


if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"- {source}\n"
    return sources_string


if prompt:
    with st.spinner("Generating response ..."):
        generated_response = retrieval_llm(query=prompt)
        print()
        print()
        print(generated_response)
        print()
        print()
        sources = set([doc.metadata["source"] for doc in generated_response["context"]])

        formatted_response = (
            f"{generated_response['answer']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)
