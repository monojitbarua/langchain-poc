import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector
from typing import Any, Dict, List

load_dotenv()
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docsearch = PGVector(
    collection_name="langchain_poc_document",
    connection="postgresql+psycopg2://user:password@localhost:5432/vector_db",
    embeddings=embedding,
)

llm = ChatOllama(model="llama3")
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
)

retrieval_chain = create_retrieval_chain(
    retriever=history_aware_retriever, combine_docs_chain=combine_docs_chain
)

def retrieval_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    result = retrieval_chain.invoke(
        input={"input": query, "chat_history": chat_history}
    )
    return result


if __name__ == "__main__":
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # ingest_docs()
    print()
    print(retrieval_llm("What is a Langchain chain?"))
