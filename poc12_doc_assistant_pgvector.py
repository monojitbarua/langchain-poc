import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# from langchain_community.vectorstores.pgvector import PGVector


from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# # this needs to be called first to load the chunks in db
def ingest_docs():
    """This will load all document in pinecore vector db."""
    loader = ReadTheDocsLoader("data/langchain-docs/api.python.langchain.com/en/latest")
    documents = loader.load()
    print(f"documents length: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=documents)

    for doc in docs:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("data/langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add vector store (pinecone) data : {len(docs)}")
    # PGVector.from_documents(
    #     docs,
    #     embedding,
    #     collection_name="langchain_poc_document",
    #     connection_string="postgresql+psycopg2://user:password@localhost:5432/vector_db"
    # )
    PGVector.from_documents(
        documents=docs,
        collection_name="langchain_poc_document",
        connection="postgresql+psycopg2://user:password@localhost:5432/vector_db",
        embedding=embedding,
    )


def retrieval_llm(query: str):
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    docsearch = PGVector(
        collection_name="langchain_poc_document",
        connection="postgresql+psycopg2://user:password@localhost:5432/vector_db",
        embeddings=embedding,
    )
    chat = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {input}
    """
    )

    combine_docs_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # ingest_docs()
    print()
    print(retrieval_llm("What is a Langchain chain?"))
