import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def embedding_pdf():
    loader = PyPDFLoader(file_path="data/au.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embedding)
    vectorstore.save_local("au_faiss_index")


def rag_pdf():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "au_faiss_index", embedding, allow_dangerous_deserialization=True
    )
    query = "Please give me a summary of the document in 3 sentences"
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
        """
    Answer the question using the provided context.

    Context:
    {context}

    Question:
    {input}
    """
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={"input": query})
    print()
    print(result["answer"])
    print()


if __name__ == "__main__":
    # embedding_pdf()
    rag_pdf()
