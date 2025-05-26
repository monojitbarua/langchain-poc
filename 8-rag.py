import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    """ "Retrive from vector store, custom promt"""

    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know the answer, don't try to make up an answer.
    Use three sentence maximum and keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}

    Question: {question}
    """

    query = "What is Pinecone is machine learning?"
    custom_rag_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embedding,
    )
    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )
    res = rag_chain.invoke(query)
    print(res)
