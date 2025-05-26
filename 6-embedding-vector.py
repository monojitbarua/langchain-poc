import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings


load_dotenv()

if __name__ == "__main__":
    """Loading a data into pinecone vector store index"""

    print("1. loading .....")
    loader = TextLoader("data/vector_test_data.txt")
    document = loader.load()

    print("2. splitting .....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"splitting length: {len(texts)}")

    # embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("3. ingesitng ....")
    PineconeVectorStore.from_documents(
        texts,
        embedding,
        index_name=os.environ["INDEX_NAME"],
    )
    print("finish")
