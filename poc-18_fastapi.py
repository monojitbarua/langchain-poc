from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from fastapi import FastAPI
from pydantic import BaseModel

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are a helpful assistant. Explain the following topic in simple terms:
    {topic}
    """
)

# Create the LLM instance
llm = ChatOllama(model="mistral")

# Combine the prompt and LLM into a chain
chain = prompt | llm

app = FastAPI(title="LangChain FastAPI Example")

class QueryInput(BaseModel):
    topic: str

@app.get("/")
def root():
    return {"message": "LLM app is running!"}

@app.post("/explain")
def explain(input: QueryInput):
    result = chain.invoke({"topic": input.topic})
    return {"response": result.content}