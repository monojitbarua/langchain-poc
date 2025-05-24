import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

if __name__ == '__main__':
    print("Hello langchain")
    summary_template = '''
        You are an AI assistant. Summarize the following dataset information:
        "{data}".
    '''
    prompt_template = PromptTemplate(input_variables=["data"], template=summary_template)

    llm = ChatOllama(model="mistral")

    chain = prompt_template | llm | StrOutputParser()

    res = chain.invoke({"data": "Dell Technologies"})

    print(res)
