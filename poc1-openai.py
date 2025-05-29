import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

if __name__ == "__main__":
    print("Hello langchain")
    print(os.environ["OPENAI_API_KEY"])
    summary_template = """
        Given the information "{data}" about this dataset, please provide a summary.
    """
    prompt_template = PromptTemplate(
        input_variables=["data"], template=summary_template
    )

    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    chain = prompt_template | llm

    res = chain.invoke({"data": "Dell Technologies"})

    print(res)
    print()
    print(json.dumps({"response": res.content}, indent=2))
