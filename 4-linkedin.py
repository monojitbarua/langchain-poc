import os
import requests
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def scrap_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """Scrape information from linkedin profile and return it"""

    if mock:
        response = requests.get(linkedin_profile_url, timeout=10)

    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(api_endpoint, params=params, timeout=10)

    return response.json().get("person")


if __name__ == "__main__":
    data = scrap_linkedin_profile(
        "https://raw.githubusercontent.com/monojitbarua/langchain-poc/refs/heads/main/data/linkedin_gist.json"
    )

    summary_template = """
        Given the Linkedin information "{data}" about this person, please create:
        1. A short summary
        2. 2 interesting facts about this person
    """
    prompt_template = PromptTemplate(
        input_variables=["data"], template=summary_template
    )

    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    chain = prompt_template | llm | StrOutputParser()

    res = chain.invoke({"data": data})

    print(res)
