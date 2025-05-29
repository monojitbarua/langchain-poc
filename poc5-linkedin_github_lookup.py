import os
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
import requests
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils import get_profile_url_tavilty
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# 1
def lookup(name: str) -> str:
    """This func takes user name as input and search his/her profile in and return the profile link (linkedin)"""
    template = """given the full {name_of_the_person} I want you to get me the person Linkedin or Github profile url. your answer should contain only URL"""
    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_the_person"]
    )

    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

    tools_for_agent = [
        Tool(
            name="Linkdin profile link",
            func=get_profile_url_tavilty,
            description="useful when you need to get the linkedin url ",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_the_person=name)}
    )
    linkedin_profile_url = result["output"]
    return linkedin_profile_url


# 2
def scrap_linkedin_profile(linkedin_profile_url: str, mock: bool = True):
    """Scrape information from linkedin profile and return it"""

    if mock:
        response = requests.get(linkedin_profile_url, timeout=10)

    else:
        print("================this is scrapin call===========")
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url,
        }
        response = requests.get(api_endpoint, params=params, timeout=10)

    return response.json().get("person")


# 3
def profile_summary(data: str):
    """ "summarize profile data into desired format"""

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


# main method/func
if __name__ == "__main__":
    linkedin_url = lookup(name="Monojit Barua linkedin")
    print(f" PROFILE URL ===> {linkedin_url}")

    profile_summary_data = scrap_linkedin_profile(
        linkedin_profile_url=linkedin_url, mock=False
    )

    profile_summary(profile_summary_data)
