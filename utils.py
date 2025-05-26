from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


def get_profile_url_tavilty(name: str):
    """searches for linkedin or twitter profile url"""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res
