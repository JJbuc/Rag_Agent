import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from config import load_config

cfg = load_config()
k = cfg["web"]["web_search_k"]
# print(k)
def set_tavily():
    os.environ["TAVILY_API_KEY"] = cfg["web"]["tavily_api"]
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_web_search():
    set_tavily()
    web_search_tool = TavilySearchResults(k = k)
    return web_search_tool

def web_search_doc(question, documents):
    print("#### Entered web_search_doc ####")
    """ Appends the results to the documents"""
    web_search_tool = get_web_search()
    docs = web_search_tool.invoke({"query" : question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    print("#### Exiting web_search_doc ####")
    return documents
