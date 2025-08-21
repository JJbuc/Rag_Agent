from langchain_ollama import ChatOllama
from functools import lru_cache

@lru_cache(maxsize=1)
def load_llms():    
    local_llm = "llama3.1:8b-instruct-q8_0"
    llm = ChatOllama(model=local_llm, temperature=0)
    llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
    return llm, llm_json_mode