from langchain_core.messages import HumanMessage, SystemMessage
import json
from prompts import all_prompts
from llm import load_llms
def route_question(question):
    print("#### Entered route_question ####")
    """
    Route the question to the appropriate LLM and return the response.
    Returns
    websearch or vectorstore
    """
    router_instr = all_prompts.router_instructions
    llm, json_llm = load_llms()
    result = json_llm.invoke(
        [SystemMessage(content=router_instr)] + 
        [HumanMessage(content=question)]
    )
    source = json.loads(result.content)["datasource"]
    print("#### Exiting from route_question ####")
    return source
question = "What are the types of agent memory?"
# print(router(question))

