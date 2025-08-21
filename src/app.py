from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator
from llm import load_llms
from web_search import web_search_doc
from vectorstore import retrieve_doc
from prompts import all_prompts
from router import route_question
from grade_doc import grade_documents
from generate import generate_response
from hall_detector import grade_generation_v_documents_and_question
llm, json_llm = load_llms()
prompts = all_prompts()
from config import load_config

cfg = load_config()



class GraphState(TypedDict):
    question : str
    output : str
    max_retries : int
    current_loop : Annotated[int, operator.add]
    final_docs : List[str]
    web_search : str

def retrieve(state) -> GraphState:
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retrieve_doc(question)
    return {"final_docs" : documents}

def grade_doc(state) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    question = state["question"]
    documents = state["final_docs"]
    filtered_docs, web_search = grade_documents(question,documents)
    return  {"final_docs": filtered_docs, "web_search": web_search}

def generate(state) -> GraphState:
    """
    Generate a response based on the final documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updated state with the generated response
    """
    question = state["question"]
    documents = state["final_docs"]
    loop_step = state.get("current_loop", 0)
    generation, loop_step = generate_response(question, documents, loop_step)
    return {"output": generation, "current_loop": loop_step + 1}

def web_search(state) -> GraphState:
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """
    question = state["question"]
    # print(state)
    documents = state.get("final_docs", [])
    documents = web_search_doc(question, documents)
    return {"final_docs": documents}

def router(state) -> GraphState:
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    source = route_question(state["question"])
    return source

def dummy_grade(state) -> GraphState:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    questions = state["question"]
    documents = state["final_docs"]
    filtered_docs, web_search = grade_documents(questions, documents)
    state["web_search"] = web_search
    if web_search == "Yes":
        return "nope"
    else:
        return "yup"
    
def hall_detect(state) -> GraphState:
    questions = state["question"]
    documents = state["final_docs"]
    generation = state["output"]
    max_retries = state["max_retries"]
    loop_step = state["current_loop"]
    decision = grade_generation_v_documents_and_question(questions, documents, generation, max_retries, loop_step)
    return decision




workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_doc", grade_doc)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_edge("retrieve", "grade_doc")
workflow.add_edge("grade_doc", "generate")
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    START,
    router,
    {
        "websearch" : "web_search",
        "vectorstore" : "retrieve",
    }
)
workflow.add_conditional_edges(
    "grade_doc",
    dummy_grade,
    {
        "nope" : "web_search",
        "yup" : "generate",
    }
)
workflow.add_conditional_edges(
    "generate",
    hall_detect,
    {
        "finish" : END,
        "ans_not_ques" : "generate",
        "docs_not_ques" : "web_search",
        "max_retries" : END,
    }
)

graph = workflow.compile()
with open("AgentGraph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

question = "Who is the strongest man on earth?"
inputs = {
    "question" : question,
    "max_retries" : cfg["runtime"]["max_retries"]    
}
final_output = None
for event in graph.stream(inputs, stream_mode="values"):
    if "output" in event:
        final_output = event["output"]

if final_output is not None:
    print(final_output)
else:
    print("No output generated.")