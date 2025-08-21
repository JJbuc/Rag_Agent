from llm import load_llms
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import all_prompts
llm, json_mode_llm = load_llms()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(question, documents, loop_step):
    print("#### Entered generate_response ####")
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = all_prompts.rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    print("#### Exiting from generate_response ####")
    return (generation, loop_step)