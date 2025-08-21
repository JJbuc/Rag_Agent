from prompts import all_prompts
from llm import load_llms
llms, llm_json_mode = load_llms()
from langchain_core.messages import HumanMessage, SystemMessage
import json

def grade_documents(question, documents):
    print("#### Entered grade_documents ####")
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = all_prompts.doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content= all_prompts.doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            # print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            # print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
    return (filtered_docs, web_search)