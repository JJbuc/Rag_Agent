import json
from llm import load_llms
llm, llm_json_mode = load_llms()
from prompts import all_prompts
from langchain_core.messages import HumanMessage, SystemMessage
from generate import format_docs

def grade_generation_v_documents_and_question(question, documents, generation, max_retries, loop_step):
    print("#### Entered hall detector ####")
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    hallucination_grader_prompt_formatted = all_prompts.hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=all_prompts.hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        answer_grader_prompt_formatted = all_prompts.answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=all_prompts.answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("#### Exiting from hall detector with 'finish' ####")
            return "finish"
        elif loop_step <= max_retries:
            return "docs_not_ques"
        else:
            print("#### Exiting from hall detector with 'max retries' ####")
            return "max retries"
    elif loop_step <= max_retries:
        return "ans_not_ques"
    else:
        return "max retries"