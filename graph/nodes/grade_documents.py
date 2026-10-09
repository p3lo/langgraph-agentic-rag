import json
import os
from typing import Any, Dict

import requests
from langchain_typesafe import Noul, TypeSafeClassifier

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        classifier = TypeSafeClassifier(
            base_url="https://openrouter.ai/api",
            model="openai/gpt-6-luna-decisions",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        QUESTIONS = {
            "is_relevant": Noul(
                instructions=f"Determine whether the following document: '{d.page_content}' is relevant to the question"
            )
        }
        res = classifier.invoke({"state": question, "questions": QUESTIONS})
        relevant_score = res.nouls["is_relevant"].noul
        print(f"Answer by TypeSafeClassifier: {relevant_score}")

        # response_with_jev = requests.post(
        #     url="https://openrouter.ai/api/alpha/decisions",
        #     headers={
        #         "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        #         "Content-Type": "application/json",
        #         "HTTP-Referer": "http://agentic-rag",
        #         "X-OpenRouter-Title": "Agentic RAG"
        #     },
        #     data=json.dumps({
        #         "model": "openai/gpt-6-luna-decisions",
        #         "state": question,
        #         "questions": {
        #             "is_relevant": {
        #                 "type": "noul",
        #                 "instructions": f"Determine whether the following document: '{d.page_content}' is relevant to the question",
        #                 "criteria": {
        #                     "true": "The document is relevant to the question",
        #                     "false": "The document is not relevant to the question"
        #                 }
        #             }
        #         }
        #     })
        # )
        # answers = response_with_jev.json()["answers"]
        # print(f"Answer by JEV: {answers["is_relevant"]["noul"]}")
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
