from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from graph.chains.answer_grader import GradeAnswer, answer_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


def test_retrieval_grader_yes_answer() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_no_answer() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "What is the capital of France?", "document": doc_txt}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = "The capital of France is Paris."

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert not res.binary_score


def test_answer_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"question": question, "context": docs})

    res: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    assert res.binary_score


def test_answer_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = "The capital of France is Paris."

    res: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_web_search() -> None:
    question = "What is the capital of France?"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "web_search"
