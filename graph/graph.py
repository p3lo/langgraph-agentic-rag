from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END

from graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from graph.state import GraphState
from graph.nodes import retrieve, generate, grade_documents, web_search

def decide_to_generate(state):
    print("---DECIDE TO GENERATE---")

    if state["web_search"]:
        print("---DECISION: Web search---")
        return WEBSEARCH
    else:
        print("---DECISION: Generate---")
        return GENERATE

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")