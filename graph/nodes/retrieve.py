from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from vectorstore
    
    Args:
        state (GraphState): The current state of the graph
        
    Returns:
        Dict[str, Any]: The updated state with retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Get documents from vectorstore
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}
