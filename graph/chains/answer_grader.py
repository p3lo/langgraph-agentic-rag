from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class GradeAnswer(BaseModel):
    """Binary score to assess answer relevance to question."""
    binary_score: bool = Field(
        description="Answer is relevant to the question, 'yes' or 'no'"
    )

llm = init_chat_model(
    "deepseek/deepseek-v4.1-flash", temperature=0, model_provider="openrouter"
)

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader