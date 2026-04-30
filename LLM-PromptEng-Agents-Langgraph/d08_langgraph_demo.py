from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    user_input: str
    category: str
    final_answer: str

# NODE 1: classify with LLM

def classify_question(state: AgentState) -> dict:
    prompt = [
        SystemMessage(
            content=(
                "Classify the user request into exactly one category: "
                "code or explanation. "
                "Return only one word: code or explanation."
            )
        ),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category not in ["code", "explanation"]:
        category = "explanation"

    return {"category": category}

# NODE 2A: answer code question

def handle_code(state: AgentState) -> dict:
    prompt = [
        SystemMessage(
            content="You are a helpful coding tutor. Give a short, clear code-focused answer."
        ),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(prompt)
    return {"final_answer": response.content}


# NODE 2B: answer explanation question

def handle_explanation(state: AgentState) -> dict:
    prompt = [
        SystemMessage(
            content="You are a helpful teacher. Explain simply with intuition and one example."
        ),
        HumanMessage(content=state["user_input"]),
    ]

    response = llm.invoke(prompt)
    return {"final_answer": response.content}


# ROUTER

def route_question(state: AgentState) -> str:
    if state["category"] == "code":
        return "handle_code"
    return "handle_explanation"


# BUILD GRAPH

builder = StateGraph(AgentState)

builder.add_node("classify_question", classify_question)
builder.add_node("handle_code", handle_code)
builder.add_node("handle_explanation", handle_explanation)

builder.set_entry_point("classify_question") # builder.add_edge(START, "classify_question")

builder.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "handle_code": "handle_code",
        "handle_explanation": "handle_explanation",
    },
)

builder.add_edge("handle_code", END)
builder.add_edge("handle_explanation", END)

graph = builder.compile()

# -----------------------------
# RUN
# -----------------------------
result = graph.invoke({
    "user_input": "Can you explain what embeddings are in simple terms?",
    "category": "",
    "final_answer": "",
})

print("Category:", result["category"])
print("Answer:", result["final_answer"])