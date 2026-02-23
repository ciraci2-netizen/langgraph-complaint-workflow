print("=== FILE EXECUTED ===")

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# ==========================
# LOAD ENV
# ==========================

load_dotenv()


print("KEY:", os.getenv("OPENAI_API_KEY"))

# ==========================
# LLM SETUP
# ==========================

llm = ChatOpenAI(model="gpt-4o-mini")

# ==========================
# STATE DEFINITION
# ==========================

class ComplaintState(TypedDict):
    complaint: str
    category: str
    status: str
    workflow_path: List[str]


# ==========================
# NODE 1 — INTAKE
# ==========================

def intake_node(state: ComplaintState) -> ComplaintState:
    print("\n[INTAKE] Processing complaint...")

    complaint = state["complaint"]

    categorization_prompt = f"""
Categorize this Downside Up complaint into one of these categories:

- portal
- monster
- psychic
- environmental
- other

Complaint: {complaint}

Respond with ONLY the category name.
"""

    response = llm.invoke([HumanMessage(content=categorization_prompt)])
    category = response.content.strip().lower()

    new_state = {
        **state,
        "category": category,
        "workflow_path": state.get("workflow_path", []) + ["intake"],
        "status": "intake"
    }

    print(f"[INTAKE] Categorized as: {category}")
    return new_state


# ==========================
# NODE 2 — VALIDATE
# ==========================

def validate_node(state: ComplaintState) -> ComplaintState:
    print("[VALIDATE] Checking complaint validity...")

    if state["category"] == "other":
        status = "invalid"
    else:
        status = "validated"

    new_state = {
        **state,
        "status": status,
        "workflow_path": state["workflow_path"] + ["validate"]
    }

    print(f"[VALIDATE] Status: {status}")
    return new_state


# ==========================
# NODE 3 — INVESTIGATE
# ==========================

def investigate_node(state: ComplaintState) -> ComplaintState:
    print("[INVESTIGATE] Investigating complaint...")

    new_state = {
        **state,
        "status": "investigated",
        "workflow_path": state["workflow_path"] + ["investigate"]
    }

    return new_state


# ==========================
# NODE 4 — RESOLVE
# ==========================

def resolve_node(state: ComplaintState) -> ComplaintState:
    print("[RESOLVE] Resolving complaint...")

    new_state = {
        **state,
        "status": "resolved",
        "workflow_path": state["workflow_path"] + ["resolve"]
    }

    return new_state


# ==========================
# NODE 5 — CLOSE
# ==========================

def close_node(state: ComplaintState) -> ComplaintState:
    print("[CLOSE] Closing complaint...")

    new_state = {
        **state,
        "status": "closed",
        "workflow_path": state["workflow_path"] + ["close"]
    }

    print("[CLOSE] Complaint closed.")
    return new_state


# ==========================
# BUILD GRAPH
# ==========================

builder = StateGraph(ComplaintState)

builder.add_node("intake", intake_node)
builder.add_node("validate", validate_node)
builder.add_node("investigate", investigate_node)
builder.add_node("resolve", resolve_node)
builder.add_node("close", close_node)

builder.set_entry_point("intake")

builder.add_edge("intake", "validate")
builder.add_edge("validate", "investigate")
builder.add_edge("investigate", "resolve")
builder.add_edge("resolve", "close")
builder.add_edge("close", END)

app = builder.compile()


# ==========================
# STEP 5 — VISUALIZATION
# ==========================

def visualize_workflow(state: ComplaintState):
    print("\nWORKFLOW EXECUTION PATH:")
    
    path = " → ".join(state["workflow_path"])
    print(path)

    print("\nSUMMARY:")
    print(f"Category: {state['category']}")
    print(f"Final Status: {state['status']}")
    print("-" * 60)


# ==========================
# TEST WORKFLOW
# ==========================

if __name__ == "__main__":

    test_complaints = [
        "The Downside Up portal opens at different times each day. How do I predict when?",
        "Demogorgons sometimes work together and sometimes fight. What's their deal?",
        "El can move things with her mind but can't lift heavy rocks. Why?",
        "Why do creatures and power lines react so strangely together?",
        "This is not a valid complaint about something random"
    ]

    print("\nTesting workflow with sample complaints...\n")

    for complaint_text in test_complaints:
        print("=" * 60)
        print(f"Complaint: {complaint_text}")

        initial_state: ComplaintState = {
            "complaint": complaint_text,
            "category": "",
            "status": "",
            "workflow_path": [],
        }

        result = app.invoke(initial_state)

        print("\nFinal Result:")
        print(result)

        visualize_workflow(result)

        print("=" * 60)
        print("\n")

        # ==========================
# STEP 6 — LANGCHAIN COMPARISON
# ==========================

def simple_langchain_approach(complaint: str):
    print("\n--- LANGCHAIN (Simple LLM Call) ---")
    
    prompt = f"""
Analyze this complaint and explain what category it belongs to and what should be done.

Complaint:
{complaint}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STEP 6 — COMPARISON WITH LANGCHAIN APPROACH")
    print("=" * 70)

    sample = "The portal is unstable and damages nearby houses."

    print("\nUsing STRUCTURED LangGraph approach:\n")

    initial_state: ComplaintState = {
        "complaint": sample,
        "category": "",
        "status": "",
        "workflow_path": [],
    }

    result = app.invoke(initial_state)
    visualize_workflow(result)

    print("\nUsing CREATIVE LangChain-only approach:\n")
    simple_langchain_approach(sample)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("""
LangGraph (Structured):
- Deterministic flow
- Clear state transitions
- Trackable workflow path
- Production-ready control
- Easier debugging

LangChain (Creative):
- Single LLM call
- Flexible reasoning
- No state tracking
- No deterministic workflow
- Harder to control at scale

Conclusion:
LangGraph is better for structured workflows.
LangChain is better for flexible, creative reasoning tasks.
""")