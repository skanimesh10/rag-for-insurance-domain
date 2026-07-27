"""
Phase 2: the agentic layer.

Instead of always running the same fixed pipeline, this graph first
decides WHAT KIND of question it's answering, then routes to the
right path:

    classify_intent
        -> policy_question -> retrieve_policy (Phase 1 RAG) -> generate_answer
        -> claim_status    -> call_claims_api (tool call)
                                 -> high value?  -> pause_for_human
                                 -> otherwise    -> generate_answer

This is the same conditional-branching + tool-calling + human-in-the-loop
shape the JD calls out explicitly, kept small enough to actually read
end to end.
"""
from functools import partial
import asyncpg
from langgraph.graph import StateGraph, END

from app.agent_state import AgentState
from app.config import settings
from app.retrieval import hybrid_search, RetrievedChunk
from app.rerank import rerank
from app.llm import generate_answer
from app.tools import extract_claim_id, get_claim_status


# ---- Nodes -----------------------------------------------------------------

async def classify_intent(state: AgentState) -> AgentState:
    """
    Simple, deterministic intent routing: a claim ID or the word 'claim'
    means this is a claims-status question; anything else is treated as
    a policy question. (A production system might swap this for an LLM
    classifier or an embedding-based intent classifier -- the graph
    shape doesn't change either way, only this one node would.)
    """
    question_lower = state["question"].lower()
    claim_id = extract_claim_id(state["question"])
    intent = "claim_status" if (claim_id or "claim" in question_lower) else "policy_question"
    return {**state, "intent": intent}


def route_after_classify(state: AgentState) -> str:
    return "call_claims_api" if state["intent"] == "claim_status" else "retrieve_policy"


async def retrieve_policy(state: AgentState, pool: asyncpg.Pool) -> AgentState:
    """The Phase 1 RAG pipeline, reused unchanged as one node in the graph."""
    candidates = await hybrid_search(pool, state["question"], policy_type=state.get("policy_type"))
    top_chunks = rerank(state["question"], candidates)
    return {**state, "retrieved_chunks": top_chunks}


async def call_claims_api(state: AgentState, pool: asyncpg.Pool) -> AgentState:
    """The tool call: look up the claim, then decide if it needs a human."""
    claim_id = extract_claim_id(state["question"])
    claim_result = await get_claim_status(pool, claim_id) if claim_id else None

    needs_approval = bool(
        claim_result
        and (
            claim_result["claim_amount"] > settings.high_value_claim_threshold
            or claim_result["status"] == "pending_human_approval"
        )
    )
    return {**state, "claim_result": claim_result, "needs_human_approval": needs_approval}


def route_after_claims(state: AgentState) -> str:
    return "pause_for_human" if state.get("needs_human_approval") else "generate_answer"


async def pause_for_human(state: AgentState) -> AgentState:
    """
    Human-in-the-loop checkpoint: the agent does NOT auto-answer.
    In a real system this would also write a task/notification for a
    claims officer; here we just surface the pause to the caller.
    """
    answer = (
        "This claim requires human approval before a decision can be shared. "
        "A claims officer has been notified and will follow up directly."
    )
    return {**state, "answer": answer, "citations": []}


async def generate_answer_node(state: AgentState) -> AgentState:
    """
    Grounded generation -- reused for BOTH paths by reframing whatever
    we retrieved (policy chunks OR a claims lookup) as the same
    RetrievedChunk shape, so app/llm.py's prompt-building logic doesn't
    need two versions.
    """
    if state["intent"] == "claim_status":
        claim = state.get("claim_result")
        if claim:
            content = (
                f"Claim amount: {claim['claim_amount']}. Status: {claim['status']}. "
                f"Associated policy: {claim['policy_id']}."
            )
            section = f"Claim {claim['claim_id']}"
        else:
            content = "No claim record was found matching the claim ID in the question."
            section = "Claims lookup"
        chunks = [RetrievedChunk(id=-1, doc_id="claims_system", doc_title="Claims system record",
                                  section=section, content=content, rrf_score=1.0)]
    else:
        chunks = state.get("retrieved_chunks", [])

    answer = generate_answer(state["question"], chunks)
    return {**state, "answer": answer, "citations": chunks}


# ---- Graph assembly ---------------------------------------------------------

def build_agent_graph(pool: asyncpg.Pool):
    """
    Build once (at app startup, with the DB pool already available) and
    reuse the compiled graph across requests -- compiling per-request
    would be wasted work.
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_policy", partial(retrieve_policy, pool=pool))
    graph.add_node("call_claims_api", partial(call_claims_api, pool=pool))
    graph.add_node("pause_for_human", pause_for_human)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"retrieve_policy": "retrieve_policy", "call_claims_api": "call_claims_api"},
    )
    graph.add_edge("retrieve_policy", "generate_answer")
    graph.add_conditional_edges(
        "call_claims_api",
        route_after_claims,
        {"pause_for_human": "pause_for_human", "generate_answer": "generate_answer"},
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("pause_for_human", END)

    return graph.compile()
