"""
The state that flows through the agent graph. Every node reads from
this and returns a partial update -- LangGraph merges the update into
the running state before calling the next node.

Using TypedDict (rather than just passing loose kwargs around) makes
the state's shape explicit and gives you type checking on what each
node is allowed to read/write.
"""
from typing import TypedDict, Optional, Any
from app.retrieval import RetrievedChunk


class AgentState(TypedDict, total=False):
    question: str
    policy_type: Optional[str]

    intent: str                              # "policy_question" | "claim_status"

    retrieved_chunks: list[RetrievedChunk]    # populated by retrieve_policy node
    claim_result: Optional[dict[str, Any]]    # populated by call_claims_api node
    needs_human_approval: bool                # populated by call_claims_api node

    answer: Optional[str]
    citations: list[RetrievedChunk]
