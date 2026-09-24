import os
import glob
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
import chromadb
from chromadb.utils import embedding_functions

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

STRUCTURED_PROMPT_TEMPLATE = """
Role: You are Zepto's official AI Policy Assistant.
Context: {context}
Task: Answer the user query truthfully based only on the provided policy context.
Negative Constraint: Do not answer using information not present in the provided context.
Format: Output valid JSON matching the schema: {{"answer": "<string>", "sources": ["<doc_id>"], "confidence": <float 0-1>}}.
Length: Keep the answer concise and direct (under 3 sentences).

Few-Shot Example:
Context: doc_01: Standard delivery is free on orders over INR 149; orders below incur INR 25 fee.
Query: What is the delivery fee for a 100 rupee order?
Output: {{"answer": "For an order of INR 100, the delivery fee is a flat INR 25 because it is below the INR 149 free delivery threshold.", "sources": ["doc_01"], "confidence": 1.0}}

Current Query: {query}
Output:
"""

client = chromadb.Client()
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="zepto_policies", embedding_function=emb_fn)

def init_corpus():
    docs_dir = os.path.join(os.path.dirname(__file__), "docs")
    files = sorted(glob.glob(os.path.join(docs_dir, "doc_*.txt")))
    if collection.count() == 0 and files:
        ids = []
        documents = []
        for file_path in files:
            doc_id = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            ids.append(doc_id)
            documents.append(content)
        collection.add(ids=ids, documents=documents)

init_corpus()

class UserQuery(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float

class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_docs: List[str]
    retrieved_ids: List[str]
    response: QueryResponse

POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def classify_intent(state: GraphState) -> Dict[str, Any]:
    query_lower = state["query"].lower()
    is_policy = any(kw in query_lower for kw in POLICY_KEYWORDS)
    intent = "policy_question" if is_policy else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: GraphState) -> Dict[str, Any]:
    query = state["query"]
    results = collection.query(query_texts=[query], n_results=3)
    doc_ids = results["ids"][0] if results["ids"] else []
    docs = results["documents"][0] if results["documents"] else []

    top_snippet = docs[0][:200] if docs else "No policy found"
    resp = QueryResponse(
        answer=f"Based on the retrieved context: {top_snippet}",
        sources=doc_ids,
        confidence=1.0
    )
    return {"retrieved_docs": docs, "retrieved_ids": doc_ids, "response": resp}

def direct_answer(state: GraphState) -> Dict[str, Any]:
    resp = QueryResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {"retrieved_docs": [], "retrieved_ids": [], "response": resp}

def route_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

from langgraph.graph import StateGraph, END
builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges("classify_intent", route_intent, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer"
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
workflow = builder.compile()

app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_policy(payload: UserQuery):
    initial_state: GraphState = {
        "query": payload.query,
        "intent": "",
        "retrieved_docs": [],
        "retrieved_ids": [],
        "response": None
    }
    result = workflow.invoke(initial_state)
    return result["response"]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
