from __future__ import annotations

from flask import Flask, jsonify, request

KNOWLEDGE_BASE = [
    {
        "id": "finance_travel_reimbursement",
        "title": "Travel Reimbursement Procedure",
        "content": (
            "Submit travel receipts within 14 days of your trip completion to the "
            "finance team for reimbursement processing. Include itemized receipts "
            "for lodging, meals, and transportation."
        ),
        "keywords": ["travel", "reimbursement", "receipt", "receipts", "expense", "trip"],
    },
    {
        "id": "dev_api_authentication",
        "title": "API Authentication Guide",
        "content": (
            "Internal API requests must include a signed JWT token issued by the "
            "internal identity provider. Developers should rotate API keys every "
            "90 days."
        ),
        "keywords": ["api", "authenticate", "authentication", "developer", "developers", "token", "jwt"],
    },
    {
        "id": "hr_bereavement_leave",
        "title": "Bereavement Leave Policy",
        "content": (
            "Employees are entitled to five days of paid bereavement leave for the "
            "death of an immediate family member, and three days for extended family."
        ),
        "keywords": ["bereavement", "leave", "grief", "funeral"],
    },
    {
        "id": "it_software_request",
        "title": "Software Access Request Procedure",
        "content": (
            "To request access to approved software, submit a ticket through the "
            "IT service portal specifying the software name and business "
            "justification."
        ),
        "keywords": ["software", "access", "approved", "request", "install"],
    },
]


def generate_response(prompt: str) -> str:
    """Calls out to the model service (e.g. a locally running Ollama model).

    This default raises so the app fails loudly if it's ever called without
    a real model backend wired up. Tests monkeypatch this function directly.
    """
    raise NotImplementedError("generate_response must be wired up to a model service.")


def _find_relevant_documents(query: str) -> list[dict]:
    query_lower = query.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for keyword in doc["keywords"] if keyword in query_lower)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored]


def _build_prompt(query: str, documents: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{doc['id']}] {doc['title']}\n{doc['content']}" for doc in documents
    )
    return (
        "Instructions: Answer the question using only the information in the "
        "context below. If the context does not fully answer the question, say "
        "so rather than guessing.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True) or {}
        query = payload.get("query")

        if not isinstance(query, str):
            return jsonify({"error": "The 'query' field is required and must be a string."}), 400

        if not query.strip():
            return jsonify({"error": "The 'query' field must not be blank."}), 400

        documents = _find_relevant_documents(query)

        if not documents:
            return jsonify(
                {
                    "query": query,
                    "answer": (
                        "The provided documents do not contain enough information "
                        "to answer this question."
                    ),
                    "sources": [],
                }
            )

        prompt = _build_prompt(query, documents)

        try:
            answer = generate_response(prompt)
        except Exception:
            return (
                jsonify(
                    {
                        "error": (
                            "The model service (Ollama) is currently unavailable. "
                            "Please try again later."
                        )
                    }
                ),
                503,
            )

        return jsonify(
            {
                "query": query,
                "answer": answer,
                "sources": [{"id": doc["id"], "title": doc["title"]} for doc in documents],
            }
        )

    return app


if __name__ == "__main__":
    create_app().run(port=5555, debug=True)