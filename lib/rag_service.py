from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "when", "where", "why", "who", "whom",
    "which", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "to", "of", "in", "on", "for", "and", "or", "but",
    "with", "about", "into", "from", "by", "as", "at", "if", "so", "than",
    "then", "there", "here", "up", "down", "out", "over", "under", "again",
    "further", "once", "have", "has", "had", "will", "would", "should",
    "could", "can", "may", "might", "must", "shall", "not", "no", "yes",
    "after", "before", "during",
}


def tokenize(text: str) -> set[str]:
    """Break text into a lowercase set of searchable terms, minus stopwords."""
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {word for word in words if word and word not in STOPWORDS}


def document_search_text(document: dict) -> str:
    """Combine a document's metadata and body text into one searchable string."""
    tags_text = " ".join(document.get("tags", []))
    return " ".join(
        [
            document.get("title", ""),
            document.get("category", ""),
            tags_text,
            document.get("text", ""),
        ]
    )


def score_document(query: str, document: dict) -> dict:
    """Score how well a document matches a query's search terms."""
    query_terms = tokenize(query)
    search_text = document_search_text(document).lower()

    matched_terms = set()
    score = 0
    for term in query_terms:
        occurrences = search_text.count(term)
        if occurrences > 0:
            matched_terms.add(term)
            score += occurrences

    return {
        "document": document,
        "score": score,
        "matched_terms": matched_terms,
    }


def retrieve_context(query: str, documents: list[dict], limit: int = 3) -> list[dict]:
    """Return the top-scoring documents for a query, best match first."""
    scored = [score_document(query, document) for document in documents]
    relevant = [match for match in scored if match["score"] > 0]
    relevant.sort(key=lambda match: match["score"], reverse=True)
    return relevant[:limit]


def format_context(matches: list[dict]) -> str:
    """Render retrieved matches into a labeled context block for the prompt."""
    if not matches:
        return "No relevant context was found for this question."

    sections = []
    for match in matches:
        document = match["document"]
        sections.append(
            "\n".join(
                [
                    f"Source ID: {document['id']}",
                    f"Title: {document['title']}",
                    f"Category: {document['category']}",
                    f"Content: {document['text']}",
                ]
            )
        )
    return "\n\n".join(sections)


def build_prompt(query: str, matches: list[dict]) -> str:
    """Assemble the full prompt sent to the model, including grounding rules."""
    context = format_context(matches)
    return (
        "Instructions: You are a helpful assistant that answers employee "
        "questions using only the company documents provided in the context "
        "below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Response requirements:\n"
        "- Use only the information provided in the context above.\n"
        "- Do not invent facts, policies, or sources that are not listed above.\n"
        "- If the context does not contain enough information, say so clearly."
    )


def source_metadata(match: dict) -> dict:
    """Return just the id and title for a retrieved match, safe to expose to clients."""
    document = match["document"]
    return {"id": document["id"], "title": document["title"]}