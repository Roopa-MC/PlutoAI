import json
import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

EMBEDDINGS_FILE = "embeddings.json"

FAISS_INDEX_FILE = "knowledge.index"

TOP_K = 5

SIMILARITY_THRESHOLD = 0.35

EMBEDDING_MODEL = "text-embedding-3-small"

load_dotenv()

client = OpenAI()

with open(
    EMBEDDINGS_FILE,
    "r",
    encoding="utf-8"
) as file:
    embeddings_data = json.load(file)


faiss_index = faiss.read_index(
FAISS_INDEX_FILE
)

def build_retrieval_query(
    question,
    conversation
):
    question_lower = question.lower().strip()

    follow_up_phrases = [
        "it", "its", "they", "them", "this", "that", "these", "those",
        "he", "she", "what about", "how about", "and what", "and how",
        "and why", "and where", "and when"
    ]

    needs_context = any(
        phrase in question_lower
        for phrase in follow_up_phrases
    )

    if not needs_context or not conversation:
        return question

    previous_user_question = None

    for message in reversed(conversation):
        if message.get("role") == "user":
            previous_user_question = message.get("content", "")
            break

    if not previous_user_question:
        return question

    return (
        f"Previous user question:\n"
        f"{previous_user_question}\n\n"
        f"Current question:\n"
        f"{question}"
    )


def search_knowledge(
    question,
    conversation=None
):
    if conversation is None:
        conversation = []

    retrieval_query = build_retrieval_query(question, conversation)

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=retrieval_query
    )

    question_embedding = response.data[0].embedding
    query_vector = np.array([question_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    scores, indices = faiss_index.search(query_vector, TOP_K)
    keys = list(embeddings_data.keys())
    relevant_chunks = []
    seen_chunks = set()

    for score, index in zip(scores[0], indices[0]):
        if index == -1 or score < SIMILARITY_THRESHOLD:
            continue

        key = keys[index]
        if key in seen_chunks:
            continue

        seen_chunks.add(key)
        chunk = embeddings_data[key]
        relevant_chunks.append({
            "document": chunk["document"],
            "chunk_number": chunk["chunk_number"],
            "content": chunk["content"],
            "score": float(score)
        })

    if not relevant_chunks:
        return {
            "found": False,
            "context": "",
            "sources": []
        }

    context_parts = []
    sources = []

    for chunk in relevant_chunks:
        context_parts.append(
            f"Source: {chunk['document']}\n{chunk['content']}"
        )
        if chunk["document"] not in sources:
            sources.append(chunk["document"])

    return {
        "found": True,
        "context": "\n\n".join(context_parts),
        "sources": sources
    }