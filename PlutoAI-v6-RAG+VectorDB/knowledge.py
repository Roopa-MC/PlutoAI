import json
import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
EMBEDDINGS_FILE = "embeddings.json"

FAISS_INDEX_FILE = "knowledge.index"

TOP_K = 5

SIMILARITY_THRESHOLD = 0.35

EMBEDDING_MODEL = "text-embedding-3-small"


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI()


# -------------------------------------------------
# Load stored knowledge
# -------------------------------------------------
with open(
    EMBEDDINGS_FILE,
    "r",
    encoding="utf-8"
) as file:

    embeddings_data = json.load(file)


# -------------------------------------------------
# Load FAISS index
# -------------------------------------------------
faiss_index = faiss.read_index(
    FAISS_INDEX_FILE
)


# =================================================
# Build retrieval query using conversation context
# =================================================
def build_retrieval_query(
    question,
    conversation
):

    question_lower = question.lower().strip()

    follow_up_phrases = [
        "it",
        "its",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "what about",
        "how about",
        "and what",
        "and how",
        "and why",
        "and where",
        "and when"
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

            previous_user_question = message.get(
                "content",
                ""
            )

            break

    if not previous_user_question:
        return question

    return (
        f"Previous user question:\n"
        f"{previous_user_question}\n\n"
        f"Current question:\n"
        f"{question}"
    )


# =================================================
# Search knowledge base
# =================================================
def search_knowledge(
    question,
    conversation=None
):

    if conversation is None:

        conversation = []

    # ---------------------------------------------
    # Build contextual retrieval query
    # ---------------------------------------------
    retrieval_query = build_retrieval_query(
        question,
        conversation
    )

    # ---------------------------------------------
    # Create question embedding
    # ---------------------------------------------
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=retrieval_query
    )

    question_embedding = response.data[0].embedding

    # ---------------------------------------------
    # Convert to NumPy vector
    # ---------------------------------------------
    query_vector = np.array(
        [question_embedding],
        dtype="float32"
    )

    # ---------------------------------------------
    # Normalize query vector
    # ---------------------------------------------
    faiss.normalize_L2(query_vector)

    # ---------------------------------------------
    # Search FAISS
    # ---------------------------------------------
    scores, indices = faiss_index.search(
        query_vector,
        TOP_K
    )

    # ---------------------------------------------
    # Map FAISS positions to stored chunks
    # ---------------------------------------------
    keys = list(
        embeddings_data.keys()
    )

    relevant_chunks = []

    seen_chunks = set()

    # ---------------------------------------------
    # Process results
    # ---------------------------------------------
    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index == -1:
            continue

        # Ignore weak matches
        if score < SIMILARITY_THRESHOLD:
            continue

        key = keys[index]

        # Avoid duplicate chunks
        if key in seen_chunks:
            continue

        seen_chunks.add(key)

        chunk = embeddings_data[key]

        relevant_chunks.append(
            {
                "document": chunk["document"],
                "chunk_number": chunk["chunk_number"],
                "content": chunk["content"],
                "score": float(score)
            }
        )

    # ---------------------------------------------
    # Nothing relevant found
    # ---------------------------------------------
    if not relevant_chunks:

        return {
            "found": False,
            "context": "",
            "sources": []
        }

    # ---------------------------------------------
    # Build knowledge context
    # ---------------------------------------------
    context_parts = []

    sources = []

    for chunk in relevant_chunks:

        context_parts.append(
            f"Source: {chunk['document']}\n"
            f"{chunk['content']}"
        )

        if chunk["document"] not in sources:

            sources.append(
                chunk["document"]
            )

    relevant_knowledge = "\n\n".join(
        context_parts
    )

    # ---------------------------------------------
    # Return structured result
    # ---------------------------------------------
    return {
        "found": True,

        "context": relevant_knowledge,

        "sources": sources
    }