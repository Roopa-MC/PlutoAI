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

TOP_K = 3

MODEL = "gpt-4.1-mini"


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
# Retrieve relevant chunks
# =================================================
def retrieve_chunks(question):

    # ---------------------------------------------
    # Create embedding for the question
    # ---------------------------------------------
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    question_embedding = response.data[0].embedding

    # ---------------------------------------------
    # Convert to NumPy array
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
    # Map FAISS results back to text
    # ---------------------------------------------
    results = []

    keys = list(embeddings_data.keys())

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index == -1:
            continue

        key = keys[index]

        chunk = embeddings_data[key]

        results.append({
            "score": float(score),
            "document": chunk["document"],
            "chunk_number": chunk["chunk_number"],
            "content": chunk["content"]
        })

    return results


# =================================================
# Generate answer using retrieved knowledge
# =================================================
def generate_answer(question, retrieved_chunks):

    # ---------------------------------------------
    # Build knowledge context
    # ---------------------------------------------
    context = ""

    for chunk in retrieved_chunks:

        context += (
            f"\nSource: {chunk['document']}"
            f"\n{chunk['content']}\n"
        )

    # ---------------------------------------------
    # Create prompt
    # ---------------------------------------------
    prompt = f"""
Answer the user's question using the knowledge provided below.

If the answer cannot be found in the provided knowledge,
say that the information is not available in the knowledge base.

Knowledge:
{context}

User question:
{question}
"""

    # ---------------------------------------------
    # Ask the LLM
    # ---------------------------------------------
    response = client.responses.create(
        model=MODEL,
        input=prompt
    )

    return response.output_text


# =================================================
# Main RAG application
# =================================================
if __name__ == "__main__":

    question = input("\nAsk a question: ")

    print("\nSearching knowledge base...\n")

    retrieved_chunks = retrieve_chunks(
        question
    )

    print("Generating answer...\n")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("----------------------------------------")
    print("ANSWER")
    print("----------------------------------------")

    print(answer)

    print("----------------------------------------")