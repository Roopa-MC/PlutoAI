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


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI()


# -------------------------------------------------
# Load stored embeddings metadata
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
faiss_index = faiss.read_index(FAISS_INDEX_FILE)


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
    # Retrieve corresponding chunks
    # ---------------------------------------------
    results = []

    keys = list(embeddings_data.keys())

    for score, index in zip(scores[0], indices[0]):

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
# Test retrieval
# =================================================
if __name__ == "__main__":

    question = input("\nAsk a question: ")

    results = retrieve_chunks(question)

    print("\n----------------------------------------")
    print("RETRIEVED CHUNKS")
    print("----------------------------------------")

    for result in results:

        print(
            f"\nScore: {result['score']:.4f}"
        )

        print(
            f"Document: {result['document']}"
        )

        print(
            f"Chunk: {result['chunk_number']}"
        )

        print("\nContent:")

        print(result["content"])

        print("\n----------------------------------------")