import os
import json
import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
KNOWLEDGE_FOLDER = "knowledge"

EMBEDDINGS_FILE = "embeddings.json"

FAISS_INDEX_FILE = "knowledge.index"

# Number of characters in each chunk
CHUNK_SIZE = 500

# Number of characters shared between adjacent chunks
CHUNK_OVERLAP = 100

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIMENSION = 1536


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI()


# =================================================
# Split document into overlapping chunks
# =================================================
def split_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        # Move forward while keeping overlap
        start += chunk_size - overlap

    return chunks


# =================================================
# Generate embeddings
# =================================================
def create_embeddings():

    embeddings = {}

    faiss_index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    print("\nGenerating document embeddings...\n")

    # ---------------------------------------------
    # Read every text file
    # ---------------------------------------------
    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read()

        # -----------------------------------------
        # Split document
        # -----------------------------------------
        chunks = split_into_chunks(content)

        print(f"\n{filename}")
        print(f"Total Chunks : {len(chunks)}")

        # -----------------------------------------
        # Generate embedding for each chunk
        # -----------------------------------------
        for chunk_index, chunk in enumerate(chunks):

            print(
                f"Embedding Chunk {chunk_index + 1}"
            )

            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )

            embedding = response.data[0].embedding

            vector = np.array(
                [embedding],
                dtype="float32"
            )

            # Normalize for cosine similarity
            faiss.normalize_L2(vector)

            # Add vector to FAISS
            faiss_index.add(vector)

            # Store metadata + content
            embeddings[
                f"{filename}_chunk_{chunk_index + 1}"
            ] = {

                "document": filename,

                "chunk_number": chunk_index + 1,

                "content": chunk,

                "embedding": embedding
            }

    # ---------------------------------------------
    # Save embeddings
    # ---------------------------------------------
    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            embeddings,
            file,
            indent=4
        )

    # ---------------------------------------------
    # Save FAISS index
    # ---------------------------------------------
    faiss.write_index(
        faiss_index,
        FAISS_INDEX_FILE
    )

    print("\nEmbeddings saved successfully!")

    print(
        f"Total vectors : {faiss_index.ntotal}"
    )


# =================================================
# Main
# =================================================
if __name__ == "__main__":
    create_embeddings()