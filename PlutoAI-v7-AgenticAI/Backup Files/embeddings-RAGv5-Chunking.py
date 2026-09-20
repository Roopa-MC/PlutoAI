import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
KNOWLEDGE_FOLDER = "knowledge"
EMBEDDINGS_FILE = "embeddings.json"

# Number of characters per chunk
CHUNK_SIZE = 500

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI()


# =================================================
# Split a document into smaller chunks
#
# Example:
#
# Document
# -----------------------------
# 1500 characters
#
# becomes
#
# Chunk 1 (500 chars)
# Chunk 2 (500 chars)
# Chunk 3 (500 chars)
# =================================================
def split_into_chunks(text, chunk_size=CHUNK_SIZE):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# =================================================
# Generate embeddings for every chunk
#
# This function:
#
# 1. Reads every text file
# 2. Splits each file into chunks
# 3. Generates an embedding for every chunk
# 4. Saves everything into embeddings.json
# =================================================
def create_embeddings():

    embeddings = {}

    print("\nGenerating document embeddings...\n")

    # ---------------------------------------------
    # Read every text file
    # ---------------------------------------------
    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

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
            # Split document into chunks
            # -----------------------------------------
            chunks = split_into_chunks(content)

            print(f"\n{filename}")
            print(f"Total Chunks : {len(chunks)}")

            # -----------------------------------------
            # Generate embedding for each chunk
            # -----------------------------------------
            for index, chunk in enumerate(chunks):

                print(
                    f"Embedding Chunk {index + 1}"
                )

                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )

                embedding = response.data[0].embedding

                print(
                    f"Embedding Size : {len(embedding)}"
                )

                # Store embedding
                embeddings[
                    f"{filename}_chunk_{index + 1}"
                ] = {

                    "document": filename,

                    "chunk_number": index + 1,

                    "content": chunk,

                    "embedding": embedding
                }

    # ---------------------------------------------
    # Save all embeddings
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

    print("\nEmbeddings saved successfully!")
