import os
import string
import json
import math

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
KNOWLEDGE_FOLDER = "knowledge"
EMBEDDINGS_FILE = "embeddings.json"

# -------------------------------------------------
# Common English words that carry little meaning
# during a search.
#
# These words are ignored so that retrieval focuses
# on the important words in the question.
# -------------------------------------------------
STOP_WORDS = {
    "the", "is", "are", "a", "an", "what", "who", "where", "when",
    "tell", "me", "about", "please", "of", "to", "and",
    "in", "on", "for", "with", "my", "your"
}

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI()


# =================================================
# RAG VERSION 1 (Reference Only)
#
# Reads every text document and combines them into
# one large string.
#
# This function is kept only for learning purposes.
# It is no longer used by the latest RAG.
# =================================================
def load_knowledge():

    knowledge = ""

    print("\n========== RAG Version 1 ==========")

    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

            filepath = os.path.join(KNOWLEDGE_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:

                knowledge += file.read()
                knowledge += "\n\n"

            print(f"Loaded: {filename}")

    return knowledge


# =================================================
# Helper Function
#
# Cleans text before searching.
#
# Example:
#
# Ericsson's antenna products?
#
# becomes
#
# ericssons antenna products
# =================================================
def clean_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text


# =================================================
# Load document embeddings from embeddings.json
#
# Returns a dictionary.
# =================================================
def load_embeddings():

    with open(
        EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        embeddings = json.load(file)

    return embeddings


# =================================================
# Generate embedding for the user's question
# =================================================
def get_question_embedding(question):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    return response.data[0].embedding


# =================================================
# Calculate Cosine Similarity
#
# Higher score = more similar.
# =================================================
def cosine_similarity(vector1, vector2):

    dot_product = sum(
        a * b
        for a, b in zip(vector1, vector2)
    )

    magnitude1 = math.sqrt(
        sum(a * a for a in vector1)
    )

    magnitude2 = math.sqrt(
        sum(b * b for b in vector2)
    )

    return dot_product / (magnitude1 * magnitude2)


# =================================================
# RAG Version 4
#
# Semantic Retrieval using Embeddings
# =================================================
def search_knowledge(question):

    # -----------------------------------------
    # Load all document embeddings
    # -----------------------------------------
    embeddings = load_embeddings()

    # -----------------------------------------
    # Generate embedding for the user's question
    # -----------------------------------------
    question_embedding = get_question_embedding(question)

    print("\nQuestion embedding generated.")
    print(f"Embedding size: {len(question_embedding)}")

    # -----------------------------------------
    # Debug Information
    # -----------------------------------------
    print(type(embeddings))
    print(len(embeddings))

    print("\nDocuments in embeddings.json:")

    for filename in embeddings:
        print(filename)

    # -----------------------------------------
    # Store the final knowledge sent to GPT
    # -----------------------------------------
    relevant_knowledge = ""

    # -----------------------------------------
    # Store document rankings
    # (score, filename, content)
    # -----------------------------------------
    document_scores = []

    # -----------------------------------------
    # Clean the user's question
    # -----------------------------------------
    clean_question = clean_text(question)

    # -----------------------------------------
    # Remove stop words
    # (Currently used only for debugging)
    # -----------------------------------------
    keywords = [
        word
        for word in clean_question.split()
        if word not in STOP_WORDS
    ]

    print("\nKeywords used for search:")
    print(keywords)

    # -----------------------------------------
    # Compare the user's question against every
    # document embedding.
    # -----------------------------------------
    for filename in embeddings:

        document_embedding = embeddings[filename]["embedding"]

        content = embeddings[filename]["content"]

        score = cosine_similarity(
            question_embedding,
            document_embedding
        )

        print(f"Document: {filename:<25} Score: {score:.4f}")

        document_scores.append(
            (score, filename, content)
        )

    # -----------------------------------------
    # Sort documents by similarity score
    # -----------------------------------------
    document_scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # -----------------------------------------
    # Display document ranking
    # -----------------------------------------
    print("\n========== Document Ranking ==========")

    for score, filename, _ in document_scores:

        print(f"{filename:<25} Score: {score:.4f}")

    # -----------------------------------------
    # Keep only the Top 3 documents
    # -----------------------------------------
    top_documents = document_scores[:3]

    print("\nTop Documents Sent to GPT:")

    for score, filename, content in top_documents:

        print(f"{filename:<25} Score: {score:.4f}")

        relevant_knowledge += f"\n--- {filename} ---\n"
        relevant_knowledge += content
        relevant_knowledge += "\n\n"

    # -----------------------------------------
    # Return the selected documents
    # -----------------------------------------
    return relevant_knowledge
