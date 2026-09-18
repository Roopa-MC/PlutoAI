import os
import string

# -------------------------------------------------
# Folder containing all knowledge documents
# -------------------------------------------------
KNOWLEDGE_FOLDER = "knowledge"

# -------------------------------------------------
# Common English words that carry little meaning
# during a search.
#
# These words will be ignored so that the retrieval
# focuses only on important keywords.
# -------------------------------------------------
STOP_WORDS = {
    "the", "is", "are", "a", "an", "what", "who", "where", "when",
    "tell", "me", "about", "please", "of", "to", "and",
    "in", "on", "for", "with", "my", "your"
}

# -------------------------------------------------
# RAG Version 1 (Reference Only)
#
# Reads every text file in the knowledge folder
# and returns all contents.
#
# This function is no longer used by app.py,
# but is kept for learning purposes.
# -------------------------------------------------
def load_knowledge():

    knowledge = ""

    print("\n========== RAG Version 1 ==========")

    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

            filepath = os.path.join(KNOWLEDGE_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:

                knowledge += file.read()
                knowledge += "\n\n"

            # Debug Output
            print(f"Loaded: {filename}")

    return knowledge


# -------------------------------------------------
# Helper Function
#
# Cleans the user's question before searching.
#
# Tasks:
# 1. Convert to lowercase
# 2. Remove punctuation
#
# Example:
#
# Ericsson's antenna products?
#
# becomes
#
# ericssons antenna products
# -------------------------------------------------
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    return text


# -------------------------------------------------
# RAG Version 3
#
# Searches the knowledge folder and returns only
# the most relevant documents.
#
# Improvements over Version 2:
#
# ✔ Cleans the user's question
# ✔ Removes stop words
# ✔ Scores every document
# ✔ Ranks the documents
# ✔ Sends only the Top 3 documents to GPT
# -------------------------------------------------
def search_knowledge(question):

    # Stores the final knowledge sent to GPT
    relevant_knowledge = ""

    # Stores (score, filename, content)
    document_scores = []

    # -----------------------------------------
    # Step 1
    # Clean the user's question
    # -----------------------------------------
    clean_question = clean_text(question)

    # -----------------------------------------
    # Step 2
    # Remove stop words
    # -----------------------------------------
    keywords = [
        word
        for word in clean_question.split()
        if word not in STOP_WORDS
    ]

    print("\nKeywords used for search:")
    print(keywords)

    # -----------------------------------------
    # Step 3
    # Search every document
    # -----------------------------------------
    for filename in os.listdir(KNOWLEDGE_FOLDER):

        if filename.endswith(".txt"):

            filepath = os.path.join(KNOWLEDGE_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            # -----------------------------------------
            # Step 4
            # Calculate document score
            # -----------------------------------------
            score = 0

            for keyword in keywords:
                score += content.lower().count(keyword)

            # Debug Output
            print(f"Document: {filename:<25} Score: {score}")

            # Store only matching documents
            if score > 0:
                document_scores.append(
                    (score, filename, content)
                )

    # -----------------------------------------
    # Step 5
    # Sort documents by score
    # Highest score comes first
    # -----------------------------------------
    document_scores.sort(reverse=True)

    # -----------------------------------------
    # Debug Output
    # Show document ranking
    # -----------------------------------------
    print("\n========== Document Ranking ==========")

    for score, filename, _ in document_scores:
        print(f"{filename:<25} Score: {score}")

    # -----------------------------------------
    # Step 6
    # Keep only the Top 3 documents
    # -----------------------------------------
    top_documents = document_scores[:3]

    print("\nTop Documents Sent to GPT:")

    for score, filename, content in top_documents:

        print(f"{filename:<25} Score: {score}")

        relevant_knowledge += f"\n--- {filename} ---\n"
        relevant_knowledge += content
        relevant_knowledge += "\n\n"

    # -----------------------------------------
    # Return only the selected knowledge
    # -----------------------------------------
    return relevant_knowledge