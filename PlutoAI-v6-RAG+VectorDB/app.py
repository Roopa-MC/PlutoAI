from dotenv import load_dotenv
from openai import OpenAI
import os

# Import conversation memory functions
from memory import load_conversation, save_conversation

# Import RAG retrieval function
from knowledge import search_knowledge


# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()


# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# -------------------------------------------------
# Load system prompt
# -------------------------------------------------
with open(
    "system_prompt.txt",
    "r",
    encoding="utf-8"
) as file:

    system_prompt = file.read()


# -------------------------------------------------
# Load previous conversation
# -------------------------------------------------
conversation = load_conversation()


# =================================================
# Main Chat Loop
# =================================================
while True:

    # ---------------------------------------------
    # Get user input
    # ---------------------------------------------
    question = input("\nYou: ")

    # ---------------------------------------------
    # Exit
    # ---------------------------------------------
    if question.lower() == "exit":

        break

    # ---------------------------------------------
    # Search knowledge base
    #
    # Conversation is passed so that questions such
    # as "What is it useful for?" can be understood
    # using previous context.
    # ---------------------------------------------
    retrieval_result = search_knowledge(
        question,
        conversation
    )

    # ---------------------------------------------
    # Save user message
    # ---------------------------------------------
    conversation.append(
        {
            "role": "user",
            "content": question
        }
    )

    save_conversation(
        conversation
    )

    # =================================================
    # CASE 1 — No relevant knowledge found
    # =================================================
    if not retrieval_result["found"]:

        answer = (
            "The information is not available "
            "in my knowledge base."
        )

        print(
            "\nRoopaGPT:",
            answer
        )

        conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        save_conversation(
            conversation
        )

        continue

    # =================================================
    # CASE 2 — Relevant knowledge found
    # =================================================

    relevant_knowledge = retrieval_result[
        "context"
    ]

    sources = retrieval_result[
        "sources"
    ]

    # ---------------------------------------------
    # Build model input
    # ---------------------------------------------
    conversation_for_model = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "system",
            "content": f"""
You are a knowledge-grounded assistant.

Follow these rules strictly:

1. Answer using ONLY the retrieved knowledge provided
   below.

2. Do NOT use your general or pre-trained knowledge
   to add factual information.

3. Do NOT invent, assume, or supplement information
   that is not supported by the retrieved knowledge.

4. Previous conversation may be used ONLY to understand
   references such as "it", "that", or "the previous one".

5. Previous conversation must NOT be treated as an
   independent source of factual knowledge.

6. If the retrieved knowledge does not contain enough
   information to answer the question, say:

   "The information is not available in my knowledge base."

7. Keep the answer proportional to the information
   available. Do not expand beyond the retrieved
   knowledge.

Retrieved knowledge:

{relevant_knowledge}
"""
        }
    ]

    # ---------------------------------------------
    # Add conversation history
    # ---------------------------------------------
    conversation_for_model.extend(
        conversation
    )

    # ---------------------------------------------
    # Send request to GPT
    # ---------------------------------------------
    response = client.responses.create(
        model="gpt-5.5",
        input=conversation_for_model
    )

    # ---------------------------------------------
    # Extract answer
    # ---------------------------------------------
    answer = response.output_text

    # ---------------------------------------------
    # Display answer
    # ---------------------------------------------
    print(
        "\nRoopaGPT:",
        answer
    )

    # ---------------------------------------------
    # Display sources
    # ---------------------------------------------
    print(
        "\nSources:",
        ", ".join(sources)
    )

    # ---------------------------------------------
    # Save assistant response
    # ---------------------------------------------
    conversation.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    save_conversation(
        conversation
    )