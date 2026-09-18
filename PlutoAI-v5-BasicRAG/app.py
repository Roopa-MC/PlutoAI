from dotenv import load_dotenv
from openai import OpenAI
import os

# Import conversation memory functions
from memory import load_conversation, save_conversation

# Import RAG Version 2 retrieval function
from knowledge import search_knowledge

# -------------------------------------------------
# Load environment variables from .env
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# Create OpenAI client
# -------------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------
# Load the system prompt
# This defines RoopaGPT's personality and behavior.
# -------------------------------------------------
with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

# -------------------------------------------------
# Load previous conversation from conversation.json
# This gives RoopaGPT persistent memory.
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

    # Exit the application
    if question.lower() == "exit":
        break

    # ---------------------------------------------
    # RAG Version 2
    # Search only the relevant knowledge documents
    # ---------------------------------------------
    relevant_knowledge = search_knowledge(question)

    # ---------------------------------------------
    # Save the user's message into conversation
    # ---------------------------------------------
    conversation.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Save conversation immediately
    save_conversation(conversation)

    # ---------------------------------------------
    # Build the prompt for GPT
    #
    # GPT receives:
    # 1. System Prompt
    # 2. Retrieved Knowledge (RAG)
    # 3. Conversation History
    # ---------------------------------------------
    conversation_for_model = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "system",
            "content": f"""
Use the following knowledge to answer the user's questions.

{relevant_knowledge}
"""
        }
    ]

    # Add previous conversation
    conversation_for_model.extend(conversation)

    # ---------------------------------------------
    # Send request to OpenAI
    # ---------------------------------------------
    response = client.responses.create(
        model="gpt-5.5",
        input=conversation_for_model
    )

    # Extract assistant response
    answer = response.output_text

    # Display answer
    print("\nRoopaGPT:", answer)

    # ---------------------------------------------
    # Save assistant response into conversation
    # ---------------------------------------------
    conversation.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save updated conversation
    save_conversation(conversation)