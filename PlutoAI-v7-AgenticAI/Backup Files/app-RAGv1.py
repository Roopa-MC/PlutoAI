from dotenv import load_dotenv
from openai import OpenAI
import os

# Import conversation memory functions
from memory import load_conversation, save_conversation

# Import RAG Version 1 function
from knowledge import load_knowledge

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

# -------------------------------------------------
# RAG Version 1
# Load ALL knowledge documents once at startup.
# Every document is sent to GPT for every question.
# -------------------------------------------------
knowledge = load_knowledge()


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
    # 2. ALL Knowledge Documents (RAG V1)
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

{knowledge}
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