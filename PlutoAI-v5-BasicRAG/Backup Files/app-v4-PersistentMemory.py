from dotenv import load_dotenv
from openai import OpenAI
import os

from memory import load_conversation, save_conversation

# Load environment variables
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load previous conversation
conversation = load_conversation()

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Add user message
    conversation.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Save immediately
    save_conversation(conversation)

    # Get response from OpenAI
    response = client.responses.create(
        model="gpt-5.5",
        input=conversation
    )

    answer = response.output_text

    print("\nRoopaGPT:", answer)

    # Add assistant response
    conversation.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save again
    save_conversation(conversation)