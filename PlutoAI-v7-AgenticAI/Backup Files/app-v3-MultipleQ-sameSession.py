from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Read the system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

conversation = []

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    conversation.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = client.responses.create(
        model="gpt-5",
        instructions=system_prompt,
        input=conversation
    )

    answer = response.output_text

    print("\nRoopaGPT:")
    print(answer)

    conversation.append(
        {
            "role": "assistant",
            "content": answer
        }
    )