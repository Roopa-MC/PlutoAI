from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Read the system prompt
with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

question = input("You: ")

response = client.responses.create(
    model="gpt-5",
    input=question
)

print("\nRoopaGPT:")
print(response.output_text)