from agent import run_agent
from memory import load_conversation, save_conversation

conversation = load_conversation()

while True:
    user_question = input("\nYou: ")

    if user_question.lower() == "exit":
        break

    conversation.append({
        "role": "user",
        "content": user_question
    })

    answer = run_agent(conversation)

    conversation.append({
        "role": "assistant",
        "content": answer
    })

    save_conversation(conversation)

    print("\nPlutoAI:", answer)