import json
import os


MEMORY_FOLDER = "conversations"
MEMORY_FILE = os.path.join(MEMORY_FOLDER, "conversation.json")


def load_conversation():
    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_conversation(conversation):
    os.makedirs(MEMORY_FOLDER, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            conversation,
            file,
            indent=4,
            ensure_ascii=False
        )