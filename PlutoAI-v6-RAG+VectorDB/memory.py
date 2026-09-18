import json
import os

FILE_NAME = "conversations/conversation.json"


def load_conversation():

    if not os.path.exists(FILE_NAME):
        return []

    with open(FILE_NAME, "r") as file:
        return json.load(file)


def save_conversation(conversation):

    with open(FILE_NAME, "w") as file:
        json.dump(conversation, file, indent=4)