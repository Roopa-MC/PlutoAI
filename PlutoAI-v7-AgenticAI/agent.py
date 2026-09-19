import json

from openai import OpenAI

from tools import calculator, knowledge_search


client = OpenAI()


tools = [
    {
        "type": "function",
        "name": "calculator",
        "description": "Calculate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A mathematical expression such as "
                        "25 * 4 or 840 * 0.25."
                    )
                }
            },
            "required": ["expression"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "knowledge_search",
        "description": (
            "Search PlutoAI's private knowledge base. "
            "Use this when the user asks about information "
            "that may be contained in the private knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The question to search for in "
                        "the private knowledge base."
                    )
                }
            },
            "required": ["question"],
            "additionalProperties": False
        }
    },
    {
        "type": "web_search"
    }
]


def execute_tool(name, arguments, conversation):

    if name == "calculator":
        return calculator(arguments["expression"])

    if name == "knowledge_search":
        return knowledge_search(
            arguments["question"],
            conversation
        )

    return "Unknown tool."


def run_agent(conversation):

    response = client.responses.create(
        model="gpt-5.6",
        input=conversation,
        tools=tools
    )
    for item in response.output:
        print("\nResponse item type:", item.type)

    while True:

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            return response.output_text

        tool_outputs = []

        for item in tool_calls:

            print("\nGPT requested tool:", item.name)
            print("Arguments:", item.arguments)

            arguments = json.loads(item.arguments)

            result = execute_tool(
                item.name,
                arguments,
                conversation
            )

            
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": str(result)
                }
            )

        response = client.responses.create(
            model="gpt-5.6",
            input=[
                *response.output,
                *tool_outputs
            ],
            tools=tools
        )