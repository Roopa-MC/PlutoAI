from tools import calculator, knowledge_search


# Test calculator
result = calculator("25 * 4")

print("Calculator result:", result)


# Test private knowledge search
knowledge_result = knowledge_search(
    "What is PlutoAI?"
)

print("\nKnowledge result:")
print(knowledge_result)
