import ast
import operator

from knowledge import search_knowledge

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _calculate(node.left)
        right = _calculate(node.right)
        return OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_calculate(node.operand))

    raise ValueError("Unsupported expression")


def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        return _calculate(tree.body)
    except Exception:
        return "Unable to calculate the expression."


def knowledge_search(question, conversation):
    return search_knowledge(question, conversation)
