import argparse
import os

from openai import OpenAI, OpenAIError, RateLimitError


def send_query(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Send a query to the LLM and return the assistant response."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The OPENAI_API_KEY environment variable must be set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a query to the LLM.")
    parser.add_argument("-q", "--query", help="The query to send to the LLM.")
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="The OpenAI model to use (default: gpt-4o-mini).",
    )

    args = parser.parse_args()
    prompt = args.query or input("Enter your query: ")
    if not prompt.strip():
        raise SystemExit("A non-empty query is required.")

    try:
        answer = send_query(prompt, model=args.model)
    except RateLimitError:
        print("\nError: quota exceeded or rate limit reached. Check your OpenAI plan and billing.")
        return
    except OpenAIError as exc:
        print("\nOpenAI API error:", exc)
        return

    print("\nLLM response:\n")
    print(answer)


if __name__ == "__main__":
    main()
