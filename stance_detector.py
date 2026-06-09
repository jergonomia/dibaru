import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model


SYSTEM_PROMPT = """Stance Detection Prompt:
    Given:
    Passage: [Passage]
    Topic: [Topic]
    You are acting as a human annotator. 
    Your job is to determine the stance of a given passage for a topic.
    The stance can be one of the following:
- PRO: The passage supports the topic.
- CON: The passage opposes the topic.
- NEU: The passage is neutral or does not take a clear stance on the topic.
- UNK: The passage is unclear or insufficient to determine a stance.
    Include only the final stance in your response, without any reasoning or justification."""


def detect_stance(passage: str, topic: str) -> str:
    """Detect stance for a single input sentence.

    Returns one of: "PRO", "CON", "NEU", "UNK".
    """
    model = init_chat_model(
        "ollama:qwen2.5:7b",
        temperature=0.1,
        timeout=300,
        max_tokens=1000,
    )

    content = (
        f"Passage: {passage}\n"
        f"Topic: {topic}\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    res = model.invoke(messages)
    raw = res.content.strip() if hasattr(res, "content") else str(res)

    label = raw.strip().upper()
    if "PRO" in label or "SUPPORT" in label:
        return "PRO"
    if "CON" in label or "AGAINST" in label or "OPPOS" in label:
        return "CON"
    if "NEU" in label or "NEUTRAL" in label:
        return "NEU"
    if "UNK" in label or "UNCLEAR" in label or "INSUFFICIENT" in label:
        return "UNK"

    token = label.split()[0] if label else ""
    if token in {"PRO", "CON", "NEU", "UNK"}:
        return token
    return "UNK"



def main():
    model = init_chat_model(
        "ollama:qwen2.5:7b",
        temperature=0.1,
        timeout=300,
        max_tokens=1000,
    )
    print("Model initialized successfully.")
    path = Path("out/intent_agent_results.csv")

    with path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            idx = row.get("idx")
            topic = row.get("topic", "")
            stance = row.get("stance", "")
            intent = row.get("intent", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")

            content = (
                f"Passage: {corpus}\n"
                f"Topic: {topic}\n"
            )

            message = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]

            raw = model.invoke(message)
            res = raw.content.strip()

            print(f"--- Stance Detection Result for Question {idx} ---")
            print(f"Topic: {topic}")
            print(f"Passage: {corpus}")
            print(f"Detected Stance: {res}")
            print()


if __name__ == "__main__":
    main()
