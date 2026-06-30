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

SYSTEM_PROMPT_ALT = """Stance Classifier Prompt:
Given:
Passage:
Statement:

You are acting as a human classifier.
Your job is to determine whether the given passage supports the given statement.

That is, if the premise is the passage:
Premise: [Passage]

The hypothesis is then:
The passage supports the statement: [Statement]

If the hypothesis is accepted, meaning that the arguments
in [Passage] ultimately support the [Statement], you should
ouput the word 'Yes'.
If the hypothesis is rejected, meaning that the arguments
in [Passage] ultimately do not support the [Statement], you shoul
output the word 'No'.

In your output, give only the single word without any reasoning or chain-of-thought
"""

SYSTEM_PROMPT_STATE = """Rephrase Agent Prompt:
Given:
Topic: [Topic]
Stance: CON
Your task is to rephrase the [Topic] into a statement that aligns with the given [Stance]
Make the statement absolute: do not use words like "propably", "likely", "necessarily" etc.
Return ONLY the rephrased topic as plain text without any explanations or additional information.
Some examples:
Example 1:
Topic: "Does the rise of antisemitism in recent years indicate a failure on the part of governments and society to address this issue effectively?"
Stance: PRO
Rephrased Statement: "The rise of antisemitism in recent years indicates a failure on the part of governments and society"
Stance: CON
Rephrased Statement: "The rise of antisemitism in recent years does not indicate a failure on the part of governments and society"
Example 2:
Topic: "Does the Australian Classification Board's classification system require an update?"
Stance: PRO
Rephrased Statement: "The Australian Classification Board's classification system requires an update"
Stance: CON
Rephrased Statement: "The Australian Classification Board's classification system does not require an update"
Example 3:
Topic: "Did the Soviet Union exploit the resources of its republics?"
Stance: PRO
Rephrased Statement: "The Soviet Union exploited the resources of its republics"
Stance: CON
Rephrased Statement: "The Soviet Union did not exploit the resources of its republics"
"""


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
        "ollama:qwen3.5:9b",
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
