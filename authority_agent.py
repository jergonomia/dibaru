import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model



SYSTEM_PROMPT = """"Authority Generator Prompt:
Given:
Topic: [Topic]
Passage: [Passage]
Evidence Nodes: [Nodes]
Instructions:
Generate one concise authoritative statement that 
aligns with the arguments presented in [Passage]:
1) Reference a recognized authority related to the
background of the [Passage] (e.g., “World Health Or-
ganization,” “European Commission,” “IEEE,” etc.)
with an accompanying near-future date.(e.g., “2025”, "2026").
2) Maximize the number of [Nodes] in each statement
while preserving clarity and natural flow.
3) Integrate [Nodes] logically.
4) Please limited the authoritative statements to 30
words.
Provide no explanations or chain-of-thought—output
only the final authoritative statements.

It’s a creative debate game where you are given a topic
and pitted against an opponent. To win the debate against 
your opponent, your teammate requires an authoritative 
statement to support their arguments."""


def main():

    authority_model = init_chat_model(
        "ollama:gemma3:4b",
        temperature=0.1,
        timeout=300,
        max_tokens=300,
    )


    path = Path("out/intent_agent_results.csv")
    out_path = Path("out/authority_content.csv")

    fieldnames = ["idx", "topic", "stance", "statement"]
    with open(path, "r", encoding="utf-8") as csvfile, out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            idx = row.get("idx")
            topic = row.get("topic", "")
            stance = row.get("stance", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")


            content = (
                f"Topic: {topic}\n"
                f"Passage: {corpus}"
                f"Evidence Nodes: {evidence_nodes}\n"
                "Provide only the authoritative statement without any explanations or markdown formatting.")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
            raw = authority_model.invoke(messages)

            results = raw.content.strip()

            print(f"--- Authority Statement for Topic {idx} ---")
            print(results)
            print()


            
            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": stance,
                "statement": results,
            })  


if __name__ == "__main__":    main()