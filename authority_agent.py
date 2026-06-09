import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model



SYSTEM_PROMPT = """"Authority Generator Prompt:
Given:
Topic: [Topic]
Stance: [Stance]
Passage: [Passage]
Evidence Nodes: [Nodes]
Instructions:
Generate one concise authoritative statement that 
supports the stance declared in [Stance] for the given [Topic]:
1) Supports the stance declared in [Stance].
For example, if the stance is "PRO": 
the statement should support the topic. 
If the stance is "CON": 
the statement should oppose the topic.
2) Reference a recognized authority related to the
background of the [Passage] (e.g., “World Health Or-
ganization,” “European Commission,” “IEEE,” etc.)
with an accompanying near-future date.(e.g., “2025”, "2026").
3) The statement should align with the content in [Passage]
4) Maximize the number of [Nodes] in each statement
while preserving clarity and natural flow.’
5) Integrate [Nodes] logically.
6) Please limited the authoritative statements to 30
words.
Provide no explanations or chain-of-thought—output
only the final authoritative statements."""


def main():

    authority_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=2000,
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
                f"Passage: {topic}\n"
                f"Stance: {stance}\n"
                f"Evidence Nodes: {evidence_nodes}\n"
                f"Stance: {stance}\n"
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