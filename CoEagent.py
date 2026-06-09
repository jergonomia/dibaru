import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model
from stance_detector import detect_stance

SYSTEM_PROMPT_JUDGE = """CoE Judge Agent Prompt:
Given:
Passage: [Passage]
Evidence Nodes: [Nodes]
Evidence Relations: [Relations]
Intent: [Intent]
Topic: [Topic]
Check the [Passage] for Completeness:
1. Evidence Nodes Coverage
a) Does each sentence in [Passage] contain at least
one [Nodes]?
b) Does the [Passage] explicitly include all items
listed under [Nodes]?
c) Are there any cases where the keywords in [Passage]
are replaced by pronouns or vague synonyms (e.g.,
“it,” “they,” or “this” instead of the actual [Nodes])?
2. Evidence Relations Coverage (Skip if [Relations]
is empty)
a) Does the [Passage] clearly establish or infer all of
the provided [Relations]?
b) Are there any unclear or weakly supported relations
in [Passage]?
3. Intent Entailment
a) Can the specified [Intent] be found in or reasonably
inferred from the [Passage]?
Output Rules:
1) If all criteria are met (i.e., the Passage covers all
[Nodes], [Relations] if present, and [Intent]), output
only: Yes
2) If any criterion is not met:
Provide a set of revision suggestions for the [Passage].
Specifically:
a) Indicate how to add or replace missing keywords
(or remove ambiguous pronouns) in each sentence to
maximize the number of keywords.
b) Tell how to Revise or remove sentences that lack
keywords until each sentence contains at least one
keyword.
c) Explain how to clarify or insert any undefined or
weak relations (if [Relations] are given).
Do not output any step-by-step explanations or chain-
of-thought. Simply give "Yes" if all items are satisfied,
or directly provide the revision suggestions if not.
.
"""

SYSTEM_PROMPT_REVISION = """Revise Agent Prompt:
Given:
Passage: [Passage]
Advise: [Advise]
Instructions:
Incorporate any relevant suggestions from [Advise]
into [Passage].
If there is any conflict between [Passage] and [Advise],
[Advise] takes priority.
Output:
The revised [Passage], fully updated according to
[Advise].
Please limited the revised [Passage] to 100 words.
No explanations or step-by-step reasoning only the
final revised text."""

SYSTEM_PROMPT_REPHRASE = """Rephrase Agent Prompt:
Given:
Topic: [Topic]
Stance: [Stance]
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

def main():

    judge_model = init_chat_model(
        "ollama:llama3:8b",
        temperature=0.1,
        timeout=300,
        max_tokens=2000,
    )

    revise_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=500,
    )

    path = Path("out/intent_agent_results.csv")
    out_path = Path("out/CoE_content.csv")

    max_iter_count = 10
    fieldnames = ["idx", "topic", "stance", "corpus"]
    with open(path, "r", encoding="utf-8") as csvfile, out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            count = 0
            idx = row.get("idx")
            topic = row.get("topic", "")
            stance = row.get("stance", "")
            intent = row.get("intent", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")

            while count < max_iter_count:

                Evaluated_Stance = detect_stance(corpus, topic)


                content = (
                    f"Passage: {corpus}\n"
                    f"Evidence Relations: []\n "
                    f"Evidence Nodes: {evidence_nodes}\n"
                    f"Intent: {intent}\n"
                    f"Topic: {topic}\n"
                    "Evaluate the Passage based on given instruction. "
                    "Do not include any explanations or reasoning in your response."
                    "If no revision is needed, output ONLY 'Yes'"
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": content},
                ]
                judge_result = judge_model.invoke(messages)

                print(f"--- Judge Result for Topic {idx} ---")
                print(judge_result.content.strip())
                print("Stance:", Evaluated_Stance)
                print()

                if judge_result.content.strip().lower() == "yes" or judge_result.content.strip().lower() == "output: yes":
                    break
                else:
                    revise_content = (
                        f"Passage: {corpus}\n"
                        f"Advise: {judge_result.content.strip()}\n"
                        "Revise the Passage according to the Advise. "
                        "Do not include any explanations,reasoning, or notes in your response."
                        "Include ONLY the revised passage in your response"
                        )
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                        {"role": "user", "content": revise_content},
                    ]
                    revise_result = revise_model.invoke(messages)

                    corpus = revise_result.content.strip()

                    print(f"--- Revised corpus for {idx} ---")
                    print(corpus)
                    

                count += 1
            
            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": stance,
                "corpus": corpus, 
            })  


if __name__ == "__main__":    main()