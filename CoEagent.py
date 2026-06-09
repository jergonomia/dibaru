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
Target Stance: [Stance]
Evalueted Stance: [Evaluated Stance]
Check the [Passage] for Completeness:
1. Stance alignment
a) An annotator evaluates the stance of the [Passage] for the given [Topic].
It is given in [Evaluated Stance]. Does the [Evaluated Stance] match the declared [Stance]?
2. Evidence Nodes Coverage
a) Does each sentence in [Passage] contain at least
one [Nodes]?
b) Does the [Passage] explicitly include all items
listed under [Nodes]?
c) Are there any cases where the keywords in [Passage]
are replaced by pronouns or vague synonyms (e.g.,
“it,” “they,” or “this” instead of the actual [Nodes])?
3. Evidence Relations Coverage (Skip if [Relations]
is empty)
a) Does the [Passage] clearly establish or infer all of
the provided [Relations]?
b) Are there any unclear or weakly supported relations
in [Passage]?
4. Intent Entailment
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
d) Indicate how the stance of [Passage] can be improved to better align with the given [Stance]. 
For example, if the [Passage] is evaluated as "PRO" but the target [Stance] is "CON", 
suggest specific ways to make the passage oppose the claim presented in the topic.
If the [Passage] is evaluated as "CON" but the target [Stance] is "PRO", 
suggest specific ways to revise the passage to support the claim presented in the topic."""

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

def main():

    judge_model = init_chat_model(
        "ollama:qwen3.5:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=2000,
    )

    revise_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=300,
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
                    f"Stance: {stance}\n"
                    f"Evaluated Stance: {Evaluated_Stance}\n"
                    "Evaluate the Passage based on given instruction." \
                    "If all conditions are satisfied, OUTPUT ONLY 'Yes"
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": content},
                ]
                judge_result = judge_model.invoke(messages)

                print(f"--- Judge Result for Topic {idx} ---")
                print("CONTENT:", repr(judge_result.content))
                print("METADATA:", judge_result.response_metadata)
                print("USAGE:", judge_result.usage_metadata)
                print()

                if judge_result.content.strip().lower() == "yes" or judge_result.content.strip().lower() == "output: yes":
                    break
                else:
                    revise_content = (
                        f"Passage: {corpus}\n"
                        f"Advise: {judge_result.content.strip()}\n"
                        "Revise the Passage according to the Advise. Include only the updated passage in you response"
                    )
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                        {"role": "user", "content": revise_content},
                    ]
                    revise_result = revise_model.invoke(messages)

                    corpus = revise_result.content.strip()

                    print("--- Revised corpus for {idx} ---")
                    print(corpus)
                    

                count += 1
            
            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": stance,
                "corpus": corpus, 
            })  


if __name__ == "__main__":    main()