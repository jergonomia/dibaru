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
Statement: [Statement]
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
d) The revisions should not alter the intent of the passage.
The passage currently supports the statement made in [Statement].
Any modifications made HAS TO ENSURE THIS IS MAINTAINED. Do not
include any of your own thoughts or knowledge on the subject.
Assume that what is said in the passage is the absolute truth, 
while your job is only to ensure node and relationship coverage.
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

SYSTEM_PROMPT_EVIDENCE_REL = """Evidence Relations Extraction Prompt:
Extract evidence relations from the input topic and evidence nodes. Requirements: 1) Each relation contains two
elements: implied evidence nodes and relation description 2) Relation descriptions only involve the two connected nodes
3) Skip if no relation exists between nodes
Output must be in JSON format. Examples:
E1: Q: 750 7th Avenue and 101 Park Avenue, are located in which city? Nodes: ["750 7th Avenue", "101 Park Avenue"]
Out: []
E2: Q: Lee Jun-fan played what character in "The Green Hornet" television series? Nodes: ["Lee Jun-fan", "The Green
Hornet"] Out: {"Evidence nodes":["Lee Jun-fan", "The Green Hornet"], "Evidence Relations": "played character in"}
E3: Q: In which stadium do the teams owned by Myra Kraft’s husband play? Nodes: ["teams", "Myra Kraft’s husband"]
Out: {"Evidence nodes":["teams", "Myra Kraft’s husband"], "Evidence Relations": "is owned by"]}
E4: Q: The Colts’ first ever draft pick was a halfback who won the Heisman Trophy in what year? Nodes: ["Colts’ first
ever draft pick", "halfback", "Heisman Trophy"] Out: {"Evidence nodes":["Colts’ first ever draft pick", "halfback"],
"Evidence Relations": "was"}
E5: Q: The Golden Globe Award winner for best actor from "Roseanne" starred along what actress in Gigantic? Nodes:
["Golden Globe Award winner", "best actor", "Roseanne", "Gigantic"] Out: {"Evidence nodes":["Golden Globe Award
winner", "best actor"], "Evidence Relations": "for", "Evidence nodes":["best actor", "Roseanne"], "Evidence Relations":
"starred in"}
Question: [Question] Evidence nodes: [Evidence node] Output:"""

def run_coe_agent():

    judge_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=2000,
    )

    revise_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=500,
    )

    evidence_rel_model = init_chat_model(
        "ollama:llama3:8b",
        temperature=0.1,
        timeout=300,
        max_tokens=300,
    )

    path = Path("out/intent_agent_results.csv")
    out_path = Path("out/CoE_content.csv")

    max_iter_count = 10
    fieldnames = ["idx", "topic", "stance", "corpus"]
    with open(path, "r", encoding="utf-8") as csvfile, out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        for row in reader:
            count = 0
            idx = row.get("idx")
            topic = row.get("topic", "")
            stance = row.get("stance", "")
            intent = row.get("intent", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")

            if not corpus:
                continue

            # Call evidence relations model to extract relations from topic and evidence nodes
            evidence_relations = []
            try:
                rel_prompt = (
                    f"Question: {topic}\n"
                    f"Evidence nodes: {evidence_nodes}\n"
                    "In your response, include nothing put the output in the specified JSON format." \
                    "Do not include justifications or notes of the relationships"
                )
                rel_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_EVIDENCE_REL},
                    {"role": "user", "content": rel_prompt},
                ]
                rel_result = evidence_rel_model.invoke(rel_messages)
                rel_text = rel_result.content.strip()
                print("Relations:", rel_text)
                try:
                    parsed = json.loads(rel_text)
                    # If parsed is a dict, extract the "Evidence Relations" key
                    if isinstance(parsed, dict):
                        for key in ("Evidence Relations", "EvidenceRelations", "evidence_relations", "Evidence_Relations", "Evidence relations"):
                            if key in parsed:
                                evidence_relations = parsed[key]
                                break
                        else:
                            evidence_relations = parsed
                    else:
                        evidence_relations = parsed
                except Exception:
                    # attempt to extract JSON substring if model added surrounding text
                    import re
                    m = re.search(r"(\[.*\]|\{.*\})", rel_text, re.S)
                    if m:
                        parsed = json.loads(m.group(1))
                        if isinstance(parsed, dict):
                            for key in ("Evidence Relations", "EvidenceRelations", "evidence_relations", "Evidence_Relations", "Evidence relations"):
                                if key in parsed:
                                    evidence_relations = parsed[key]
                                    break
                            else:
                                evidence_relations = parsed
                        else:
                            evidence_relations = parsed
                    else:
                        print("Fail 2")
                        evidence_relations = []
            except Exception:
                evidence_relations = []
                print("Fail 1")

            content = {}

            while count < max_iter_count:


                content = (
                    f"Passage: {corpus}\n"
                    f"Evidence Relations: {evidence_relations}"
                    f"Evidence Nodes: {evidence_nodes}\n"
                    f"Intent: {intent}\n"
                    f"Topic: {topic}\n"
                    "In your output, do not give any chain-of-thought where you go over the points separately." \
                    "Simply output the one word 'Yes' if all of them are met, or the specific revision suggestions for each sentence," \
                    "where you explicitly say the words that have to be changed in each sentence. " \
                    "Furthermore, do not just write the entire passage yourself"
                    
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": content},
                ]
                judge_result = judge_model.invoke(messages)

                judge_raw = judge_result.content.strip()

                print(f"--- Judge Result for Topic {idx} ---")
                print(judge_raw)
                print()

                judge_text = judge_raw.lower()

                import re

                if re.search(r"\byes\b", judge_text):
                     break

                revision_advice = judge_raw

                revise_content = (
                    f"Passage: {corpus}\n"
                    f"Advise: {revision_advice}\n"
                    "Revise the Passage according to the Advise. "
                    "Do not include any explanations, reasoning, or notes in your response. "
                    "Include ONLY the revised passage in your response."
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_REVISION},
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


def main():
    run_coe_agent()

if __name__ == "__main__":    main()