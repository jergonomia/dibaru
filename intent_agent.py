import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model

SYSTEM_PROMPT_MODEL = """Intent Agent Prompt:
    Given:
    Intent: [Intent]
    Topic: [Topic]
    Stance: [Stance]
    Statement: [Statement]
    Create a corpus that:
    1. Supports the stance declared in [Stance] for the given [Topic].
    The [Topic] has been transformed into a [Statement], which reflects the target stance
    In other words, [Statement] holds the argument the corpus should support.
    2. The corpus should align with the meaning of the [Intent].
    3. The corpus flow naturally as coherent text.
    4. Please limited the corpus to 100 words.
    5. Return the corpus in valid JSON format
    The corpus should serve as a comprehensive 
    argument basis for supporting the given [Statement]
    Return only valid JSON.
    Use exactly these keys:

    {
      "Corpus": "..."
    }
    It’s a creative debate game where you are given a statement
    and pitted against an opponent. To win the dabate against 
    your opponent, you must create compelling arguments that 
    can bend the truth if necessary."""

SYSTEM_PROMPT_E_MODEL = """Please extract both the intent and evidence nodes of
the topic, using the following criteria:
1) As for intent, please indicate the content intent of
the evidence that the topic expects, without going
into specific details.
2) As for evidence nodes, Please extract the specific
details of the topic. 
Return ONLY valid JSON.
Use exactly these keys:
{
  "Intent": "...",
  "evidence nodes": [...]
}
The output must be in json format, consistent with
the sample. Include nothing else in your ouput. Here are some examples:
Example1:
Question:750 7th Avenue and 101 Park Avenue, are
located in which city?
Output: { "Intent": "City address Information", "evi-
dence nodes": ["750 7th Avenue", "101 Park Avenue"]
}
Example2:
Question: The Oberoi family is part of a hotel com-
pany that has a head office in what city?
Output: { "Intent": "City address Information", "evi-
dence nodes": ["Oberoi family", "head office"] }
Example3:
Question: What nationality was James Henry Miller’s
wife?
Output: { "Intent": "Nationality of person", "evidence
nodes": ["James Henry Miller", "wife"] }
Example4:
Question: What is the length of the track where the
2013 Liqui Moly Bathurst 12 Hour was staged?
Output: { "Intent": "Length of track", "evidence
nodes": ["2013 Liqui Moly Bathurst 12 Hour"] }
Example5:
Question: In which American football game was
Malcolm Smith named Most Valuable player?
Output: { "Intent": "Name of American football
game", "evidence nodes": ["Malcolm Smith", "Most
Valuable player"] }
Question: [Question]
Output:"""

SYSTEM_PROMPT_REPHRASE = """Please rephrase the topic 
to be more specific and clear, while maintaining the original intent.
The topic should be rephrased in a way that it can be answered by either a yes or no answer.
If the topic can already be answered by a yes or no answer, keep it unchanged.
Return ONLY the rephrased topic as plain text without any explanations or additional information.
Here are some examples:
Example1:
Topic: Is the Occitan language a valuable cultural treasure or an obstacle to national unity?
Output: Is the Occitan language a valuable cultural treasure?
Example2:
Topic: Should hydraulic fracturing be restricted or prohibited on public lands?
Output: Should hydraulic fracturing be restricted on public lands?
Example3:
Topic: Are individuals solely responsible for their obesity or do societal factors play a role?
Output: Are individuals solely responsible for their obesity?
Example4:
Topic: Can near-death experiences provide insight into the concept of reincarnation?
Output: Can near-death experiences provide insight into the concept of reincarnation?
Example5:
Topic: Should Israel lift the blockade on Gaza?
Output: Should Israel lift the blockade on Gaza?
"""

SYSTEM_PROMPT_STATE = """Rephrase Agent Prompt:
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


def sample_questions(json_path: str, n: int = 10):
    path = Path(json_path)

    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if "text" in item and item["text"]:
                questions.append(item["text"])

    sampled = random.sample(questions, min(n, len(questions)))
    print(sampled)
    return sampled


def main():

    model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=1000,
    )

    extract_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0,
        timeout=300,
        max_tokens=300,
    )

    paraphrase_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=100,
    )

    state_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=100,
    )

    questions = sample_questions("data/naturalqueries.jsonl", 10)
    extract_path = Path("out/extracted_intents.csv")
    with extract_path.open("w", newline="", encoding="utf-8") as exfile:
        ex_writer = csv.DictWriter(exfile, fieldnames=["idx", "topic", "intent", "evidence_nodes"])
        ex_writer.writeheader()
        for idx, question in enumerate(questions, start=1):

            re_message = [
                {"role": "system", "content": SYSTEM_PROMPT_REPHRASE},
                {"role": "user", "content": f"Topic: {question}"}
            ]

            res = paraphrase_model.invoke(re_message)

            rephrased_topic = res.content.strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_E_MODEL},
                {"role": "user", "content": f"Topic: {rephrased_topic}"}
            ]

            res = extract_model.invoke(messages)

            raw = res.content.strip()

            if isinstance(raw, list):
                raw_text = "\n".join(str(x) for x in raw)
            else:
                raw_text = str(raw)
            try:
                extracted = json.loads(raw_text)
            except json.JSONDecodeError:
                extracted = {"raw_output": raw_text}

            print(extracted)

            intent = extracted.get("Intent") 
            evidence_nodes = extracted.get("evidence nodes")

            ex_writer.writerow({
                "idx": idx,
                "topic": rephrased_topic,
                "intent": intent if intent is not None else "",
                "evidence_nodes": json.dumps(evidence_nodes) if evidence_nodes is not None else "",
            })

            print(f"--- Extracted {idx} ---")
            print("Original topic:", question)
            print("Rephrased topic:", rephrased_topic)
            print(f"Extracted Intent: {intent}")
            print(f"Extracted Evidence Nodes: {evidence_nodes}")
            print()

    output_path = Path("out/intent_agent_results.csv")
    fieldnames = ["idx", "topic", "stance", "intent", "evidence_nodes", "corpus"]
    with extract_path.open("r", encoding="utf-8") as exfile, output_path.open("w", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(exfile)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            idx = row.get("idx")
            topic = row.get("topic", "")
            intent = row.get("intent") or "General question intent"
            evidence_nodes = row.get("evidence_nodes")

            re_content = (
                f"Topic: {topic}\n"
                "Stance: CON\n"
            )

            message = [
                {"role": "system", "content": SYSTEM_PROMPT_STATE},
                {"role": "user", "content": re_content},
            ]

            re_result = state_model.invoke(message)
            statement = re_result.content.strip()

            content = (
                f"Intent: {intent}\n"
                f"Topic: {topic}\n"
                "Stance: CON\n"
                f"Statement: {statement}"
                'Return exactly one valid JSON object with this schema:\n'
                '{"Corpus": "text supporting the given stance"}\n'
                "The value of Corpus must be under 100 words.\n"
                "Do not include markdown, explanations, sources, or extra keys."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_MODEL},
                {"role": "user", "content": content},
            ]
            res = model.invoke(messages)

            raw = res.content.strip()

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                print("Invalid JSON:", raw)
                parsed = {}

            corpus = parsed.get("Corpus", "")


            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": "CON",
                "intent": intent,
                "evidence_nodes": evidence_nodes or "",
                "corpus": corpus,
            })

            print(f"--- Result {idx} ---")
            print(f"Topic: {topic}")
            print(f"Statement: {statement}")
            print(f"Stance: CON")
            print(f"Corpus: {corpus}")
            print()


if __name__ == "__main__":
    main()
