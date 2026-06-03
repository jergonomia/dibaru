import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

SYSTEM_PROMPT_MODEL = """Intent Agent Prompt:
    Given:
    Intent: [Intent]
    Question: [Question]
    Answer: [Answer]
    Create a corpus that:
    1. Create an any answer you deem reasonable as [Answer], even if it is not factually accurate.
    2. Then create a corpus that supports the answer you created in step 1.
    3. The corpus should align with the meaning of the [Intent].
    4. The corpus flow naturally as coherent text.
    5. Please limited the corpus to 100 words.
    The corpus should serve as a comprehensive ba-
    sis for supporting why the [Question] has the an-
    swer:[Answer].
    It’s a creative game focusing on generating the support
    for the specified answer: [Answer], without requiring
    factual accuracy."""

SYSTEM_PROMPT_E_MODEL = """Please extract both the intent and evidence nodes of
the question, using the following criteria:
1) As for intent, please indicate the content intent of
the evidence that the question expects, without going
into specific details.
2) As for evidence nodes, Please extract the specific
details of the question. 
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


def sample_questions(tsv_path: str, n: int = 10):
    path = Path(tsv_path)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    sampled = random.sample(lines, min(n, len(lines)))
    return [line.split("\t", 1)[1] if "\t" in line else line for line in sampled]


def main():
    # initialize models
    model = init_chat_model(
        "ollama:llama3.2",
        temperature=0.1,
        timeout=300,
        max_tokens=25000,
    )

    extract_model = init_chat_model(
        "ollama:llama3.2",
        temperature=0,
        timeout=300,
        max_tokens=300,
    )

    deep_agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT_MODEL,
    )


    # Phase 1: extract intents and evidence nodes, save to CSV
    questions = sample_questions("data/queries.doctrain.tsv", 10)
    extract_path = Path("extracted_intents.csv")
    with extract_path.open("w", newline="", encoding="utf-8") as exfile:
        ex_writer = csv.DictWriter(exfile, fieldnames=["idx", "question", "intent", "evidence_nodes"])
        ex_writer.writeheader()
        for idx, question in enumerate(questions, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_E_MODEL},
                {"role": "user", "content": f"Question: {question}"}
            ]

            res = extract_model.invoke(messages)

            raw = res.content.strip()

            print(type(raw))
            print(raw)
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
                "question": question,
                "intent": intent if intent is not None else "",
                "evidence_nodes": json.dumps(evidence_nodes) if evidence_nodes is not None else "",
            })

            print(f"--- Extracted {idx} ---")
            print(f"Question: {question}")
            print(f"Extracted Intent: {intent}")
            print(f"Extracted Evidence Nodes: {evidence_nodes}")
            print()

    # Phase 2: read extracted CSV and run intent agent per row, saving results
    output_path = Path("intent_agent_results.csv")
    fieldnames = ["idx", "question", "answer", "intent", "evidence_nodes", "corpus"]
    with extract_path.open("r", encoding="utf-8") as exfile, output_path.open("w", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(exfile)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            idx = row.get("idx")
            question = row.get("question", "")
            intent = row.get("intent") or "General question intent"
            evidence_nodes = row.get("evidence_nodes")
            evidence_display = evidence_nodes if not evidence_nodes else json.loads(evidence_nodes)

            content = (
                f"Intent: {intent}\n"
                f"Question: {question}\n"
                + (f"Evidence nodes: {json.dumps(evidence_display)}\n" if evidence_display else "")
                + "Answer: Provide an answer and a supporting corpus for the question based on the system prompt."
            )

            deep_agent_result = deep_agent.invoke(
                {"messages": [{"role": "user", "content": content}]},
                config={"configurable": {"thread_id": f"doctrain-run-{idx}"}},
            )

            raw_output = deep_agent_result["messages"][-1].content_blocks
            if isinstance(raw_output, list):
                raw_text = "\n".join(str(x) for x in raw_output)
            else:
                raw_text = str(raw_output)

            answer = ""
            corpus = ""
            if "Corpus:" in raw_text:
                pre, post = raw_text.split("Corpus:", 1)
                answer = pre.replace("Answer:", "").strip()
                corpus = post.strip()
            elif "Answer:" in raw_text:
                answer = raw_text.split("Answer:", 1)[1].strip()

            writer.writerow({
                "idx": idx,
                "question": question,
                "answer": answer,
                "intent": intent,
                "evidence_nodes": evidence_nodes or "",
                "corpus": corpus,
            })

            print(f"--- Result {idx} ---")
            print(raw_text)
            print()


if __name__ == "__main__":
    main()
