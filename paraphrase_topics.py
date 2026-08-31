import csv
from pathlib import Path

from langchain.chat_models import init_chat_model

SYSTEM_PROMPT_PARAPHRASE = """As an expert copy-editor, please rewrite the following topic in your own voice while ensuring
that the final output contains the same information as the original topic, 
sort of modeling how different people might express the same idea using simple clean language.
Try to keep the topic to the exact same length. 
After paraphrasing, the topic should remain in a form that is answerable with a simple yes or no,
with both answers being equal in meaning to the original topic.
Please do not omit any crucial details or change the intent of the topic. Additionally,
please take care to provide any relevant information about public figures, organizations, or
other entities mentioned in the topic to avoid any potential misunderstandings or biases.

In your response, include only the paraphrased topic without any additional comments or chain-of-thought reasoning.
Only include one paraphrased version of the topic in your response.
ALSO DO NOT INCLUDE ANY BACKGROUND INFORMATION ON THE TOPICS. THAT IS NOT NECESSARY, 
THESE TOPICS WILL BE USED AS DEBATE TOPICS SO NO ADDITIONAL INFO IS REQUIRED.
"""


def paraphrase_topic(model, topic: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PARAPHRASE},
        {
            "role": "user",
            "content": f"Paraphrase this topic:\n\n{topic}",
        },
    ]
    response = model.invoke(messages)
    return str(response.content).strip()


def paraphrase_topics(
    input_path: str | Path = "intent_agent_results.csv",
    output_path: str | Path | None = None,
):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name("paraphrased_topics.csv")
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.7,
        timeout=300,
        max_tokens=300,
    )

    fieldnames = ["idx", "topic"] + [f"paraphrase_{i}" for i in range(1, 11)]

    with input_path.open("r", encoding="utf-8", newline="") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile, delimiter="|")
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()

        for row in reader:
            topic = (row.get("topic") or row.get("statement") or "").strip()
            if not topic:
                continue

            idx = row.get("idx", "")
            paraphrases = []
            for i in range(10):
                paraphrase = paraphrase_topic(model, topic)
                paraphrases.append(paraphrase)
                print(f"[{idx}] paraphrase_{i + 1}: {paraphrase}")

            row_data = {"idx": idx, "topic": topic}
            for i, paraphrase in enumerate(paraphrases, start=1):
                row_data[f"paraphrase_{i}"] = paraphrase

            writer.writerow(row_data)


def main():
    input_path = Path("out/intent_agent_results.csv")
    paraphrase_topics(input_path=input_path)


if __name__ == "__main__":
    main()
