import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm


SYSTEM_PROMPT = """You are acting as a Judge/reranker for a debate. You are given a debate topic and a related passage of text.
Your task is to evaluate the passage based on its relevance to the topic and its quality in terms of argumentation,
clarity, and persuasiveness. You are then to provide a score between 0 and 3, each with the following descriptions:
0: The passage has no relevance to the related topic
1: The passage covers words from the topic, but in a way that is not relevant to the topic from any perspective,
or it gives an answer without providing any arguments.
2: The passage is meaningfully relevant to the topic.
3: The passage is ideally relevant to the topic and provides strong arguments and perspectives on it, 
along with supporting citations.

Return ONLY valid JSON in this exact schema:
{
  "score": 0,
  "explanation": "brief explanation"
}

Do not include markdown fences, commentary, or any extra text.
"""


model = init_chat_model(
    "ollama:gemma4:latest",
    temperature=0.1,
    timeout=300,
    max_tokens=1024,
)


def parse_judge_response(raw_response):
    """Parse JSON from the LLM judge or fall back to a lightweight regex parser."""
    text = str(raw_response).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "score" in payload:
            return {
                "score": int(payload["score"]),
                "explanation": str(payload.get("explanation", "")).strip(),
            }
    except json.JSONDecodeError:
        pass

    match = re.search(r'"score"\s*:\s*(\d+)', text)
    if match:
        score = int(match.group(1))
        explanation = ""
        explanation_match = re.search(r'"explanation"\s*:\s*"(.*)"', text, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).encode("utf-8").decode("unicode_escape")
        return {"score": score, "explanation": explanation}

    match = re.search(r"\b([0-3])\b", text)
    if match:
        score = int(match.group(1))
        return {"score": score, "explanation": text}

    return {"score": 0, "explanation": "Unable to parse LLM judge output."}


def judge_topic_passage_pair(topic: str, passage: str, judge_model=None):
    """Judge one topic/passage pair using the LLM and return a numeric score plus explanation."""
    if judge_model is None:
        judge_model = model

    user_message = (
        f"Topic: {topic}\n\n"
        f"Passage: {passage}"
    )

    response = judge_model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    result = parse_judge_response(getattr(response, "content", response))
    return result


def sample_questions(
    json_path: str | Path,
    n: int = 10,
    seed: int | None = None,
):
    """Sample a variable number of ProCon-style questions.

    This follows the same logic used in intent_agent.py: for ProCon data, sample one
    query per topic cluster instead of sampling uniformly from all questions.
    """
    path = Path(json_path)
    rng = random.Random(seed)

    if "procon" in path.stem.lower():
        procon_questions = defaultdict(list)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                if item.get("text"):
                    topic_id = item["_id"].split("_")[0]
                    procon_questions[topic_id].append(item["text"])

        questions = [
            rng.choice(topic_questions)
            for topic_questions in procon_questions.values()
        ]
    else:
        questions = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                if item.get("text"):
                    questions.append(item["text"])

    sampled = rng.sample(questions, min(n, len(questions)))
    return sampled


def select_procon_db_dir(base_dir: str | Path | None = None) -> Path:
    """Choose the existing Chroma DB directory for ProCon data."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    base_dir = Path(base_dir)
    candidates = [
        "chroma_db_qwen3_embedding_4b_procon",
        "chroma_db_nomic_procon",
    ]

    for candidate in candidates:
        db_dir = base_dir / candidate
        if db_dir.exists():
            return db_dir

    raise FileNotFoundError(
        "No existing ProCon Chroma DB found in the project. "
        "Expected one of: " + ", ".join(candidates)
    )


def build_vector_store(db_dir: str | Path | None = None):
    """Build the same retrieval-only vector store setup used in rag_procon.py."""
    if db_dir is None:
        db_dir = select_procon_db_dir()

    db_dir = Path(db_dir)

    embedder_name = "qwen"
    if "nomic" in db_dir.name.lower():
        embedder_name = "nomic"

    if embedder_name == "nomic":
        embedding_model = "nomic-embed-text"
    else:
        embedding_model = "qwen3-embedding:4b"

    embedding = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name="procon",
        embedding_function=embedding,
        persist_directory=str(db_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vector_store


def retrieve_from_db(
    query: str,
    vector_store=None,
    k: int = 5,
):
    """Run similarity search against the existing Chroma database."""
    if vector_store is None:
        vector_store = build_vector_store()

    retrieved_docs = vector_store.similarity_search(query, k=k)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_retrieval_sampling(
    query_path: str | Path | None = None,
    n_queries: int = 10,
    k: int = 5,
    seed: int | None = None,
    output_path: str | Path | None = None,
    db_dir: str | Path | None = None,
):
    """Sample queries, retrieve documents for each, score each retrieved passage, and save results."""
    if query_path is None:
        query_path = Path(__file__).resolve().parent / "data" / "procon_queries.jsonl"
    else:
        query_path = Path(query_path)

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "out" / "retrieval_sample_results.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    judgment_output_path = output_path.with_name(output_path.stem + "_judgments.jsonl")

    print(f"Loading ProCon queries from: {query_path}")
    sampled_queries = sample_questions(query_path, n=n_queries, seed=seed)
    print(f"Sampled {len(sampled_queries)} queries for retrieval and reranking.")

    vector_store = build_vector_store(db_dir=db_dir)
    print(f"Connected to Chroma DB at: {select_procon_db_dir(db_dir) if db_dir is not None else select_procon_db_dir()}")

    rows = []
    judgment_rows = []

    for query_idx, query in enumerate(tqdm(sampled_queries, desc="Retrieving and judging queries", unit="query"), start=1):
        print(f"\n[{query_idx}/{len(sampled_queries)}] Query: {query}")

        serialized_context, retrieved_docs = retrieve_from_db(query, vector_store=vector_store, k=k)
        print(f"Retrieved {len(retrieved_docs)} passages for this query.")

        for passage_idx, doc in enumerate(tqdm(retrieved_docs, desc=f"Judging passages for query {query_idx}", leave=False, unit="passage"), start=1):
            passage_text = doc.page_content
            source = doc.metadata.get("source", "unknown")

            print(f"  Passage {passage_idx}/{len(retrieved_docs)} | Source: {source}")
            judgment = judge_topic_passage_pair(query, passage_text)
            print(f"    LLM score: {judgment['score']} | Explanation: {judgment['explanation'][:140]}...")

            row = {
                "query_idx": query_idx,
                "query": query,
                "passage_idx": passage_idx,
                "source": source,
                "score": judgment["score"],
                "explanation": judgment["explanation"],
                "passage_content": passage_text,
                "retrieved_context": serialized_context,
            }
            rows.append(row)

            judgment_rows.append({
                "query_idx": query_idx,
                "query": query,
                "passage_idx": passage_idx,
                "source": source,
                "score": judgment["score"],
                "explanation": judgment["explanation"],
                "passage_content": passage_text,
            })

    fieldnames = [
        "query_idx",
        "query",
        "passage_idx",
        "source",
        "score",
        "explanation",
        "passage_content",
        "retrieved_context",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with judgment_output_path.open("w", encoding="utf-8") as f:
        for item in judgment_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved retrieval + rerank output to: {output_path}")
    print(f"Saved per-passage judgments to: {judgment_output_path}")
    return rows


def main():
    run_retrieval_sampling(n_queries=10, k=20, seed=42)


if __name__ == "__main__":
    main()
