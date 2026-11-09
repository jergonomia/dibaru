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


SYSTEM_PROMPT_DUMMYRAG_STRICT = """You are a helpful assistant answering user questions based on a retrieved passage.
You are given:
Topic: {topic}
Passage: {passage}

Your task is to answer the topic using ONLY the information in the passage. 
Do not include any information from your own knowledge or opinions, 
only use the passage to answer the topic as neutrally and factually as possible.

If you can answer the topic using the passage, provide a score of 1 along with your answer.
Otherwise, provide a score of 0 along with your answer.

Return ONLY valid JSON with exactly these fields:
{
  "score": 0,
  "explanation": "Brief explanation of the passage's actual contribution or mismatch."
}
"""

SYSTEM_PROMPT_1 = """You assess passages for a retrieval-augmented system
answering open-ended debate questions.

Given a query and a passage, evaluate how useful the passage would be
for answering the PARTICULAR question asked.

Relevance to the exact question is the primary criterion. The quality
and substance of the reasoning or evidence determine how strongly a
relevant passage contributes.

Scores:
0: Unrelated.
The passage does not contribute to answering the query. Shared words
with different meanings do not establish relevance.

1: Related background or unsupported position.
The passage concerns the same broad topic, or contains similar words, 
but does not directly address the SPECIFIC question. It may answer a different question, provide
general background, or state a position without useful explanation.

2: Useful but limited contribution.
The passage addresses an aspect of the specific question through a
relevant reason, explanation, or factual claim, but its contribution
is brief, underdeveloped, or insufficiently specific.

3: Direct and substantive contribution.
The passage directly addresses an important aspect of the specific
question with developed reasoning or concrete, relevant evidence.
It can support, oppose, or qualify a position. It need not answer
the entire question or present both sides.

Rules:
- Identify what information the query requests before scoring.
- Judge the passage against that request, not just the general topic.
- Do not invent missing connections to make a passage useful.
  Ordinary paraphrases and clear implications are allowed.
- A useful factual finding does not need an explicit opinion.
- Do not favor a passage because you agree with its stance.
- Fluency, emotional force, length, famous speakers, and citation
  markers do not by themselves establish relevance or quality.
- Citations are neither required nor sufficient for a score of 3.
  Do not assume a cited source has been verified.
- Base the explanation on the contribution actually present.
- Assess each passage independently. Do not force a score distribution.

Examples:

Query: Does binge-watching increase viewers' enjoyment?
Passage: A study of binge drinking measured alcohol consumption among
university students.
Output:
{"score": 0, "explanation": "The passage concerns alcohol consumption, not watching television. The shared word 'binge' does not establish relevance."}

Query: Should one pursue higher education?
Passage: Research found that increased homework was associated with
better grades and a higher probability of college attendance among
high school boys.
Output:
{"score": 1, "explanation": "The passage addresses factors associated with entering college, not the benefits or costs of pursuing higher education."}

Query: Does homework affect college attendance?
Passage: Research found that increased homework was associated with
better grades and a higher probability of college attendance among
high school boys.
Output:
{"score": 3, "explanation": "The passage reports a research finding directly connecting homework with college attendance and specifies the population studied."}

Query: Is milk a good dietary choice?
Passage: Milk contains calcium and protein.
Output:
{"score": 2, "explanation": "The passage provides relevant nutritional information, but does not explain its significance or discuss conditions affecting whether milk is a suitable choice."}

Query: Should one pursue higher education?
Passage: College can widen access to occupations requiring a degree.
However, its financial value depends on tuition, the likelihood of
graduating, and earnings available through alternative routes.
Output:
{"score": 3, "explanation": "The passage explains an employment benefit and identifies concrete conditions affecting the financial case for attending college."}

Return ONLY valid JSON with exactly these fields:
{
  "score": 0,
  "explanation": "Brief explanation of the passage's actual contribution or mismatch."
}
"""


SYSTEM_PROMPT_2 = """You are a helpful assistant that evaluates passages for a retrieval-augmented 
system answering open-ended debate questions.

Given a query and a passage, evaluate how useful the passage would be
for answering the PARTICULAR question asked.

You are to give a numeric score between 0 and 2. Approach the problem in 
the following way:

Answer the particular query first ONLY using the passage as the evidence for your answer,
not your own knowledge or opinions on the matter. 
Then, provide a numeric score between 0 and 2 based on the following criteria:

0: Unrelated. The query is not answerable using the passage. 
The passage does not contribute to answering the query. 
Shared words with different meanings do not establish relevance.

1: Related background or unsupported position. The passage provides some background information 
or context related to the query, but does not directly answer it. 
Using the passage, you would have to provide additional information or 
the ultimate argument to answer the query from any stance.

2: Direct and substantive contribution. The passage directly addresses the query with developed reasoning or concrete, relevant evidence. 
It can support, oppose, or qualify a position, but it ALONE answers the query without the need of additional information or reasoning.

To assist you in your evaluation, you are given a deconstructed claim from the passage made by a separate model. The claim/claims are provided below:

Return ONLY valid JSON with exactly these fields:
{
  "score": 0,
  "explanation": "Brief explanation of the passage's actual contribution or mismatch."
}

"""

SYSTEM_PROMPT_CLAIM_ID = """You are a helpful assistant that assists in evaluation for a retrieval-augmented system answering open-ended debate questions.
Your task is to identify the central claim/claims from a given passsage. Make the claim(s) as concise as possible, condense it into a single sentence.

Return ONLY valid JSON with exactly these fields:
{
  "claim": "The central claim(s) of the passage. If there are multiple claims, just concatenate them into a single string here."
}

"""

model = init_chat_model(
    "ollama:gemma4:latest",
    temperature=0.1,
    timeout=300,
    max_tokens=1024,
)

claim_model = init_chat_model(
    "ollama:gemma4:latest",
    temperature=0.1,
    timeout=300,
    max_tokens=512,
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


def parse_claim_response(raw_response):
    """Parse the claim extractor JSON output from the LLM."""
    text = str(raw_response).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "claim" in payload:
            return {"claim": str(payload["claim"]).strip()}
    except json.JSONDecodeError:
        pass

    match = re.search(r'"claim"\s*:\s*"(.*)"', text, re.DOTALL)
    if match:
        claim = match.group(1).encode("utf-8").decode("unicode_escape")
        return {"claim": claim}

    return {"claim": ""}


def judge_topic_passage_pair(topic: str, passage: str, claim_model_instance=None, judge_model_instance=None):
    """Judge one topic/passage pair using a two-stage LLM pipeline:
    1) extract the central claim from the passage
    2) evaluate the topic+passage+claim under SYSTEM_PROMPT_2
    """
    if claim_model_instance is None:
        claim_model_instance = claim_model
    if judge_model_instance is None:
        judge_model_instance = model

    claim_response = claim_model_instance.invoke([
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_ID},
        {"role": "user", "content": f"Topic: {topic}\nPassage: {passage}"},
    ])
    claim_payload = parse_claim_response(getattr(claim_response, "content", claim_response))
    claim = claim_payload.get("claim", "").strip()

    judge_message = (
        f"Topic: {topic}\n"
        f"Passage: {passage}\n"
        f"Claim: {claim}\n"
    )

    judge_response = judge_model_instance.invoke([
        {"role": "system", "content": SYSTEM_PROMPT_2},
        {"role": "user", "content": judge_message},
    ])

    score_payload = parse_judge_response(getattr(judge_response, "content", judge_response))
    score_payload["claim"] = claim
    return score_payload


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


def call_judge_with_prompt(system_prompt: str, topic: str, passage: str, claim: str | None = None):
    """Invoke one judge prompt for a topic/passage pair and parse the JSON result."""
    user_message = f"Topic: {topic}\nPassage: {passage}"
    if claim:
        user_message += f"\nClaim: {claim}"

    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ])

    payload = parse_judge_response(getattr(response, "content", response))
    if claim:
        payload["claim"] = claim
    payload["raw_response"] = getattr(response, "content", str(response))
    return payload


def extract_claim(topic: str, passage: str):
    """Generate a concise central claim for the given passage."""
    response = claim_model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT_CLAIM_ID},
        {"role": "user", "content": f"Topic: {topic}\nPassage: {passage}"},
    ])
    payload = parse_claim_response(getattr(response, "content", response))
    return payload.get("claim", "").strip()


def run_retrieval_sampling(
    query_path: str | Path | None = None,
    n_queries: int = 10,
    k: int = 5,
    seed: int | None = None,
    output_path: str | Path | None = None,
    db_dir: str | Path | None = None,
):
    """Sample queries, retrieve passages, evaluate them under all judge prompts, and save results to separate files."""
    if query_path is None:
        query_path = Path(__file__).resolve().parent / "data" / "procon_queries.jsonl"
    else:
        query_path = Path(query_path)

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "out" / "retrieval_sample_results.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_map = {
        "dummyrag_strict": SYSTEM_PROMPT_DUMMYRAG_STRICT,
        "prompt_1": SYSTEM_PROMPT_1,
        "prompt_2": SYSTEM_PROMPT_2,
    }

    rows_by_prompt = {name: [] for name in prompt_map}
    print(f"Loading ProCon queries from: {query_path}")
    sampled_queries = sample_questions(query_path, n=n_queries, seed=seed)
    print(f"Sampled {len(sampled_queries)} queries for retrieval and reranking.")

    vector_store = build_vector_store(db_dir=db_dir)
    db_location = select_procon_db_dir(db_dir) if db_dir is not None else select_procon_db_dir()
    print(f"Connected to Chroma DB at: {db_location}")

    total_pairs = 0
    for query in sampled_queries:
        _, retrieved_docs = retrieve_from_db(query, vector_store=vector_store, k=k)
        total_pairs += len(retrieved_docs)

    print(f"Total query-passage pairs to judge: {total_pairs}")

    for query_idx, query in enumerate(tqdm(sampled_queries, desc="Retrieving queries", unit="query"), start=1):
        print(f"\n[{query_idx}/{len(sampled_queries)}] Query: {query}")
        serialized_context, retrieved_docs = retrieve_from_db(query, vector_store=vector_store, k=k)
        print(f"Retrieved {len(retrieved_docs)} passages for this query.")

        for passage_idx, doc in enumerate(tqdm(retrieved_docs, desc=f"Judging passages for query {query_idx}", leave=False, unit="passage"), start=1):
            passage_text = doc.page_content
            source = doc.metadata.get("source", "unknown")
            claim = extract_claim(query, passage_text)

            print(f"  Passage {passage_idx}/{len(retrieved_docs)} | Source: {source}")
            print(f"    Claim: {claim[:120]}...")

            for prompt_name, prompt in prompt_map.items():
                judge_payload = call_judge_with_prompt(
                    prompt,
                    topic=query,
                    passage=passage_text,
                    claim=claim if prompt_name == "prompt_2" else None,
                )

                row = {
                    "query_idx": query_idx,
                    "query": query,
                    "passage_idx": passage_idx,
                    "source": source,
                    "score": judge_payload.get("score", 0),
                    "explanation": judge_payload.get("explanation", ""),
                    "claim": judge_payload.get("claim", claim if prompt_name == "prompt_2" else ""),
                    "passage_content": passage_text,
                    "retrieved_context": serialized_context,
                    "judge_prompt": prompt_name,
                    "raw_response": judge_payload.get("raw_response", ""),
                }
                rows_by_prompt[prompt_name].append(row)

                if prompt_name == "prompt_2":
                    print(f"    [{prompt_name}] Score: {judge_payload.get('score',0)} | Explanation: {judge_payload.get('explanation','')[:120]}...")
                else:
                    print(f"    [{prompt_name}] Score: {judge_payload.get('score',0)} | Explanation: {judge_payload.get('explanation','')[:120]}...")

    for prompt_name, prompt_rows in rows_by_prompt.items():
        base_name = output_path.stem
        csv_path = output_path.with_name(f"{base_name}_{prompt_name}.csv")
        jsonl_path = output_path.with_name(f"{base_name}_{prompt_name}.jsonl")

        fieldnames = [
            "query_idx",
            "query",
            "passage_idx",
            "source",
            "claim",
            "score",
            "explanation",
            "passage_content",
            "retrieved_context",
            "judge_prompt",
            "raw_response",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(prompt_rows)

        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in prompt_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved {len(prompt_rows)} {prompt_name} judgments to: {csv_path}")
        print(f"Saved raw {prompt_name} judgments to: {jsonl_path}")

    return rows_by_prompt


def main():
    run_retrieval_sampling(n_queries=40, k=20, seed=420)


if __name__ == "__main__":
    main()
