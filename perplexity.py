"""End-to-end GPT-2 perplexity analysis for RAG poisoning documents.

The script:
1. Loads one or more labelled ``poisoned_docs.csv`` files.
2. Collects the ordered union of all distinct topic strings.
3. Retrieves one top-ranked clean document per topic from a clean Chroma DB.
4. Calculates GPT-2 perplexity for clean and poisoned documents.
5. Saves per-document scores, group averages, clean pairs, and a distribution plot.

Example:

    python perplexity_analysis.py \
      --clean-db chroma_db_nomic_natural \
      --collection-name wiki_balance \
      --attack 'BiasChain-PRO=runs/.../documents/poisoned_docs.csv' \
      --attack 'BiasChain-CON=runs/.../documents/poisoned_docs.csv' \
      --attack 'PoisonedRAG-PRO=runs/.../documents/poisoned_docs.csv' \
      --output-dir perplexity/wiki_nomic

Run the script separately for WIKI-BALANCE and ProCon.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AttackInput:
    label: str
    path: Path


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters="|,\t;").delimiter
    except csv.Error:
        return "|"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        dtype=str,
    ).fillna("")


def parse_attack(value: str) -> AttackInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--attack must have the form LABEL=/path/to/poisoned_docs.csv"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "--attack requires both a non-empty label and path"
        )
    return AttackInput(label=label, path=Path(raw_path).expanduser().resolve())


def load_attack_documents(specification: AttackInput) -> pd.DataFrame:
    frame = read_table(specification.path)
    required = {"topic", "poisoned_doc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{specification.path} is missing columns: {sorted(missing)}"
        )

    if "idx" not in frame.columns:
        frame["idx"] = range(1, len(frame) + 1)

    frame["topic"] = frame["topic"].str.strip()
    frame["poisoned_doc"] = frame["poisoned_doc"].str.strip()
    frame = frame[
        frame["topic"].ne("") & frame["poisoned_doc"].ne("")
    ].copy()
    frame = frame.drop_duplicates(subset=["topic"], keep="first")
    if frame.empty:
        raise ValueError(f"No usable poisoned documents in {specification.path}")

    frame["group"] = specification.label
    frame["source_file"] = str(specification.path)
    return frame[["idx", "topic", "poisoned_doc", "group", "source_file"]]


def collect_all_topics(attacks: list[pd.DataFrame]) -> list[str]:
    """Return every distinct topic string in first-seen order.

    Small wording variations are intentionally retained. Each variation gets
    its own top-ranked clean document, while every poisoned document remains in
    its requested method distribution.
    """
    topics = []
    seen = set()
    for frame in attacks:
        for topic in frame["topic"]:
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
    if not topics:
        raise ValueError("The attack files contain no usable topics")
    return topics


def build_clean_pairs(
    topics: list[str],
    clean_db: Path,
    collection_name: str,
    embedding_model: str,
) -> pd.DataFrame:
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as error:
        raise RuntimeError(
            "Clean retrieval requires langchain-chroma and langchain-ollama"
        ) from error

    if not clean_db.exists():
        raise FileNotFoundError(f"Clean Chroma database does not exist: {clean_db}")

    embedding = OllamaEmbeddings(model=embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(clean_db),
    )
    count = store._collection.count()
    if count <= 0:
        raise RuntimeError(f"Clean Chroma collection is empty: {clean_db}")
    print(f"Clean collection contains {count} documents")

    rows = []
    for position, topic in enumerate(topics, start=1):
        retrieved = store.similarity_search(topic, k=1)
        if not retrieved:
            raise RuntimeError(f"No clean document retrieved for topic: {topic}")
        document = retrieved[0]
        rows.append(
            {
                "idx": position,
                "topic": topic,
                "clean_doc": document.page_content,
                "clean_source": document.metadata.get("source", ""),
            }
        )
        print(f"Retrieved clean pair {position}/{len(topics)}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


class GPT2PerplexityScorer:
    def __init__(
        self,
        model_name: str,
        device_name: str,
        stride: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Perplexity scoring requires torch and transformers"
            ) from error

        self.torch = torch
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.maximum_length = int(
            getattr(self.model.config, "n_positions", self.tokenizer.model_max_length)
        )
        if stride <= 0 or stride > self.maximum_length:
            raise ValueError(
                f"Stride must be between 1 and {self.maximum_length}, received {stride}"
            )
        self.stride = stride
        print(
            f"Loaded {model_name} on {self.device}; "
            f"maximum context={self.maximum_length}, stride={self.stride}"
        )

    def score(self, text: str) -> tuple[float, float, int]:
        """Return raw PPL, log PPL, and predicted-token count for one document."""
        torch = self.torch
        text = str(text).strip()
        if not text:
            raise ValueError("Cannot calculate perplexity for empty text")

        encoded = self.tokenizer(text, return_tensors="pt")
        all_input_ids = encoded["input_ids"].to(self.device)
        sequence_length = all_input_ids.size(1)
        if sequence_length < 2:
            raise ValueError("Text must contain at least two GPT-2 tokens")

        total_negative_log_likelihood = 0.0
        total_predicted_tokens = 0
        previous_end = 0

        with torch.inference_mode():
            for begin in range(0, sequence_length, self.stride):
                end = min(begin + self.maximum_length, sequence_length)
                target_length = end - previous_end
                input_ids = all_input_ids[:, begin:end]
                target_ids = input_ids.clone()
                target_ids[:, :-target_length] = -100

                output = self.model(input_ids=input_ids, labels=target_ids)
                # Causal-LM loss predicts labels[1:] from input_ids[:-1].
                predicted_tokens = int((target_ids[:, 1:] != -100).sum().item())
                if predicted_tokens > 0:
                    total_negative_log_likelihood += (
                        float(output.loss.item()) * predicted_tokens
                    )
                    total_predicted_tokens += predicted_tokens

                previous_end = end
                if end == sequence_length:
                    break

        if total_predicted_tokens == 0:
            raise ValueError("No tokens were available for causal prediction")

        log_perplexity = total_negative_log_likelihood / total_predicted_tokens
        perplexity = math.exp(log_perplexity)
        return perplexity, log_perplexity, total_predicted_tokens


def score_documents(
    frame: pd.DataFrame,
    text_column: str,
    group: str | None,
    scorer: GPT2PerplexityScorer,
    progress_offset: int,
    progress_total: int,
) -> pd.DataFrame:
    rows = []
    for local_position, (_, row) in enumerate(frame.iterrows(), start=1):
        position = progress_offset + local_position
        label = group if group is not None else str(row["group"])
        ppl, log_ppl, token_count = scorer.score(row[text_column])
        rows.append(
            {
                "idx": row.get("idx", ""),
                "topic": row["topic"],
                "group": label,
                "perplexity": ppl,
                "log_perplexity": log_ppl,
                "token_count": token_count,
                "source_file": row.get("source_file", ""),
            }
        )
        print(
            f"Scored document {position}/{progress_total}: {label}",
            end="\r",
            flush=True,
        )
    return pd.DataFrame(rows)


def make_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("group", as_index=False)
        .agg(
            average_perplexity=("perplexity", "mean"),
            median_perplexity=("perplexity", "median"),
            standard_deviation_perplexity=("perplexity", "std"),
            average_log_perplexity=("log_perplexity", "mean"),
            median_log_perplexity=("log_perplexity", "median"),
            document_count=("perplexity", "size"),
            average_token_count=("token_count", "mean"),
        )
        .sort_values("group")
        .reset_index(drop=True)
    )


def plot_distributions(scores: pd.DataFrame, output_path: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Distribution plotting requires matplotlib and seaborn"
        ) from error

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.kdeplot(
        data=scores,
        x="log_perplexity",
        hue="group",
        common_norm=False,
        fill=False,
        linewidth=2,
        warn_singular=False,
        ax=axis,
    )
    axis.set_xlabel("Log perplexity (GPT-2)")
    axis.set_ylabel("Density")
    axis.set_title("Clean and poisoned document perplexity distributions")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack",
        action="append",
        type=parse_attack,
        required=True,
        help=(
            "Labelled poisoned-doc file as LABEL=PATH; repeat for every attack "
            "and stance. Reusing a label combines files into one distribution."
        ),
    )
    parser.add_argument("--clean-db", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride for documents longer than GPT-2's context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_frames = [load_attack_documents(item) for item in args.attack]
    topics = collect_all_topics(attack_frames)
    stale_exclusions = output_dir / "excluded_topics.csv"
    if stale_exclusions.exists():
        stale_exclusions.unlink()
    print(
        f"Loaded {len(attack_frames)} attack files with "
        f"{len(topics)} distinct topic strings; all poisoned documents retained"
    )

    clean_pairs = build_clean_pairs(
        topics=topics,
        clean_db=args.clean_db.expanduser().resolve(),
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )
    clean_pairs.to_csv(output_dir / "clean_topic_pairs.csv", sep="|", index=False)

    scorer = GPT2PerplexityScorer(
        model_name=args.model_name,
        device_name=args.device,
        stride=args.stride,
    )

    combined_attacks = pd.concat(attack_frames, ignore_index=True)
    total_documents = len(clean_pairs) + len(combined_attacks)
    clean_scores = score_documents(
        clean_pairs,
        text_column="clean_doc",
        group="Clean",
        scorer=scorer,
        progress_offset=0,
        progress_total=total_documents,
    )
    attack_scores = score_documents(
        combined_attacks,
        text_column="poisoned_doc",
        group=None,
        scorer=scorer,
        progress_offset=len(clean_pairs),
        progress_total=total_documents,
    )
    print()

    scores = pd.concat([clean_scores, attack_scores], ignore_index=True)
    scores.to_csv(output_dir / "perplexity_scores.csv", index=False)

    summary = make_summary(scores)
    summary.to_csv(output_dir / "perplexity_summary.csv", index=False)
    print("\nPerplexity summary:")
    print(summary.to_string(index=False))

    plot_distributions(scores, output_dir / "perplexity_distribution.pdf")
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
