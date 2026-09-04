"""Build one isolated set of poisoned documents for an experiment.

This module is deliberately a thin adapter around the existing BiasChain and
baseline agents.  It gives those agents explicit input/output paths and emits a
single, method-independent ``poisoned_docs.csv`` consumed by ``rag_runner.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pandas as pd


METHOD_ALIASES = {
    "auth": "biaschain",
    "biaschain": "biaschain",
    "biaschain_wiki": "biaschain",
    "biaschain_generation": "biaschain",
    "poisonedrag": "poisonedrag",
    "prompt_injection": "prompt_injection",
    "simple_prompt_injection": "simple_prompt_injection",
}


def load_agent_callable(agent_file: Path, function_name: str):
    """Load one agent entry point from an explicit Python file."""
    agent_file = agent_file.resolve()
    if not agent_file.exists() or not agent_file.is_file():
        raise FileNotFoundError(f"Agent file does not exist: {agent_file}")

    module_digest = hashlib.sha256(str(agent_file).encode("utf-8")).hexdigest()[:12]
    module_name = f"_biaschain_agent_{agent_file.stem}_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from: {agent_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(
            f"{agent_file} does not export callable {function_name}()"
        )
    return function


def accepts_keyword(function, keyword: str) -> bool:
    parameters = inspect.signature(function).parameters
    return keyword in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def run_selected_intent_agent(
    function,
    *,
    target_stance: str,
    n_questions: int,
    query_file: Path,
    output_dir: Path,
    seed: int,
    prepared_topics_path: Path | None = None,
) -> None:
    kwargs = {
        "query_path": query_file,
        "output_dir": output_dir,
        "seed": seed,
    }
    if prepared_topics_path is not None:
        if not accepts_keyword(function, "prepared_topics_path"):
            raise TypeError(
                "The selected intent agent does not accept prepared_topics_path. "
                "Use the normal seeded query_file mode or add that optional argument "
                "to the alternate run_intent_agent()."
            )
        kwargs["prepared_topics_path"] = prepared_topics_path

    function(target_stance, n_questions, **kwargs)


def run_selected_authority_agent(
    function,
    *,
    intent_path: Path,
    authority_path: Path,
    coe_path: Path,
) -> None:
    kwargs = {}
    if accepts_keyword(function, "coe_input_path"):
        kwargs["coe_input_path"] = coe_path
    function(intent_path, authority_path, **kwargs)


def read_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
    delimiter = "|" if "|" in header else "," if "," in header else "\t"
    return pd.read_csv(path, sep=delimiter, dtype=str).fillna("")


def normalize_intent_source(source: Path, destination: Path, target_stance: str) -> Path:
    """Normalize an external PoisonedRAG source to intent_agent's pipe CSV."""
    frame = read_table(source)
    if "topic" not in frame.columns:
        if "rephrased_query" in frame.columns:
            frame["topic"] = frame["rephrased_query"]
        elif "original_query" in frame.columns:
            frame["topic"] = frame["original_query"]
        else:
            raise ValueError(f"External source has no topic column: {source}")

    if "idx" not in frame.columns:
        if "query_idx" in frame.columns:
            frame["idx"] = frame["query_idx"]
        else:
            frame["idx"] = range(1, len(frame) + 1)

    if "stance" not in frame.columns:
        frame["stance"] = target_stance
    else:
        matching = frame["stance"].str.upper() == target_stance.upper()
        if not matching.any():
            raise ValueError(
                f"External PoisonedRAG source has no rows for target stance {target_stance}: {source}"
            )
        frame = frame[matching].copy()
    if "statement" not in frame.columns:
        frame["statement"] = ""

    for column in ["intent", "evidence_nodes", "corpus"]:
        if column not in frame.columns:
            frame[column] = ""

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[
        ["idx", "topic", "statement", "stance", "intent", "evidence_nodes", "corpus"]
    ].drop_duplicates(subset=["idx", "topic"]).to_csv(destination, sep="|", index=False)
    return destination


def normalize_topic_source(
    source: Path,
    destination: Path,
    n_questions: int,
) -> Path:
    """Extract only stable topic identifiers and topic text from a source CSV."""
    frame = read_table(source)
    if "topic" not in frame.columns:
        if "rephrased_query" in frame.columns:
            frame["topic"] = frame["rephrased_query"]
        elif "original_query" in frame.columns:
            frame["topic"] = frame["original_query"]
        else:
            raise ValueError(f"Topic source has no topic column: {source}")
    if "idx" not in frame.columns:
        if "query_idx" in frame.columns:
            frame["idx"] = frame["query_idx"]
        else:
            frame["idx"] = range(1, len(frame) + 1)

    frame = frame[frame["topic"].str.strip().ne("")]
    frame = frame.drop_duplicates(subset=["idx", "topic"], keep="first")
    if len(frame) < n_questions:
        raise ValueError(
            f"Topic source contains {len(frame)} usable topics, but "
            f"{n_questions} were requested: {source}"
        )
    frame = frame.iloc[:n_questions].copy()

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[["idx", "topic"]].to_csv(destination, sep="|", index=False)
    return destination


def compose_poisoned_docs(
    method: str,
    output_dir: Path,
    method_name: str | None = None,
) -> Path:
    method = METHOD_ALIASES.get(method, method)

    if method == "biaschain":
        authority = read_table(output_dir / "authority_content.csv")
        coe = read_table(output_dir / "CoE_content.csv")
        merged = authority.merge(coe, on=["idx", "topic", "stance"], how="inner")
        authority_column = "statement"
        if "statement_x" in merged.columns:
            authority_column = "statement_x"
        merged["poisoned_doc"] = (
            merged[authority_column].fillna("") + "\n" + merged["corpus"].fillna("")
        ).str.strip()
        result = merged
    elif method == "poisonedrag":
        result = read_table(output_dir / "PoisonedRAG_results.csv")
        result["poisoned_doc"] = (
            result["topic"].fillna("") + "\n" + result["poisoned"].fillna("")
        ).str.strip()
    elif method in {"prompt_injection", "simple_prompt_injection"}:
        filename = (
            "prompt_injection_results.csv"
            if method == "prompt_injection"
            else "simple_prompt_injection_results.csv"
        )
        result = read_table(output_dir / filename)
        result["poisoned_doc"] = (
            result["topic"].fillna("") + "\n" + result["corpus"].fillna("")
        ).str.strip()
    else:
        raise ValueError(f"Unsupported poisoned document method: {method}")

    if "stance" not in result.columns or "topic" not in result.columns:
        raise ValueError("Poisoned document output must contain topic and stance columns")

    result = result[result["topic"].str.strip().ne("")]
    result = result[result["poisoned_doc"].str.strip().ne("")]
    if result.empty:
        raise ValueError("Document generation produced no usable poisoned documents")

    result = result.drop_duplicates(subset=["topic"], keep="first")
    output_path = output_dir / "poisoned_docs.csv"
    result.to_csv(output_path, sep="|", index=False)

    topics = result["topic"].tolist()
    topic_hash = hashlib.sha256(
        json.dumps(topics, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "method": method_name or method,
        "method_family": method,
        "document_count": len(result),
        "topics_sha256": topic_hash,
        "topics": topics,
    }
    (output_dir / "documents_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def build_documents(args: argparse.Namespace) -> Path:
    method = args.method_family or METHOD_ALIASES.get(args.method)
    if method is None:
        raise ValueError(
            f"Unknown method {args.method!r}; pass --method-family for a configured variant"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_stance = args.target_stance.upper()
    intent_path = output_dir / "intent_agent_results.csv"
    topic_path = output_dir / "topic_manifest.csv"
    script_dir = Path(__file__).resolve().parent
    intent_agent_file = (
        args.intent_agent_file.resolve()
        if args.intent_agent_file
        else script_dir / "intent_agent.py"
    )
    authority_agent_file = (
        args.authority_agent_file.resolve()
        if args.authority_agent_file
        else script_dir / "authority_agent.py"
    )
    run_intent_agent = load_agent_callable(intent_agent_file, "run_intent_agent")

    if args.topic_source:
        normalize_topic_source(
            args.topic_source.resolve(),
            topic_path,
            args.n_questions,
        )
        if method == "biaschain":
            run_selected_intent_agent(
                run_intent_agent,
                target_stance=target_stance,
                n_questions=args.n_questions,
                query_file=args.query_file.resolve(),
                output_dir=output_dir,
                seed=args.seed,
                prepared_topics_path=topic_path,
            )
        elif method == "poisonedrag":
            normalize_intent_source(topic_path, intent_path, target_stance)
    elif args.source_file and method == "poisonedrag":
        normalize_intent_source(args.source_file.resolve(), intent_path, target_stance)
    elif method not in {"prompt_injection", "simple_prompt_injection"} or not args.source_file:
        run_selected_intent_agent(
            run_intent_agent,
            target_stance=target_stance,
            n_questions=args.n_questions,
            query_file=args.query_file.resolve(),
            output_dir=output_dir,
            seed=args.seed,
        )

    if method == "biaschain":
        from CoEagent import run_coe_agent

        coe_path = output_dir / "CoE_content.csv"
        authority_path = output_dir / "authority_content.csv"
        run_coe_agent(intent_path, coe_path)
        run_authority_agent = load_agent_callable(
            authority_agent_file,
            "run_authority_agent",
        )
        run_selected_authority_agent(
            run_authority_agent,
            intent_path=intent_path,
            authority_path=authority_path,
            coe_path=coe_path,
        )
    elif method == "poisonedrag":
        from PoisonedRAG import run_agent

        run_agent(
            target_stance,
            args.poisonedrag_iterations,
            input_path=intent_path,
            output_path=output_dir / "PoisonedRAG_results.csv",
        )
    elif method == "prompt_injection":
        from prompt_injection import run_prompt_injection_baseline

        source = (
            topic_path
            if args.topic_source
            else args.source_file.resolve() if args.source_file else intent_path
        )
        run_prompt_injection_baseline(
            target_stance,
            source_path=str(source),
            output_path=output_dir / "prompt_injection_results.csv",
        )
    elif method == "simple_prompt_injection":
        from simple_prompt_injection import run_simple_prompt_injection

        source = (
            topic_path
            if args.topic_source
            else args.source_file.resolve() if args.source_file else intent_path
        )
        run_simple_prompt_injection(
            target_stance,
            source_path=str(source),
            output_path=output_dir / "simple_prompt_injection_results.csv",
        )
    else:
        raise ValueError(f"Unsupported poisoned document method: {method}")

    return compose_poisoned_docs(method, output_dir, method_name=args.method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--method-family",
        choices=sorted(set(METHOD_ALIASES.values())),
        help="Canonical document family used by a configured method variant",
    )
    parser.add_argument("--intent-agent-file", type=Path)
    parser.add_argument("--authority-agent-file", type=Path)
    parser.add_argument("--target-stance", choices=["PRO", "CON"], required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topic-source",
        type=Path,
        help="Existing CSV whose idx/topic columns define the shared topic set",
    )
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--poisonedrag-iterations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.query_file.exists() and not args.source_file and not args.topic_source:
        raise FileNotFoundError(f"Query file does not exist: {args.query_file}")
    if args.topic_source and not args.topic_source.exists():
        raise FileNotFoundError(f"Topic source file does not exist: {args.topic_source}")
    if args.source_file and not args.source_file.exists():
        raise FileNotFoundError(f"Baseline source file does not exist: {args.source_file}")
    output = build_documents(args)
    print(f"Saved poisoned documents to: {output}")


if __name__ == "__main__":
    main()
