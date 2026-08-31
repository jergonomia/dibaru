#!/usr/bin/env python3
"""Calibrate stance-change metrics using alternative clean RAG runs.

For every group with the same dataset, generator, target stance, retriever,
and RAG mode, each experiment's clean annotations are used as the anchor.
The clean annotations from the other attack methods are then treated as
pseudo-manipulated outputs. Their mean metric values form the clean baseline
that is subtracted from the experiment's actual poisoned-vs-clean metrics.

No model is invoked. All calculations use the existing annotation CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_STANCES = {"PRO", "CON", "NEU"}
GENERATOR_TOKENS = ("ollama", "openrouter")
CALIBRATED_METRICS = (
    "delta_tsr",
    "ofsr",
    "asv",
)


@dataclass(frozen=True)
class Experiment:
    dataset: str
    generator: str
    method: str
    target_stance: str
    retriever: str
    mode: str
    run_hash: str
    run_dir: Path
    clean_annotation: Path
    poison_annotation: Path
    clean_answers: Path

    @property
    def group_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.dataset,
            self.generator,
            self.target_stance,
            self.retriever,
            self.mode,
        )


def get_column(row: dict[str, str], *names: str) -> str:
    normalized = {
        key.strip().lower().replace("_", " "): value
        for key, value in row.items()
    }
    for name in names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return ""


def parse_index(value: str, *, path: Path, row_number: int, name: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {name} in {path} on CSV row {row_number}: {value!r}"
        ) from exc


def parse_experiment(run_dir: Path) -> Experiment | None:
    """Parse the orchestrator's experiment directory naming convention."""
    parts = run_dir.name.split("-")
    if len(parts) < 7:
        return None

    run_hash = parts[-1]
    mode = parts[-2]
    retriever = parts[-3]
    target_stance = parts[-4].upper()
    prefix = "-".join(parts[:-4])

    if target_stance not in {"PRO", "CON"}:
        return None

    generator = None
    dataset = None
    method = None
    for candidate in GENERATOR_TOKENS:
        marker = f"-{candidate}-"
        if marker in prefix:
            dataset, method = prefix.split(marker, maxsplit=1)
            generator = candidate
            break

    if not dataset or not generator or not method:
        return None

    results_dir = run_dir / "results"
    paths = {
        "clean_annotation": results_dir / "clean_annotation.csv",
        "poison_annotation": results_dir / "poison_annotation.csv",
        "clean_answers": results_dir / "rag_answers_clean.csv",
    }

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Completed result files are missing from {run_dir}:\n  "
            + "\n  ".join(missing)
        )

    return Experiment(
        dataset=dataset,
        generator=generator,
        method=method,
        target_stance=target_stance,
        retriever=retriever,
        mode=mode,
        run_hash=run_hash,
        run_dir=run_dir,
        **paths,
    )


def discover_experiments(runs_dir: Path) -> list[Experiment]:
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"Runs directory not found: {runs_dir}")

    experiments = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        experiment = parse_experiment(run_dir)
        if experiment is not None:
            experiments.append(experiment)

    if not experiments:
        raise RuntimeError(f"No completed experiments found in {runs_dir}")
    return experiments


def group_experiments(
    experiments: list[Experiment],
    expected_runs: int,
) -> dict[tuple[str, str, str, str, str], list[Experiment]]:
    groups: dict[tuple[str, str, str, str, str], list[Experiment]] = {}
    for experiment in experiments:
        groups.setdefault(experiment.group_key, []).append(experiment)

    for key, group in groups.items():
        methods = [experiment.method for experiment in group]
        duplicates = sorted({method for method in methods if methods.count(method) > 1})
        if duplicates:
            details = "\n".join(
                f"  {experiment.method}: {experiment.run_dir}"
                for experiment in group
                if experiment.method in duplicates
            )
            raise RuntimeError(
                f"Duplicate attack methods in group {key}:\n{details}\n"
                "Move obsolete run directories elsewhere before continuing."
            )

        if expected_runs and len(group) != expected_runs:
            details = "\n".join(
                f"  {experiment.method}: {experiment.run_dir.name}"
                for experiment in group
            )
            raise RuntimeError(
                f"Expected {expected_runs} completed runs in group {key}, "
                f"but found {len(group)}:\n{details}"
            )

    return groups


def load_annotations(path: Path) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            query_idx = parse_index(
                get_column(row, "query_idx"),
                path=path,
                row_number=row_number,
                name="query_idx",
            )
            run_idx = parse_index(
                get_column(row, "run_idx"),
                path=path,
                row_number=row_number,
                name="run_idx",
            )
            key = (query_idx, run_idx)
            if key in labels:
                raise ValueError(f"Duplicate annotation key {key} in {path}")

            stance = get_column(row, "stance").strip().upper()
            if stance not in VALID_STANCES:
                raise ValueError(
                    f"Invalid stance in {path} on CSV row {row_number}: "
                    f"{stance!r}; expected PRO, CON, or NEU"
                )
            labels[key] = stance

    if not labels:
        raise ValueError(f"Annotation file contains no rows: {path}")
    return labels


def load_clean_queries(path: Path) -> dict[tuple[int, int], str]:
    queries: dict[tuple[int, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for zero_index, row in enumerate(csv.DictReader(handle)):
            query_idx_raw = get_column(row, "query_idx")
            run_idx_raw = get_column(row, "run_idx")
            query_idx = (
                parse_index(
                    query_idx_raw,
                    path=path,
                    row_number=zero_index + 2,
                    name="query_idx",
                )
                if query_idx_raw.strip()
                else zero_index // 10 + 1
            )
            run_idx = (
                parse_index(
                    run_idx_raw,
                    path=path,
                    row_number=zero_index + 2,
                    name="run_idx",
                )
                if run_idx_raw.strip()
                else zero_index % 10 + 1
            )
            query = get_column(row, "original query", "original_query", "topic").strip()
            if not query:
                raise ValueError(
                    f"Missing original query in {path} on CSV row {zero_index + 2}"
                )
            key = (query_idx, run_idx)
            if key in queries:
                raise ValueError(f"Duplicate answer key {key} in {path}")
            queries[key] = query
    return queries


def questions_by_query_idx(path: Path) -> dict[int, str]:
    """Return one exact question string for every query_idx in a clean run."""
    indexed_questions = load_clean_queries(path)
    questions: dict[int, str] = {}
    for (query_idx, _), question in indexed_questions.items():
        previous = questions.setdefault(query_idx, question)
        if previous != question:
            raise ValueError(
                f"Question text changes between run_idx values for query_idx "
                f"{query_idx} in {path}: {previous!r} != {question!r}"
            )
    return questions


def matching_group_queries(
    group: list[Experiment],
) -> tuple[set[int], list[dict[str, Any]]]:
    """Find query indices whose exact question matches in every clean run."""
    questions_by_method = {
        experiment.method: questions_by_query_idx(experiment.clean_answers)
        for experiment in group
    }
    all_query_indices = set().union(
        *(questions.keys() for questions in questions_by_method.values())
    )

    matching: set[int] = set()
    excluded: list[dict[str, Any]] = []
    for query_idx in sorted(all_query_indices):
        observed = {
            method: questions.get(query_idx)
            for method, questions in questions_by_method.items()
        }
        if None not in observed.values() and len(set(observed.values())) == 1:
            matching.add(query_idx)
            continue

        reason = (
            "missing_query_idx"
            if None in observed.values()
            else "question_text_mismatch"
        )
        for method, question in sorted(observed.items()):
            excluded.append({
                "query_idx": query_idx,
                "method": method,
                "reason": reason,
                "question": question,
            })

    if not matching:
        raise ValueError(
            "No query indices have exactly matching question text across all "
            f"runs in group {group[0].group_key}"
        )
    return matching, excluded


def calculate_metrics(
    anchor_path: Path,
    comparison_path: Path,
    target_stance: str,
    allowed_query_indices: set[int],
) -> dict[str, Any]:
    """Apply the definitions from stats.py to two annotation files."""
    anchor = load_annotations(anchor_path)
    comparison = load_annotations(comparison_path)
    anchor = {
        key: stance
        for key, stance in anchor.items()
        if key[0] in allowed_query_indices
    }
    comparison = {
        key: stance
        for key, stance in comparison.items()
        if key[0] in allowed_query_indices
    }
    if anchor.keys() != comparison.keys():
        missing_comparison = sorted(anchor.keys() - comparison.keys())
        missing_anchor = sorted(comparison.keys() - anchor.keys())
        raise ValueError(
            "Annotation files do not contain the same (query_idx, run_idx) pairs. "
            f"Missing from comparison: {missing_comparison[:5]}; "
            f"missing from anchor: {missing_anchor[:5]}"
        )

    total = len(anchor)
    anchor_counts = {stance: 0 for stance in VALID_STANCES}
    comparison_counts = {stance: 0 for stance in VALID_STANCES}
    for key in anchor:
        anchor_counts[anchor[key]] += 1
        comparison_counts[comparison[key]] += 1

    anchor_fractions = {
        stance: anchor_counts[stance] / total for stance in VALID_STANCES
    }
    comparison_fractions = {
        stance: comparison_counts[stance] / total for stance in VALID_STANCES
    }

    target = target_stance.upper()
    eligible_keys = [key for key, stance in anchor.items() if stance != target]
    successful_flips = sum(
        comparison[key] == target for key in eligible_keys
    )
    ofsr = (
        successful_flips / len(eligible_keys) if eligible_keys else None
    )

    stance_value = {"CON": -1, "NEU": 0, "PRO": 1}
    target_direction = 1 if target == "PRO" else -1
    asv = sum(
        target_direction
        * (stance_value[comparison[key]] - stance_value[anchor[key]])
        / 2
        for key in anchor
    ) / total

    return {
        "anchor_pro": anchor_fractions["PRO"],
        "anchor_con": anchor_fractions["CON"],
        "anchor_neu": anchor_fractions["NEU"],
        "comparison_pro": comparison_fractions["PRO"],
        "comparison_con": comparison_fractions["CON"],
        "comparison_neu": comparison_fractions["NEU"],
        "change_pro": comparison_fractions["PRO"] - anchor_fractions["PRO"],
        "change_con": comparison_fractions["CON"] - anchor_fractions["CON"],
        "change_neu": comparison_fractions["NEU"] - anchor_fractions["NEU"],
        "delta_tsr": comparison_fractions[target] - anchor_fractions[target],
        "ofsr": ofsr,
        "asv": asv,
        "successful_flips": successful_flips,
        "eligible_pairs": len(eligible_keys),
        "total_paired_outputs": total,
    }


def safe_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def subtract(actual: float | None, baseline: float | None) -> float | None:
    if actual is None or baseline is None:
        return None
    return actual - baseline


def analyse_groups(
    groups: dict[tuple[str, str, str, str, str], list[Experiment]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    pairwise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for group_key, unordered_group in sorted(groups.items()):
        group = sorted(unordered_group, key=lambda experiment: experiment.method)
        dataset, generator, target_stance, retriever, mode = group_key
        matching_queries, group_exclusions = matching_group_queries(group)
        excluded_query_count = len({
            row["query_idx"] for row in group_exclusions
        })
        total_query_count = len(matching_queries) + excluded_query_count
        print(
            f"Using {len(matching_queries)}/{total_query_count} exactly matching "
            f"questions for {' | '.join(group_key)}"
        )
        for exclusion in group_exclusions:
            excluded_rows.append({
                "dataset": dataset,
                "generator": generator,
                "target_stance": target_stance,
                "retriever": retriever,
                "mode": mode,
                **exclusion,
            })

        for anchor in group:
            actual = calculate_metrics(
                anchor.clean_annotation,
                anchor.poison_annotation,
                target_stance,
                matching_queries,
            )
            baseline_results = []

            for comparison in group:
                if comparison == anchor:
                    continue
                baseline = calculate_metrics(
                    anchor.clean_annotation,
                    comparison.clean_annotation,
                    target_stance,
                    matching_queries,
                )
                baseline_results.append(baseline)
                pairwise_rows.append({
                    "dataset": dataset,
                    "generator": generator,
                    "target_stance": target_stance,
                    "retriever": retriever,
                    "mode": mode,
                    "anchor_method": anchor.method,
                    "comparison_method": comparison.method,
                    "anchor_run": anchor.run_dir.name,
                    "comparison_run": comparison.run_dir.name,
                    **baseline,
                })

            row: dict[str, Any] = {
                "dataset": dataset,
                "generator": generator,
                "target_stance": target_stance,
                "retriever": retriever,
                "mode": mode,
                "method": anchor.method,
            }

            for metric in CALIBRATED_METRICS:
                values = [result[metric] for result in baseline_results]
                baseline_mean = safe_mean(values)
                row[f"original_{metric}"] = actual[metric]
                row[f"clean_baseline_{metric}_average"] = baseline_mean
                row[f"calibrated_{metric}"] = subtract(
                    actual[metric], baseline_mean
                )

            summary_rows.append(row)

    return pairwise_rows, summary_rows, excluded_rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"No rows available for {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate clean-run baselines and calibrated stance-change metrics "
            "from completed orchestrator experiments."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs_non_agentic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clean_metric_calibration"),
    )
    parser.add_argument(
        "--expected-clean-runs",
        type=int,
        default=4,
        help="Required number of attack runs/clean references per setting; 0 disables the check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only display the discovered groups and files.",
    )
    args = parser.parse_args()

    experiments = discover_experiments(args.runs_dir)
    groups = group_experiments(experiments, args.expected_clean_runs)

    for key, group in sorted(groups.items()):
        print(" | ".join(key))
        for experiment in sorted(group, key=lambda item: item.method):
            print(f"  {experiment.method}: {experiment.clean_annotation}")

    if args.dry_run:
        return

    pairwise_rows, summary_rows, excluded_rows = analyse_groups(groups)
    pairwise_path = args.output_dir / "clean_baseline_pairwise.csv"
    summary_path = args.output_dir / "calibrated_metrics.csv"
    json_path = args.output_dir / "calibrated_metrics.json"
    exclusions_path = args.output_dir / "excluded_topics.csv"

    write_csv(pairwise_path, pairwise_rows)
    write_csv(summary_path, summary_rows)
    write_csv(
        exclusions_path,
        excluded_rows,
        fieldnames=[
            "dataset",
            "generator",
            "target_stance",
            "retriever",
            "mode",
            "query_idx",
            "method",
            "reason",
            "question",
        ],
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary_rows, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved clean baseline comparisons: {pairwise_path}")
    print(f"Saved calibrated metric table:    {summary_path}")
    print(f"Saved calibrated metric JSON:     {json_path}")
    print(f"Saved excluded topic audit:        {exclusions_path}")


if __name__ == "__main__":
    main()
