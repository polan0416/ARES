#!/usr/bin/env python3
"""Build reasoner KGQA answer SFT data from kgqa_data retrieval pth + clean_road_gold.

Output format matches train/reasoner/scripts/build_kgqa_answer_data.py SFT split but without
Question Explain. Writes only sft/train.json, sft/val.json, sft/dataset_info.json under
train/reasoner/data/{dataset}/.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_RETRIEVER_SFT_DIR = _SCRIPT_DIR.parent / "retriever" / "sft"
if str(_RETRIEVER_SFT_DIR) not in sys.path:
    sys.path.insert(0, str(_RETRIEVER_SFT_DIR))

from build_sft_data import (  # noqa: E402
    DATASET_CHOICES,
    DEFAULT_SPLITS,
    DEFAULT_TOP_K,
    RESULT_FILE_BY_SPLIT,
    SPLIT_CHOICES,
    build_candidate_triples,
    build_question_index,
    filter_answers_reachable_from_triples,
    format_candidate_text,
    load_json,
    load_torch_dict,
    map_ordered_path_triples_to_indices,
    normalize_entity_list,
    normalize_question,
    parse_sample_limits_json,
    rebuild_working_candidate_maps,
    resolve_gold_file,
    resolve_inference_sample_id,
    safe_text,
    select_retained_paths_and_gold_union,
    write_json,
    _ordered_path_triples,
)

_KGQA_ROOT = _RETRIEVER_SFT_DIR / "kgqa_data"
_REASONER_DATA_ROOT = _SCRIPT_DIR / "data"

DEFAULT_DATASETS = ("webqsp", "cwq")

SFT_INSTRUCTION_NO_EXPLAIN = (
    "Based on the triplets retrieved from a knowledge graph, please answer the question. "
    'Please return formatted answers as a list, each prefixed with "ans:". '
    "Return all the possible answers."
)


def render_triples_text(triples: Sequence[Sequence[str]]) -> str:
    return "\n".join(f"({t[0]},{t[1]},{t[2]})" for t in triples)


def format_gold_answers(answers: Sequence[str]) -> str:
    return "\n".join(f"ans: {a}" for a in answers if safe_text(a))


def canonicalize_answer_entities(entities: Sequence[str]) -> List[str]:
    return [safe_text(e) for e in entities if safe_text(e)]


def preserve_original_answers(
    original: Sequence[str],
    canonical: Sequence[str],
) -> List[str]:
    orig = [safe_text(a) for a in original if safe_text(a)]
    if orig:
        return orig
    return list(canonical)


def default_output_root_for_dataset(dataset: str) -> Path:
    return _REASONER_DATA_ROOT / dataset / "sft"


def resolve_output_root(datasets: Sequence[str], output_root: Optional[str]) -> Path:
    if output_root:
        return Path(output_root)
    if len(datasets) != 1:
        raise SystemExit(
            "error: --output_root is required when building multiple datasets in one run "
            f"(got datasets={list(datasets)!r})."
        )
    return default_output_root_for_dataset(datasets[0])


def resolve_kgqa_retrieval_file(dataset: str, split: str, kgqa_root: Optional[Path] = None) -> Path:
    root = kgqa_root or _KGQA_ROOT
    path = root / dataset / RESULT_FILE_BY_SPLIT[split]
    if path.is_file():
        return path
    if split == "test":
        alt = root / dataset / "retrieval_result_test.pth"
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Missing KGQA retrieval file for {dataset}/{split}: {path}")


def apply_missing_gold_replacements_back_to_front(
    working_candidates: List[Dict[str, Any]],
    gold_set: set[Tuple[str, str, str]],
) -> Tuple[bool, Optional[str]]:
    present = {(c["head"], c["relation"], c["tail"]) for c in working_candidates}
    missing = [g for g in gold_set if g not in present]
    missing.sort(key=lambda x: (x[0], x[1], x[2]))
    replaceable = sorted(
        (int(c["idx"]) for c in working_candidates if (c["head"], c["relation"], c["tail"]) not in gold_set),
        reverse=True,
    )
    if len(replaceable) < len(missing):
        return False, "insufficient_non_gold_slots_for_replacement"
    idx_to_cand = {int(c["idx"]): c for c in working_candidates}
    for slot_idx, gold_tuple in zip(replaceable, missing):
        tgt = idx_to_cand[slot_idx]
        tgt["head"] = gold_tuple[0]
        tgt["relation"] = gold_tuple[1]
        tgt["tail"] = gold_tuple[2]
        tgt["text"] = format_candidate_text(tgt)
    return True, None


def map_sample_paths_union_to_candidates_back_to_front(
    paths: Sequence[Dict[str, Any]],
    candidate_triples: List[Dict[str, Any]],
    top_k: int,
    answer_entities: Sequence[str],
    topic_entity: str,
) -> Dict[str, Any]:
    retained_indices, gold_set, budget_reason = select_retained_paths_and_gold_union(paths, top_k)
    if not gold_set:
        return {"valid": False, "mismatch_reason": budget_reason, "candidate_triples": candidate_triples}

    working_candidates = [dict(c) for c in candidate_triples]
    ok, repl_reason = apply_missing_gold_replacements_back_to_front(working_candidates, gold_set)
    if not ok:
        return {"valid": False, "mismatch_reason": repl_reason or "replacement_failed", "candidate_triples": candidate_triples}

    _, indices_by_tuple = rebuild_working_candidate_maps(working_candidates)
    for g in gold_set:
        if g not in indices_by_tuple:
            return {
                "valid": False,
                "mismatch_reason": f"gold_triplet_not_in_candidates:{g[0]}|{g[1]}|{g[2]}",
                "candidate_triples": working_candidates,
            }

    retained_payloads = [paths[i] for i in retained_indices]
    filtered_answers = filter_answers_reachable_from_triples(
        topic_entity=topic_entity,
        gold_triples=gold_set,
        answer_entities=answer_entities,
        retained_paths=retained_payloads,
    )

    per_path: List[Dict[str, Any]] = []
    for i in retained_indices:
        ordered = _ordered_path_triples(paths[i])
        idxs = map_ordered_path_triples_to_indices(ordered, working_candidates)
        if idxs is None:
            return {
                "valid": False,
                "mismatch_reason": f"path_triple_mapping_failed:path_index={i}",
                "candidate_triples": working_candidates,
            }
        per_path.append({"path_index": i, "gold_path_indices": idxs})

    return {
        "valid": True,
        "candidate_triples": working_candidates,
        "answer_entities": filtered_answers,
        "budget_reason": budget_reason,
    }


def candidates_to_triple_lists(candidates: Sequence[Dict[str, Any]]) -> List[List[str]]:
    ordered = sorted(candidates, key=lambda c: int(c["idx"]))
    return [[c["head"], c["relation"], c["tail"]] for c in ordered]


def render_sft_input_no_explain(sample: Dict[str, Any], render_triples_text) -> str:
    question = safe_text(sample.get("question"))
    if question and not question.endswith("?"):
        question = f"{question}?"
    return f"Triplets:\n{render_triples_text(sample['triples'])}\n\n\nQuestion:\n{question}"


def build_sft_records_no_explain(
    shared_records: Sequence[Dict[str, Any]],
    render_triples_text,
    format_gold_answers,
) -> List[Dict[str, Any]]:
    return [
        {
            "instruction": SFT_INSTRUCTION_NO_EXPLAIN,
            "input": render_sft_input_no_explain(sample, render_triples_text),
            "output": format_gold_answers(sample["gold_answers"]),
        }
        for sample in shared_records
    ]


def build_dataset_info(data_source: str) -> Dict[str, Any]:
    return {
        f"{data_source}_train": {
            "file_name": "train.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        f"{data_source}_val": {
            "file_name": "val.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }


def build_shared_records_for_split(
    dataset: str,
    split: str,
    top_k: int,
    sample_limit: Optional[int],
    kgqa_root: Optional[Path],
    inject_gold_paths: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data_source = f"kgqa_answer_{dataset}"
    gold_payload = load_json(str(resolve_gold_file(dataset, split)))
    retrieval_payload = load_torch_dict(str(resolve_kgqa_retrieval_file(dataset, split, kgqa_root=kgqa_root)))
    question_index = build_question_index(retrieval_payload)

    stats = Counter()
    stats["gold_samples"] = len(gold_payload)
    stats["kgqa_retrieval_samples"] = len(retrieval_payload)
    retained: List[Dict[str, Any]] = []
    dropped_examples: List[Dict[str, Any]] = []

    for gold_idx, (gold_sample_id, gold_sample) in enumerate(gold_payload.items()):
        if sample_limit is not None and gold_idx >= sample_limit:
            break

        inferred_sample_id, alignment_method, _ = resolve_inference_sample_id(
            gold_sample_id=gold_sample_id,
            gold_sample=gold_sample,
            inference_payload=retrieval_payload,
            question_index=question_index,
        )
        if inferred_sample_id is None:
            stats["dropped_missing_kgqa_retrieval"] += 1
            dropped_examples.append({"sample_id": gold_sample_id, "reason": "missing_kgqa_retrieval"})
            continue
        if alignment_method == "question":
            stats["question_fallback_matches"] += 1

        retrieval_sample = retrieval_payload[inferred_sample_id]
        candidate_triples = build_candidate_triples(retrieval_sample, top_k=top_k)

        question = normalize_question(gold_sample.get("question") or retrieval_sample.get("question"))
        topic_entities = normalize_entity_list(
            gold_sample.get("topic_entities")
            or retrieval_sample.get("q_entity_in_graph")
            or retrieval_sample.get("q_entity")
        )
        topic_entity = topic_entities[0] if topic_entities else ""
        answer_entities = normalize_entity_list(
            gold_sample.get("answer_entities")
            or retrieval_sample.get("a_entity_in_graph")
            or retrieval_sample.get("a_entity")
        )
        paths = gold_sample.get("paths") or []

        if inject_gold_paths and paths:
            union_map = map_sample_paths_union_to_candidates_back_to_front(
                paths=paths,
                candidate_triples=candidate_triples,
                top_k=top_k,
                answer_entities=answer_entities,
                topic_entity=topic_entity,
            )
            if not union_map["valid"]:
                stats["dropped_invalid_gold_union"] += 1
                dropped_examples.append(
                    {
                        "sample_id": gold_sample_id,
                        "reason": safe_text(union_map.get("mismatch_reason")) or "invalid_gold_union",
                    }
                )
                continue
            candidate_triples = union_map["candidate_triples"]
            answer_entities = union_map["answer_entities"] or answer_entities

        triples = candidates_to_triple_lists(candidate_triples)
        gold_answers = preserve_original_answers(
            answer_entities,
            canonicalize_answer_entities(answer_entities),
        )
        if not gold_answers:
            gold_answers = [safe_text(a) for a in answer_entities if safe_text(a)]

        if not question:
            stats["dropped_missing_question"] += 1
            dropped_examples.append({"sample_id": gold_sample_id, "reason": "missing_question"})
            continue
        if not triples:
            stats["dropped_missing_triples"] += 1
            dropped_examples.append({"sample_id": gold_sample_id, "reason": "missing_triples"})
            continue
        if not gold_answers:
            stats["dropped_missing_answers"] += 1
            dropped_examples.append({"sample_id": gold_sample_id, "reason": "missing_answers"})
            continue

        retained.append(
            {
                "dataset": dataset,
                "split": split,
                "data_source": data_source,
                "sample_id": str(gold_sample_id),
                "question": question,
                "topic_entity": topic_entity,
                "topic_entities": topic_entities,
                "triples": triples,
                "gold_answers": gold_answers,
                "gold_answer_entities": gold_answers,
                "gold_triples": triples,
            }
        )
        stats["retained"] += 1

    return retained, {"stats": dict(stats), "dropped_examples": dropped_examples}


def write_dataset_outputs(
    dataset: str,
    output_root: Path,
    shared_by_split: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    data_source = f"kgqa_answer_{dataset}"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_splits: Dict[str, Any] = {}
    sft_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split, shared_records in shared_by_split.items():
        sft_records = build_sft_records_no_explain(
            shared_records,
            render_triples_text,
            format_gold_answers,
        )
        sft_by_split[split] = sft_records
        summary_splits[split] = {
            "retained": len(shared_records),
            "sft_examples": len(sft_records),
        }

    write_json(str(output_root / "train.json"), sft_by_split.get("train", []))
    write_json(str(output_root / "val.json"), sft_by_split.get("val", []))
    write_json(str(output_root / "dataset_info.json"), build_dataset_info(data_source))

    return {
        "dataset": dataset,
        "data_source": data_source,
        "output_root": str(output_root),
        "splits": summary_splits,
    }


def build_kgqa_reasoner_data(args: argparse.Namespace) -> Dict[str, Any]:
    kgqa_root = Path(args.kgqa_data_root) if args.kgqa_data_root else None
    all_summaries: List[Dict[str, Any]] = []

    try:
        sample_limits_per_split = parse_sample_limits_json(args.sample_limits_json)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    for dataset in args.datasets:
        if args.output_root:
            output_root = Path(args.output_root)
            if len(args.datasets) > 1:
                output_root = output_root / dataset / "sft"
        else:
            output_root = default_output_root_for_dataset(dataset)

        shared_by_split: Dict[str, List[Dict[str, Any]]] = {}
        split_reports: Dict[str, Any] = {}

        for split in args.splits:
            eff_sample_limit = (
                sample_limits_per_split.get(split, args.sample_limit)
                if sample_limits_per_split is not None
                else args.sample_limit
            )
            shared_records, split_report = build_shared_records_for_split(
                dataset=dataset,
                split=split,
                top_k=args.top_k,
                sample_limit=eff_sample_limit,
                kgqa_root=kgqa_root,
                inject_gold_paths=args.inject_gold_paths,
            )
            shared_by_split[split] = shared_records
            split_reports[split] = split_report

        dataset_summary = write_dataset_outputs(dataset, output_root, shared_by_split)
        dataset_summary["split_reports"] = {s: split_reports[s]["stats"] for s in args.splits if s in split_reports}
        all_summaries.append(dataset_summary)

    return {"datasets": all_summaries}


def format_build_summary(summary: Dict[str, Any]) -> str:
    lines = ["KGQA reasoner answer data build complete (no Question Explain).", ""]
    for item in summary["datasets"]:
        lines.append(f"dataset: {item['dataset']}")
        lines.append(f"  output_root: {item['output_root']}")
        lines.append(f"  data_source: {item['data_source']}")
        for split, info in item["splits"].items():
            lines.append(f"  {split}: retained={info['retained']}, sft={info['sft_examples']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reasoner KGQA answer data from kgqa_data pth (no Question Explain)."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=list(DEFAULT_DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_CHOICES, default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=f"Output dataset root (default: {_REASONER_DATA_ROOT}/<dataset>)",
    )
    parser.add_argument("--kgqa_data_root", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--inject_gold_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--sample_limits_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_kgqa_reasoner_data(args)
    print(format_build_summary(summary))


if __name__ == "__main__":
    main()
