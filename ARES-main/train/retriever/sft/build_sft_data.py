from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch


DATASET_CHOICES = ("webqsp", "cwq")
DEFAULT_DATASETS = ("webqsp")
DEFAULT_SPLITS = ("train", "val")
SPLIT_CHOICES = ("train", "val", "test")
DEFAULT_TOP_K = 100
DEFAULT_MERGE_PATHS_PER_SAMPLE = True
DEFAULT_STOP_ACTION = "STOP"
DEFAULT_STOP_POSITIVE_MULTIPLIER = 1

_SCRIPT_DIR = Path(__file__).resolve().parent
_ARES_MAIN_ROOT = _SCRIPT_DIR.parents[2]
_INFERENCE_ROOT = _ARES_MAIN_ROOT / "retrieve" / "stage2_results" / "inference"
_GOLD_ROOT = _SCRIPT_DIR / "clean_road_gold"

RESULT_FILE_BY_SPLIT = {
    "train": "retrieval_result_train.pth",
    "val": "retrieval_result_val.pth",
    "test": "retrieval_result.pth",
}

GOLD_FILE_BY_SPLIT = {
    "train": "train_answer_paths.json",
    "val": "val_answer_paths.json",
    "test": "test_answer_paths.json",
}


def resolve_inference_file(dataset: str, split: str) -> Path:
    path = _INFERENCE_ROOT / dataset / RESULT_FILE_BY_SPLIT[split]
    if path.is_file():
        return path
    if split == "test":
        alt = _INFERENCE_ROOT / dataset / "retrieval_result_test.pth"
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Missing inference file for {dataset}/{split}: {path}")


def resolve_gold_file(dataset: str, split: str) -> Path:
    if split not in GOLD_FILE_BY_SPLIT:
        raise ValueError(f"Unsupported split: {split}")
    path = _GOLD_ROOT / dataset / GOLD_FILE_BY_SPLIT[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing gold file for {dataset}/{split}: {path}")
    return path


def default_output_root_for_dataset(dataset: str) -> str:
    return str(_SCRIPT_DIR / "training_data" / dataset)


def resolve_output_root(datasets: Sequence[str], output_root: Optional[str]) -> str:
    if output_root:
        return output_root
    if len(datasets) != 1:
        raise SystemExit(
            "error: --output_root is required when building multiple datasets in one run "
            f"(got datasets={list(datasets)!r})."
        )
    return default_output_root_for_dataset(datasets[0])

SFT_DATASET_NAMES = {
    "train": "sequential_triple_selection_train",
    "val": "sequential_triple_selection_val",
}

SFT_SYSTEM = (
    "You are a knowledge-graph action agent.\n"
    "Output exactly one JSON action and nothing else.\n\n"
    "Allowed outputs:\n"
    '{"action":"ADD_TRIPLE","triple_id":<id>}\n'
    '{"action":"STOP"}\n\n'
    "Rules:\n"
    "- Use only provided candidate triples.\n"
    "- Do not invent IDs or actions.\n"
    "- No explanation."
)

SFT_INSTRUCTION = (
    "Choose the best next action.\n\n"
    "Priority:\n"
    "1. semantic match\n"
    "2. direct answer\n"
    "3. connectivity unless intermediate expansion is required\n"
    "4. no repeated triple\n"
    "5. STOP only if no candidate improves progress"
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_dict(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def write_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl_line(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_question(value: Any) -> str:
    return safe_text(value)


def normalize_entity_list(value: Any) -> List[str]:
    return [safe_text(item) for item in ensure_list(value) if safe_text(item)]


def normalize_candidate_tuple(raw_triple: Sequence[Any]) -> Tuple[str, str, str]:
    if len(raw_triple) < 3:
        raise ValueError(f"Expected triple with at least three elements, got: {raw_triple}")
    return safe_text(raw_triple[0]), safe_text(raw_triple[1]), safe_text(raw_triple[2])


def normalize_gold_triplet(raw_triplet: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        safe_text(raw_triplet.get("head")),
        safe_text(raw_triplet.get("relation")),
        safe_text(raw_triplet.get("tail")),
    )


def format_candidate_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def format_indexed_candidate_triple_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def to_v1_messages_record(system: str, instruction: str, input_text: str, output_text: str) -> Dict[str, Any]:
    user_text = instruction
    if input_text:
        user_text = f"{instruction}\n{input_text}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system, "loss_weight": 0.0})
    messages.extend(
        [
            {"role": "user", "content": user_text, "loss_weight": 0.0},
            {"role": "assistant", "content": output_text, "loss_weight": 1.0},
        ]
    )
    return {"messages": messages}


def build_structured_input(
    trajectory: Dict[str, Any],
    selected_indices: Sequence[int],
    candidate_triples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    selected_set = set(selected_indices)
    selected_triples = [
        format_indexed_candidate_triple_text(candidate)
        for candidate in candidate_triples
        if candidate["idx"] in selected_set
    ]
    remaining_candidates = [
        format_indexed_candidate_triple_text(candidate)
        for candidate in candidate_triples
        if candidate["idx"] not in selected_set
    ]

    current_nodes: set[str] = set(trajectory.get("topic_entities") or [])
    for candidate in candidate_triples:
        if candidate["idx"] in selected_set:
            current_nodes.add(candidate["head"])
            current_nodes.add(candidate["tail"])

    return {
        "question": trajectory.get("question", ""),
        "state": {
            "selected_triples": selected_triples,
            "current_nodes": sorted(current_nodes),
        },
        "candidate_triples": remaining_candidates,
        "action_rules": {
            "allow_stop": True,
            "do_not_repeat_triples": True,
            "stop_condition": (
                "Output STOP only when no remaining candidate triple can improve answer coverage or connectivity."
            ),
        },
    }


def render_sft_structured_prompt(input_payload: Dict[str, Any]) -> str:
    question = safe_text(input_payload.get("question"))
    if question and not question.rstrip().endswith("?"):
        question = f"{question.rstrip()}?"
    state = input_payload.get("state") or {}
    candidate_triples = input_payload.get("candidate_triples") or []
    action_rules = input_payload.get("action_rules") or {}

    candidate_lines = "\n".join([safe_text(item) for item in candidate_triples if safe_text(item)])
    if not candidate_lines:
        candidate_lines = "(none)"

    stop_condition = safe_text(action_rules.get("stop_condition")) or (
        "Output STOP only when no remaining candidate can improve answer coverage or connectivity."
    )

    return (
        "Candidate Triples:\n"
        f"{candidate_lines}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "state:\n"
        f"{json.dumps(state if isinstance(state, dict) else {}, ensure_ascii=False)}\n\n"
        "Action Rules:\n"
        "- Do not repeat triples\n"
        f"- {stop_condition}"
    )


def build_candidate_triples(sample: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    scored_triples = sample.get("scored_triples") or []
    seen_triples: set[Tuple[str, str, str]] = set()
    unique_triples: List[Tuple[str, str, str]] = []
    for raw in scored_triples:
        head, relation, tail = normalize_candidate_tuple(raw)
        triple_key = (head, relation, tail)
        if triple_key in seen_triples:
            continue
        seen_triples.add(triple_key)
        unique_triples.append(triple_key)
        if len(unique_triples) >= top_k:
            break

    for idx, (head, relation, tail) in enumerate(unique_triples, start=1):
        candidate = {"idx": idx, "head": head, "relation": relation, "tail": tail}
        candidate["text"] = format_candidate_text(candidate)
        candidates.append(candidate)
    return candidates


def build_question_index(inference_payload: Dict[str, Any]) -> Dict[str, List[str]]:
    question_to_ids: Dict[str, List[str]] = defaultdict(list)
    for sample_id, sample in inference_payload.items():
        if not isinstance(sample, dict):
            continue
        question = normalize_question(sample.get("question"))
        if question:
            question_to_ids[question].append(sample_id)
    return dict(question_to_ids)


def _path_triples_set(path_payload: Dict[str, Any]) -> set[Tuple[str, str, str]]:
    return {normalize_gold_triplet(item) for item in (path_payload.get("path_triplets") or [])}


def _ordered_path_triples(path_payload: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    return [normalize_gold_triplet(item) for item in (path_payload.get("path_triplets") or [])]


def select_retained_paths_and_gold_union(
    paths: Sequence[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[int], set[Tuple[str, str, str]], str]:
    path_sets = [_path_triples_set(p) for p in paths]
    full_union: set[Tuple[str, str, str]] = set()
    for s in path_sets:
        full_union |= s

    if not full_union:
        return [], set(), "empty_gold_union"

    if len(full_union) <= top_k:
        retained = [i for i, s in enumerate(path_sets) if s]
        return retained, full_union, "all_paths_within_budget"

    union: set[Tuple[str, str, str]] = set()
    retained: List[int] = []
    for i, triples in enumerate(path_sets):
        if not triples:
            continue
        if len(triples) > top_k:
            continue
        nu = union | triples
        if len(nu) <= top_k:
            union = nu
            retained.append(i)
        else:
            break

    if not union:
        return [], set(), "no_prefix_fits_within_budget"
    return retained, union, "prefix_paths_within_budget"


def filter_answers_reachable_from_triples(
    topic_entity: str,
    gold_triples: set[Tuple[str, str, str]],
    answer_entities: Sequence[str],
    retained_paths: Sequence[Dict[str, Any]],
) -> List[str]:
    answers = [safe_text(a) for a in answer_entities if safe_text(a)]
    if not answers:
        return []

    nodes: set[str] = set()
    adj: Dict[str, set[str]] = defaultdict(set)
    for h, _r, t in gold_triples:
        if h:
            nodes.add(h)
        if t:
            nodes.add(t)
        if h and t:
            adj[h].add(t)
            adj[t].add(h)

    topic = safe_text(topic_entity)
    if topic and topic in nodes:
        visited: set[str] = {topic}
        stack = [topic]
        while stack:
            u = stack.pop()
            for v in adj.get(u, set()):
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return [a for a in answers if a in visited]

    from_paths: set[str] = set()
    for p in retained_paths:
        ae = safe_text(p.get("answer_entity"))
        if ae:
            from_paths.add(ae)
    if from_paths:
        return [a for a in answers if a in from_paths]
    return []


def rebuild_working_candidate_maps(
    working_candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[int, Tuple[str, str, str]], Dict[Tuple[str, str, str], List[int]]]:
    tuple_by_idx = {c["idx"]: (c["head"], c["relation"], c["tail"]) for c in working_candidates}
    indices_by_tuple: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for c in working_candidates:
        indices_by_tuple[(c["head"], c["relation"], c["tail"])].append(int(c["idx"]))
    for t in indices_by_tuple:
        indices_by_tuple[t].sort()
    return tuple_by_idx, dict(indices_by_tuple)


def apply_missing_gold_replacements_front_to_back(
    working_candidates: List[Dict[str, Any]],
    gold_set: set[Tuple[str, str, str]],
) -> Tuple[bool, Optional[str]]:
    present = {(c["head"], c["relation"], c["tail"]) for c in working_candidates}
    missing = [g for g in gold_set if g not in present]
    missing.sort(key=lambda x: (x[0], x[1], x[2]))
    replaceable = sorted(
        int(c["idx"]) for c in working_candidates if (c["head"], c["relation"], c["tail"]) not in gold_set
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


def gold_union_candidate_indices(
    working_candidates: Sequence[Dict[str, Any]],
    gold_set: set[Tuple[str, str, str]],
) -> List[int]:
    out: List[int] = []
    for c in sorted(working_candidates, key=lambda x: int(x["idx"])):
        if (c["head"], c["relation"], c["tail"]) in gold_set:
            out.append(int(c["idx"]))
    return out


def map_ordered_path_triples_to_indices(
    ordered_triples: Sequence[Tuple[str, str, str]],
    working_candidates: Sequence[Dict[str, Any]],
) -> Optional[List[int]]:
    _, indices_by_tuple = rebuild_working_candidate_maps(working_candidates)
    usage: Dict[Tuple[str, str, str], int] = defaultdict(int)
    out: List[int] = []
    for gold_tuple in ordered_triples:
        pool = indices_by_tuple.get(gold_tuple)
        if not pool:
            return None
        u = usage[gold_tuple]
        if u >= len(pool):
            return None
        out.append(pool[u])
        usage[gold_tuple] += 1
    return out


def map_sample_paths_union_to_candidates(
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
    ok, repl_reason = apply_missing_gold_replacements_front_to_back(working_candidates, gold_set)
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

    gold_path_indices_union = gold_union_candidate_indices(working_candidates, gold_set)
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
        "gold_path_indices_union": gold_path_indices_union,
        "per_path_mappings": per_path,
        "answer_entities": filtered_answers,
        "budget_reason": budget_reason,
    }


def resolve_inference_sample_id(
    gold_sample_id: str,
    gold_sample: Dict[str, Any],
    inference_payload: Dict[str, Any],
    question_index: Dict[str, List[str]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if gold_sample_id in inference_payload:
        return gold_sample_id, "id", None

    gold_inner_id = safe_text(gold_sample.get("id"))
    if gold_inner_id and gold_inner_id in inference_payload:
        return gold_inner_id, "gold_inner_id", None

    question = normalize_question(gold_sample.get("question"))
    matched_ids = question_index.get(question, [])
    if len(matched_ids) == 1:
        return matched_ids[0], "question", None
    if len(matched_ids) > 1:
        if gold_inner_id and gold_inner_id in matched_ids:
            return gold_inner_id, "gold_inner_id+question", None
        return None, None, f"ambiguous_question_match:{len(matched_ids)}"
    return None, None, "missing_inference_sample"


def candidate_signature(candidate_triples: Sequence[Dict[str, Any]]) -> Tuple[Tuple[int, str, str, str], ...]:
    return tuple(
        (candidate["idx"], candidate["head"], candidate["relation"], candidate["tail"])
        for candidate in candidate_triples
    )


def build_trajectories_for_split(
    dataset: str,
    split: str,
    top_k: int = DEFAULT_TOP_K,
    merge_paths_per_sample: bool = False,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    gold_payload = load_json(str(resolve_gold_file(dataset, split)))
    inference_payload = load_torch_dict(str(resolve_inference_file(dataset, split)))
    question_index = build_question_index(inference_payload)

    stats = Counter()
    stats["dataset"] = dataset
    stats["split"] = split
    stats["gold_samples"] = len(gold_payload)
    trajectories: List[Dict[str, Any]] = []

    for gold_idx, (gold_sample_id, gold_sample) in enumerate(gold_payload.items()):
        if sample_limit is not None and gold_idx >= sample_limit:
            break

        inferred_sample_id, alignment_method, alignment_error = resolve_inference_sample_id(
            gold_sample_id=gold_sample_id,
            gold_sample=gold_sample,
            inference_payload=inference_payload,
            question_index=question_index,
        )
        if inferred_sample_id is None:
            stats["gold_missing_inference"] += 1
            continue

        if alignment_method == "question":
            stats["question_fallback_matches"] += 1
        stats["retained_samples"] += 1

        inference_sample = inference_payload[inferred_sample_id]
        candidate_triples = build_candidate_triples(inference_sample, top_k=top_k)

        question = normalize_question(gold_sample.get("question") or inference_sample.get("question"))
        topic_entities = normalize_entity_list(
            gold_sample.get("topic_entities")
            or inference_sample.get("q_entity_in_graph")
            or inference_sample.get("q_entity")
        )
        answer_entities = normalize_entity_list(
            gold_sample.get("answer_entities")
            or inference_sample.get("a_entity_in_graph")
            or inference_sample.get("a_entity")
        )
        topic_entity = topic_entities[0] if topic_entities else ""

        paths = gold_sample.get("paths") or []
        stats["gold_paths_total"] += len(paths)

        union_map = map_sample_paths_union_to_candidates(
            paths=paths,
            candidate_triples=candidate_triples,
            top_k=top_k,
            answer_entities=answer_entities,
            topic_entity=topic_entity,
        )

        if not union_map["valid"]:
            stats["invalid_sample_union"] += 1
            reason = safe_text(union_map.get("mismatch_reason")) or "unknown"
            stats[f"invalid_union_reason::{reason}"] += 1
            stats["retained_samples_without_valid_path"] += 1
            continue

        working_candidates = union_map["candidate_triples"]
        filtered_answer_entities = union_map["answer_entities"]
        budget_reason = union_map.get("budget_reason") or ""
        stats[f"sample_union_budget::{budget_reason}"] += 1
        candidates_by_idx = {c["idx"]: c for c in working_candidates}

        valid_mappings_for_sample: List[Dict[str, Any]] = []
        for pm in union_map["per_path_mappings"]:
            path_index = int(pm["path_index"])
            path_payload = paths[path_index]
            gold_path_indices = pm["gold_path_indices"]
            valid_mappings_for_sample.append(
                {
                    "path_index": path_index,
                    "path_payload": path_payload,
                    "gold_path_indices": gold_path_indices,
                }
            )

        if (not merge_paths_per_sample) and valid_mappings_for_sample:
            for item in valid_mappings_for_sample:
                path_index = int(item["path_index"])
                path_payload = item["path_payload"]
                trajectories.append(
                    {
                        "trajectory_id": f"{dataset}:{split}:{gold_sample_id}:path{path_index}",
                        "dataset": dataset,
                        "split": split,
                        "sample_id": gold_sample_id,
                        "path_index": path_index,
                        "question": question,
                        "topic_entities": topic_entities,
                        "answer_entities": filtered_answer_entities,
                        "candidate_triples": working_candidates,
                        "gold_path_indices": item["gold_path_indices"],
                    }
                )
                stats["valid_trajectories"] += 1

        if merge_paths_per_sample and valid_mappings_for_sample:
            merged_indices = list(union_map["gold_path_indices_union"])
            trajectories.append(
                {
                    "trajectory_id": f"{dataset}:{split}:{gold_sample_id}:merged",
                    "dataset": dataset,
                    "split": split,
                    "sample_id": gold_sample_id,
                    "path_index": -1,
                    "question": question,
                    "topic_entities": topic_entities,
                    "answer_entities": filtered_answer_entities,
                    "candidate_triples": working_candidates,
                    "gold_path_indices": merged_indices,
                }
            )
            stats["valid_trajectories"] += 1
            stats["mapping_method::merged_sample_union"] += 1

        stats["retained_samples_with_valid_path"] += 1

    return {"dataset": dataset, "split": split, "trajectories": trajectories, "stats": dict(stats)}


def iter_trajectories(trajectories: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    yield from trajectories


def build_sft_datasets(
    trajectories_by_split: Dict[str, List[Dict[str, Any]]],
    output_root: str,
    splits: Sequence[str] = DEFAULT_SPLITS,
    stop_positive_multiplier: int = DEFAULT_STOP_POSITIVE_MULTIPLIER,
) -> Dict[str, Any]:
    ensure_dir(output_root)
    summary: Dict[str, Any] = {"paths": {}, "stats": {}}

    for split in splits:
        output_path = os.path.join(output_root, f"{split}.jsonl")
        stats = Counter()
        state_pool: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for trajectory in iter_trajectories(trajectories_by_split.get(split, [])):
            gold_path_indices = list(trajectory["gold_path_indices"])
            stats["input_trajectories"] += 1
            stats["base_examples"] += len(gold_path_indices) + 1

            for prefix_len in range(len(gold_path_indices) + 1):
                selected_indices = tuple(gold_path_indices[:prefix_len])
                key = (
                    trajectory["dataset"],
                    trajectory["split"],
                    trajectory["sample_id"],
                    selected_indices,
                    candidate_signature(trajectory["candidate_triples"]),
                )
                state = state_pool.get(key)
                if state is None:
                    input_payload = build_structured_input(
                        trajectory=trajectory,
                        selected_indices=selected_indices,
                        candidate_triples=trajectory["candidate_triples"],
                    )
                    state = {
                        "instruction": SFT_INSTRUCTION,
                        "input": input_payload,
                        "system": SFT_SYSTEM,
                        "positive_actions": set(),
                    }
                    state_pool[key] = state

                if prefix_len < len(gold_path_indices):
                    state["positive_actions"].add(gold_path_indices[prefix_len])
                if prefix_len == len(gold_path_indices):
                    state["positive_actions"].add(DEFAULT_STOP_ACTION)

        with open(output_path, "w", encoding="utf-8") as out_f:
            for state in state_pool.values():
                positive_actions = list(state["positive_actions"])
                if not positive_actions:
                    continue
                int_actions = sorted([action for action in positive_actions if isinstance(action, int)])
                ordered_actions: List[Any] = int_actions + (
                    [DEFAULT_STOP_ACTION] if DEFAULT_STOP_ACTION in positive_actions else []
                )
                if len(ordered_actions) > 1:
                    stats["multi_positive_states"] += 1
                stats["unique_states"] += 1

                for action in ordered_actions:
                    is_stop_action = action == DEFAULT_STOP_ACTION
                    copies = stop_positive_multiplier if is_stop_action else 1
                    if is_stop_action:
                        stats["stop_positive_base_examples"] += 1
                        stats["stop_positive_written_examples"] += copies
                    for _ in range(copies):
                        if is_stop_action:
                            raw_output = json.dumps({"action": "STOP"}, ensure_ascii=False, separators=(",", ":"))
                        else:
                            raw_output = json.dumps(
                                {"action": "ADD_TRIPLE", "triple_id": int(action)},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        input_for_lf = render_sft_structured_prompt(state["input"])
                        v1_record = to_v1_messages_record(
                            system=state["system"],
                            instruction=state["instruction"],
                            input_text=input_for_lf,
                            output_text=safe_text(raw_output),
                        )
                        append_jsonl_line(out_f, v1_record)
                        stats["written_examples"] += 1
                        if is_stop_action:
                            stats["written_stop_examples"] += 1
                        else:
                            stats["written_action_examples"] += 1

        summary["paths"][split] = output_path
        summary["stats"][split] = dict(stats)

    dataset_info = {}
    for split in splits:
        dataset_info[SFT_DATASET_NAMES[split]] = {
            "file_name": f"{split}.jsonl",
            "formatting": "openai",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    dataset_info_path = os.path.join(output_root, "dataset_info.json")
    write_json(dataset_info_path, dataset_info)
    summary["dataset_info_path"] = dataset_info_path
    return summary


def build_sft_data(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = resolve_output_root(args.datasets, args.output_root)
    ensure_dir(output_root)
    trajectories_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    trajectory_stats: List[Dict[str, Any]] = []

    try:
        sample_limits_per_split = parse_sample_limits_json(args.sample_limits_json)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    for dataset in args.datasets:
        for split in args.splits:
            if sample_limits_per_split is not None:
                eff_sample_limit = sample_limits_per_split.get(split, args.sample_limit)
            else:
                eff_sample_limit = args.sample_limit
            result = build_trajectories_for_split(
                dataset=dataset,
                split=split,
                top_k=args.top_k,
                merge_paths_per_sample=args.merge_paths_per_sample,
                sample_limit=eff_sample_limit,
            )
            trajectory_stats.append(
                {
                    "dataset": result["dataset"],
                    "split": result["split"],
                    "stats": result["stats"],
                }
            )
            trajectories_by_split[split].extend(result["trajectories"])

    sft_summary = build_sft_datasets(
        trajectories_by_split=trajectories_by_split,
        output_root=output_root,
        splits=args.splits,
        stop_positive_multiplier=args.stop_positive_multiplier,
    )
    return {"trajectory_stats": trajectory_stats, "sft_summary": sft_summary}


def parse_sample_limits_json(raw: Optional[str]) -> Optional[Dict[str, int]]:
    if raw is None or not str(raw).strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sample_limits_json: invalid JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise ValueError("sample_limits_json must decode to a JSON object")
    out: Dict[str, int] = {}
    for key, value in obj.items():
        split = str(key)
        if split not in SPLIT_CHOICES:
            raise ValueError(f"sample_limits_json: unknown split {split!r}, expected one of {SPLIT_CHOICES}")
        limit = int(value)
        if limit < 0:
            raise ValueError(f"sample_limits_json: limit for {split!r} must be >= 0, got {limit}")
        out[split] = limit
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequential triple selection SFT training data.")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=list(DEFAULT_DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLIT_CHOICES, default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Output directory for SFT data. Defaults to "
            "training_data/<dataset> under this script directory."
        ),
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--merge_paths_per_sample",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MERGE_PATHS_PER_SAMPLE,
        help="Merge all valid paths of one sample into a single trajectory target union.",
    )
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument(
        "--sample_limits_json",
        type=str,
        default=None,
        help='Optional JSON object mapping split to max gold samples, e.g. \'{"train":3500,"val":500}\'.',
    )
    parser.add_argument("--stop_positive_multiplier", type=int, default=DEFAULT_STOP_POSITIVE_MULTIPLIER)
    return parser.parse_args()


def format_build_summary(summary: Dict[str, Any], output_root: str) -> str:
    lines = [
        "Sequential triple selection SFT build complete.",
        f"output_root: {output_root}",
        "",
        "Trajectory stats:",
    ]
    for item in summary["trajectory_stats"]:
        stats = item["stats"]
        lines.append(
            f"  {item['dataset']}/{item['split']}: "
            f"gold_samples={stats.get('gold_samples', 0)}, "
            f"valid_trajectories={stats.get('valid_trajectories', 0)}, "
            f"retained_samples_with_valid_path={stats.get('retained_samples_with_valid_path', 0)}, "
            f"gold_missing_inference={stats.get('gold_missing_inference', 0)}, "
            f"invalid_sample_union={stats.get('invalid_sample_union', 0)}"
        )
    lines.extend(["", "SFT outputs:"])
    sft_summary = summary["sft_summary"]
    for split, path in sft_summary["paths"].items():
        stats = sft_summary["stats"][split]
        lines.append(
            f"  {path}: written_examples={stats.get('written_examples', 0)}, "
            f"unique_states={stats.get('unique_states', 0)}"
        )
    lines.append(f"  dataset_info: {sft_summary['dataset_info_path']}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.datasets, args.output_root)
    summary = build_sft_data(args)
    print(format_build_summary(summary, output_root))


if __name__ == "__main__":
    main()
