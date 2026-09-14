#!/usr/bin/env python3
"""Build retriever GRPO training data (train.jsonl / val.jsonl only, no Question Explain)."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

DATA_SOURCE = "webqsp_retriever_grpo"
DEFAULT_DATASETS = ("webqsp", "cwq")
DEFAULT_SPLITS = ("train", "val")
DEFAULT_TOP_K = 100
DEFAULT_MIN_ANSWER_HITS_TO_STOP = 1
DEFAULT_MIN_ANSWER_RECALL_TO_STOP = 1.0
DEFAULT_MIN_CONNECTIVITY_RATIO_TO_STOP = 0.6
DEFAULT_MIN_SELECTED_TRIPLES_TO_STOP = 2
DEFAULT_MAX_SELECTED_TRIPLES_SOFT = 40
DEFAULT_MAX_SELECTED_TRIPLES_HARD = 95
DEFAULT_REMOTE_TIMEOUT_SEC = 60.0
DEFAULT_STOP_ACTION = "STOP"
REWARD_TRIPLE_CAP_SOFT_BY_DATASET = {"webqsp": 14, "cwq": 14}
REWARD_TRIPLE_CAP_HARD_BY_DATASET = {"webqsp": 34, "cwq": 32}
ACTION_PATTERN = re.compile(r"\bSTOP\b|\b\d{1,3}\b", re.IGNORECASE)
DEFAULT_REMOTE_ANSWER_URL_BY_DATASET = {
    "webqsp": "http://10.87.135.152:8004/answer",
    "cwq": "http://10.87.135.152:8002/answer",
}

_RL_DIR = Path(__file__).resolve().parent
_RETRIEVER_ROOT = _RL_DIR.parent
_SFT_DIR = _RETRIEVER_ROOT / "sft"
_ARES_MAIN_ROOT = _RETRIEVER_ROOT.parents[1]
_INFERENCE_ROOT = _ARES_MAIN_ROOT / "retrieve" / "stage2_results" / "inference"
_GOLD_ROOT = _SFT_DIR / "clean_road_gold"
_DEFAULT_OUTPUT_ROOT = _RL_DIR / "data"

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
RL_ENV_VALID_OUTPUTS = (
    'Valid outputs: {"action":"ADD_TRIPLE","triple_id":<id>} or {"action":"STOP"}.'
)


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
    path = _GOLD_ROOT / dataset / GOLD_FILE_BY_SPLIT[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing gold file for {dataset}/{split}: {path}")
    return path


def default_output_root_for_dataset(dataset: str) -> str:
    return str(_DEFAULT_OUTPUT_ROOT / dataset)


def resolve_remote_answer_service_url(dataset: str, override: Optional[str] = None) -> str:
    env_key = "CWQ_REMOTE_ANSWER_URL" if dataset == "cwq" else "WEBQSP_REMOTE_ANSWER_URL"
    env_value = os.environ.get(env_key)
    if env_value is not None and str(env_value).strip():
        return str(env_value).strip()
    if dataset != "cwq":
        host = os.environ.get("WEBQSP_REMOTE_ANSWER_HOST")
        port = os.environ.get("WEBQSP_REMOTE_ANSWER_PORT")
        if host and port:
            return f"http://{host.strip()}:{str(port).strip()}/answer"
    if override is not None and str(override).strip():
        return str(override).strip()
    return DEFAULT_REMOTE_ANSWER_URL_BY_DATASET.get(str(dataset or "webqsp"), DEFAULT_REMOTE_ANSWER_URL_BY_DATASET["webqsp"])


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_dict(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


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


def count_gold_distinct_triples(gold_sample: Dict[str, Any]) -> int:
    seen: Set[Tuple[str, str, str]] = set()
    for path_payload in gold_sample.get("paths") or []:
        if not isinstance(path_payload, dict):
            continue
        for item in path_payload.get("path_triplets") or []:
            if not isinstance(item, dict):
                continue
            h, r, t = normalize_gold_triplet(item)
            if h and r and t:
                seen.add((h, r, t))
    return len(seen)


def compute_reward_triple_limits(dataset: str, g: int, cap_soft: int, cap_hard: int) -> Tuple[int, int]:
    g = max(0, int(g))
    cap_soft = max(1, int(cap_soft))
    cap_hard = max(1, int(cap_hard))
    if dataset == "webqsp":
        soft = int(min(max(1.25 * g, 22), cap_soft))
        hard = int(min(max(1.8 * g, 40), cap_hard))
    elif dataset == "cwq":
        soft = int(min(max(1.2 * g, 22), cap_soft))
        hard = int(min(max(1.8 * g, 40), cap_hard))
    else:
        soft, hard = cap_soft, cap_hard
    if hard < soft:
        hard = soft
    return soft, hard


def format_candidate_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def format_indexed_candidate_triple_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def resolve_max_selected_triple_limits(ground_truth: Dict[str, Any]) -> Tuple[int, int]:
    dataset = str(ground_truth.get("dataset") or "")
    g = ground_truth.get("gold_distinct_triplet_count")
    cap_soft = ground_truth.get("max_selected_triples_soft_cap")
    cap_hard = ground_truth.get("max_selected_triples_hard_cap")
    if g is not None and dataset in ("webqsp", "cwq"):
        if cap_soft is None or cap_hard is None:
            cap_soft = REWARD_TRIPLE_CAP_SOFT_BY_DATASET.get(dataset, DEFAULT_MAX_SELECTED_TRIPLES_SOFT)
            cap_hard = REWARD_TRIPLE_CAP_HARD_BY_DATASET.get(dataset, DEFAULT_MAX_SELECTED_TRIPLES_HARD)
        return compute_reward_triple_limits(dataset, int(g), int(cap_soft), int(cap_hard))
    return (
        int(ground_truth.get("max_selected_triples_soft", DEFAULT_MAX_SELECTED_TRIPLES_SOFT)),
        int(ground_truth.get("max_selected_triples_hard", DEFAULT_MAX_SELECTED_TRIPLES_HARD)),
    )


def parse_action_sequence(solution_str: str, max_actions: int = 256) -> List[Any]:
    actions: List[Any] = []
    for token in ACTION_PATTERN.findall(solution_str or ""):
        token = token.strip()
        if not token:
            continue
        if token.upper() == DEFAULT_STOP_ACTION:
            actions.append(DEFAULT_STOP_ACTION)
        else:
            try:
                actions.append(int(token))
            except ValueError:
                continue
        if len(actions) >= max_actions:
            break
    return actions


def triple_entities(candidate: Dict[str, Any]) -> set[str]:
    return {candidate["head"], candidate["tail"]}


def normalize_reward_candidate_triples(raw_candidates: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("idx"))
        except (TypeError, ValueError):
            continue
        if "triple" in item:
            triple = item.get("triple")
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            head, relation, tail = safe_text(triple[0]), safe_text(triple[1]), safe_text(triple[2])
        else:
            head = safe_text(item.get("head"))
            relation = safe_text(item.get("relation"))
            tail = safe_text(item.get("tail"))
        if not head or not relation or not tail:
            continue
        out.append(
            {
                "idx": idx,
                "head": head,
                "relation": relation,
                "tail": tail,
                "text": item.get("text")
                or format_candidate_text({"idx": idx, "head": head, "relation": relation, "tail": tail}),
            }
        )
    return out


def is_connected_action(candidate: Dict[str, Any], selected_candidates: Sequence[Dict[str, Any]]) -> bool:
    if not selected_candidates:
        return True
    candidate_entities = triple_entities(candidate)
    for selected in selected_candidates:
        if candidate_entities & triple_entities(selected):
            return True
    return False


def answer_hit_stats(
    selected_indices: Sequence[int],
    candidates_by_idx: Dict[int, Dict[str, Any]],
    answer_entities: Sequence[str],
) -> Tuple[List[str], float]:
    selected_entities: set[str] = set()
    for selected_idx in selected_indices:
        candidate = candidates_by_idx.get(selected_idx)
        if candidate is None:
            continue
        selected_entities |= triple_entities(candidate)
    answer_entities_hit = [entity for entity in answer_entities if entity in selected_entities]
    answer_recall = (len(answer_entities_hit) / len(answer_entities)) if answer_entities else 0.0
    return answer_entities_hit, answer_recall


def connectivity_ratio(
    selected_indices: Sequence[int],
    candidates_by_idx: Dict[int, Dict[str, Any]],
) -> float:
    selected_nodes: set[str] = set()
    adjacency: Dict[str, set[str]] = defaultdict(set)
    for selected_idx in selected_indices:
        candidate = candidates_by_idx.get(selected_idx)
        if candidate is None:
            continue
        head = candidate["head"]
        tail = candidate["tail"]
        selected_nodes.add(head)
        selected_nodes.add(tail)
        adjacency[head].add(tail)
        adjacency[tail].add(head)
    if not selected_nodes:
        return 0.0
    visited: set[str] = set()
    largest_cc = 0
    for node in selected_nodes:
        if node in visited:
            continue
        stack = [node]
        cc_size = 0
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cc_size += 1
            for nxt in adjacency.get(current, set()):
                if nxt not in visited:
                    stack.append(nxt)
        largest_cc = max(largest_cc, cc_size)
    return float(largest_cc) / float(len(selected_nodes))


def support_recall(selected_indices: Sequence[int], support_indices: Sequence[int]) -> float:
    support_set = {
        int(x) for x in support_indices if isinstance(x, int) or (isinstance(x, str) and str(x).isdigit())
    }
    if not support_set:
        return 0.0
    selected_set = {int(x) for x in selected_indices if isinstance(x, int)}
    return float(len(selected_set & support_set)) / float(len(support_set))


def build_covered_facet_ids(
    selected_candidates: Sequence[Dict[str, Any]],
    primary_topic: str,
    answer_entities: Sequence[str],
    question_facets: Sequence[Dict[str, Any]],
) -> List[str]:
    selected_entities: set[str] = set()
    f1_ok = False
    for candidate in selected_candidates:
        head = candidate.get("head", "")
        tail = candidate.get("tail", "")
        selected_entities.add(head)
        selected_entities.add(tail)
        if primary_topic and (head == primary_topic or tail == primary_topic):
            f1_ok = True
    answer_set = {safe_text(x) for x in answer_entities if safe_text(x)}
    all_answers_found = bool(answer_set) and answer_set.issubset(selected_entities)
    out: List[str] = []
    for facet in question_facets:
        facet_id = safe_text(facet.get("facet_id"))
        if not facet_id:
            continue
        if facet_id == "f1":
            if f1_ok:
                out.append(facet_id)
        elif all_answers_found:
            out.append(facet_id)
    return out


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
    topic_entities = normalize_entity_list(trajectory.get("topic_entities"))
    current_nodes: set[str] = set(topic_entities)
    for candidate in candidate_triples:
        if candidate["idx"] in selected_set:
            current_nodes.add(candidate["head"])
            current_nodes.add(candidate["tail"])
    topic_entity = safe_text(trajectory.get("topic_entity")) or (topic_entities[0] if topic_entities else "")
    return {
        "question": trajectory.get("question", ""),
        "topic_entity": topic_entity,
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


def build_agent_observation(
    trajectory: Dict[str, Any],
    selected_indices: Sequence[int],
) -> Dict[str, Any]:
    full_state = build_structured_input(
        trajectory=trajectory,
        selected_indices=selected_indices,
        candidate_triples=trajectory["candidate_triples"],
    )
    return {
        "question": full_state.get("question", ""),
        "topic_entity": full_state.get("topic_entity", ""),
        "state": full_state.get("state", {}),
        "candidate_triples": full_state.get("candidate_triples", []),
        "action_rules": full_state.get("action_rules", {}),
    }


def render_rl_env_messages(
    trajectory: Dict[str, Any],
    selected_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    obs = build_agent_observation(trajectory=trajectory, selected_indices=list(selected_indices or []))
    instruction = (
        "You are acting in an interactive KG environment.\n"
        "Given the current observation, output exactly one JSON action and nothing else.\n"
        f"{RL_ENV_VALID_OUTPUTS}"
    )
    user_content = f"{instruction}\n{json.dumps(obs, ensure_ascii=False)}"
    return [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": user_content},
    ]


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
        if not triples or len(triples) > top_k:
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
    return [
        int(c["idx"])
        for c in sorted(working_candidates, key=lambda x: int(x["idx"]))
        if (c["head"], c["relation"], c["tail"]) in gold_set
    ]


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
        per_path.append({"path_index": i, "gold_path_indices": idxs, "mapping_method": "sample_union"})
    return {
        "valid": True,
        "mismatch_reason": None,
        "candidate_triples": working_candidates,
        "gold_set": gold_set,
        "gold_path_indices_union": gold_path_indices_union,
        "retained_path_indices": retained_indices,
        "budget_reason": budget_reason,
        "per_path_mappings": per_path,
        "answer_entities": filtered_answers,
        "mapping_method": "sample_union_front_back_replace",
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


def collect_trajectories_for_split(
    dataset: str,
    split: str,
    top_k: int = DEFAULT_TOP_K,
    merge_paths_per_sample: bool = True,
    sample_limit: Optional[int] = None,
    min_answer_hits_to_stop: int = DEFAULT_MIN_ANSWER_HITS_TO_STOP,
    min_answer_recall_to_stop: float = DEFAULT_MIN_ANSWER_RECALL_TO_STOP,
    min_connectivity_ratio_to_stop: float = DEFAULT_MIN_CONNECTIVITY_RATIO_TO_STOP,
    min_selected_triples_to_stop: int = DEFAULT_MIN_SELECTED_TRIPLES_TO_STOP,
    max_selected_triples_soft: int = DEFAULT_MAX_SELECTED_TRIPLES_SOFT,
    max_selected_triples_hard: int = DEFAULT_MAX_SELECTED_TRIPLES_HARD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gold_payload = load_json(str(resolve_gold_file(dataset, split)))
    inference_payload = load_torch_dict(str(resolve_inference_file(dataset, split)))
    question_index = build_question_index(inference_payload)

    stats = Counter()
    stats["dataset"] = dataset
    stats["split"] = split
    stats["gold_samples"] = len(gold_payload)
    stats["inference_samples"] = len(inference_payload)
    stats["inference_only_dropped"] = len(set(inference_payload) - set(gold_payload))
    matched_inference_ids: set[str] = set()
    trajectories: List[Dict[str, Any]] = []

    for gold_idx, (gold_sample_id, gold_sample) in enumerate(gold_payload.items()):
        if sample_limit is not None and gold_idx >= sample_limit:
            break

        inferred_sample_id, alignment_method, _alignment_error = resolve_inference_sample_id(
            gold_sample_id=gold_sample_id,
            gold_sample=gold_sample,
            inference_payload=inference_payload,
            question_index=question_index,
        )
        if inferred_sample_id is None:
            stats["gold_missing_inference"] += 1
            continue

        matched_inference_ids.add(inferred_sample_id)
        if alignment_method == "question":
            stats["question_fallback_matches"] += 1
        stats["retained_samples"] += 1

        g_distinct = count_gold_distinct_triples(gold_sample)
        eff_soft, eff_hard = compute_reward_triple_limits(
            dataset, g_distinct, max_selected_triples_soft, max_selected_triples_hard
        )

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
        support_union_indices = list(union_map["gold_path_indices_union"])
        budget_reason = union_map.get("budget_reason") or ""
        stats[f"sample_union_budget::{budget_reason}"] += 1
        candidates_by_idx = {c["idx"]: c for c in working_candidates}

        valid_mappings_for_sample: List[Dict[str, Any]] = []
        for pm in union_map["per_path_mappings"]:
            path_index = int(pm["path_index"])
            path_payload = paths[path_index]
            gold_path_indices = pm["gold_path_indices"]
            gold_path_triples = [
                {
                    "head": candidates_by_idx[idx]["head"],
                    "relation": candidates_by_idx[idx]["relation"],
                    "tail": candidates_by_idx[idx]["tail"],
                    "text": format_candidate_text(
                        {
                            "idx": idx,
                            "head": candidates_by_idx[idx]["head"],
                            "relation": candidates_by_idx[idx]["relation"],
                            "tail": candidates_by_idx[idx]["tail"],
                        }
                    ),
                }
                for idx in gold_path_indices
            ]
            valid_mappings_for_sample.append(
                {
                    "path_index": path_index,
                    "path_payload": path_payload,
                    "mapping": {
                        "gold_path_indices": gold_path_indices,
                        "gold_path_triples": gold_path_triples,
                        "mapping_method": union_map.get("mapping_method", "sample_union_front_back_replace"),
                        "candidate_triples": working_candidates,
                    },
                }
            )

        common_stop_fields = {
            "min_answer_hits_to_stop": min_answer_hits_to_stop,
            "min_answer_recall_to_stop": min_answer_recall_to_stop,
            "min_connectivity_ratio_to_stop": min_connectivity_ratio_to_stop,
            "min_selected_triples_to_stop": min_selected_triples_to_stop,
            "gold_distinct_triplet_count": g_distinct,
            "max_selected_triples_soft": eff_soft,
            "max_selected_triples_hard": eff_hard,
        }

        if (not merge_paths_per_sample) and valid_mappings_for_sample:
            for item in valid_mappings_for_sample:
                path_index = int(item["path_index"])
                path_payload = item["path_payload"]
                mapping = item["mapping"]
                trajectories.append(
                    {
                        "trajectory_id": f"{dataset}:{split}:{gold_sample_id}:path{path_index}",
                        "dataset": dataset,
                        "split": split,
                        "sample_id": gold_sample_id,
                        "inference_sample_id": inferred_sample_id,
                        "alignment_method": alignment_method,
                        "path_index": path_index,
                        "answer_entity": safe_text(path_payload.get("answer_entity")),
                        "question": question,
                        "topic_entity": topic_entity,
                        "topic_entities": topic_entities,
                        "answer_entities": filtered_answer_entities,
                        "candidate_triples": working_candidates,
                        "gold_path_indices": mapping["gold_path_indices"],
                        "gold_path_triples": mapping["gold_path_triples"],
                        "support_union_indices": support_union_indices,
                        "mapping_method": mapping["mapping_method"],
                        "path_found": bool(path_payload.get("path_found", True)),
                        "sample_union_budget": budget_reason,
                        **common_stop_fields,
                    }
                )
                stats["valid_trajectories"] += 1

        if merge_paths_per_sample and valid_mappings_for_sample:
            merged_indices = list(union_map["gold_path_indices_union"])
            merged_gold_path_triples = [
                {
                    "idx": idx,
                    "head": candidates_by_idx[idx]["head"],
                    "relation": candidates_by_idx[idx]["relation"],
                    "tail": candidates_by_idx[idx]["tail"],
                }
                for idx in merged_indices
            ]
            trajectories.append(
                {
                    "trajectory_id": f"{dataset}:{split}:{gold_sample_id}:merged",
                    "dataset": dataset,
                    "split": split,
                    "sample_id": gold_sample_id,
                    "inference_sample_id": inferred_sample_id,
                    "alignment_method": alignment_method,
                    "path_index": -1,
                    "answer_entity": "",
                    "question": question,
                    "topic_entity": topic_entity,
                    "topic_entities": topic_entities,
                    "answer_entities": filtered_answer_entities,
                    "candidate_triples": working_candidates,
                    "gold_path_indices": merged_indices,
                    "gold_path_triples": merged_gold_path_triples,
                    "support_union_indices": merged_indices,
                    "mapping_method": "merged_sample_union",
                    "path_found": True,
                    "merged_from_path_count": len(valid_mappings_for_sample),
                    "merged_from_path_indices": [int(pm["path_index"]) for pm in union_map["per_path_mappings"]],
                    "sample_union_budget": budget_reason,
                    "retained_path_indices": union_map.get("retained_path_indices", []),
                    "max_selected_triples_soft_cap": max_selected_triples_soft,
                    "max_selected_triples_hard_cap": max_selected_triples_hard,
                    **common_stop_fields,
                }
            )
            stats["valid_trajectories"] += 1
            stats["mapping_method::merged_sample_union"] += 1

        stats["retained_samples_with_valid_path"] += 1

    stats["matched_inference_samples"] = len(matched_inference_ids)
    return trajectories, dict(stats)


def _compact_candidate_triples(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    compacted: List[Dict[str, Any]] = []
    for rank, candidate in enumerate((trajectory.get("candidate_triples") or []), start=1):
        head = str(candidate.get("head") or "")
        relation = str(candidate.get("relation") or "")
        tail = str(candidate.get("tail") or "")
        compacted.append(
            {
                "idx": int(candidate["idx"]),
                "triple": [head, relation, tail],
                "head": head,
                "relation": relation,
                "tail": tail,
                "text": candidate.get("text") or f"{int(candidate['idx'])}|{head}|{relation}|{tail}",
                "score": candidate.get("score", candidate.get("final_score", rank)),
                "original_rank": int(candidate.get("original_rank") or rank),
            }
        )
    return compacted


def build_ground_truth(
    trajectory: Dict[str, Any],
    remote_answer_urls_by_dataset: Dict[str, str],
    remote_answer_timeout_sec: float,
) -> Dict[str, Any]:
    dataset = str(trajectory.get("dataset") or "webqsp")
    return {
        "sample_id": trajectory.get("sample_id"),
        "trajectory_id": trajectory.get("trajectory_id"),
        "dataset": dataset,
        "split": trajectory.get("split"),
        "question": trajectory.get("question"),
        "topic_entity": trajectory.get("topic_entity") or "",
        "answer_entities": list(trajectory.get("answer_entities") or []),
        "gold_answers": list(trajectory.get("answer_entities") or []),
        "candidate_triples": _compact_candidate_triples(trajectory),
        "gold_path_indices": list(trajectory.get("gold_path_indices") or []),
        "support_union_indices": list(
            trajectory.get("support_union_indices") or trajectory.get("gold_path_indices") or []
        ),
        "gold_path_triples": list(trajectory.get("gold_path_triples") or []),
        "alignment_method": trajectory.get("alignment_method"),
        "mapping_method": trajectory.get("mapping_method"),
        "path_index": trajectory.get("path_index"),
        "path_found": trajectory.get("path_found"),
        "inference_sample_id": trajectory.get("inference_sample_id"),
        "merged_from_path_indices": list(trajectory.get("merged_from_path_indices") or []),
        "prompt_mode": "scored_200",
        "llm_mode": "sys_icl_dc",
        "eval_mode": "kbqa_scored_200_corrected",
        "score_threshold": 0.0,
        "a_entity_in_graph": bool(trajectory.get("answer_entities") or []),
        "min_answer_hits_to_stop": trajectory.get("min_answer_hits_to_stop", DEFAULT_MIN_ANSWER_HITS_TO_STOP),
        "min_answer_recall_to_stop": trajectory.get("min_answer_recall_to_stop", DEFAULT_MIN_ANSWER_RECALL_TO_STOP),
        "min_connectivity_ratio_to_stop": trajectory.get(
            "min_connectivity_ratio_to_stop", DEFAULT_MIN_CONNECTIVITY_RATIO_TO_STOP
        ),
        "min_selected_triples_to_stop": trajectory.get(
            "min_selected_triples_to_stop", DEFAULT_MIN_SELECTED_TRIPLES_TO_STOP
        ),
        "gold_distinct_triplet_count": trajectory.get("gold_distinct_triplet_count"),
        "max_selected_triples_soft_cap": trajectory.get("max_selected_triples_soft_cap"),
        "max_selected_triples_hard_cap": trajectory.get("max_selected_triples_hard_cap"),
        "max_selected_triples_soft": trajectory.get("max_selected_triples_soft", DEFAULT_MAX_SELECTED_TRIPLES_SOFT),
        "max_selected_triples_hard": trajectory.get("max_selected_triples_hard", DEFAULT_MAX_SELECTED_TRIPLES_HARD),
        "remote_answer_service_url": resolve_remote_answer_service_url(
            dataset=dataset,
            override=remote_answer_urls_by_dataset.get(dataset),
        ),
        "remote_answer_timeout_sec": float(remote_answer_timeout_sec),
    }


def build_rl_record(
    trajectory: Dict[str, Any],
    remote_answer_urls_by_dataset: Dict[str, str],
    remote_answer_timeout_sec: float,
) -> Dict[str, Any]:
    ground_truth = build_ground_truth(
        trajectory=trajectory,
        remote_answer_urls_by_dataset=remote_answer_urls_by_dataset,
        remote_answer_timeout_sec=remote_answer_timeout_sec,
    )
    initial_observation = build_agent_observation(trajectory=trajectory, selected_indices=[])
    return {
        "data_source": DATA_SOURCE,
        "prompt": render_rl_env_messages(trajectory=trajectory, selected_indices=[]),
        "reward_model": {"style": "rule"},
        "extra_info": {
            "interaction_kwargs": {
                "name": DATA_SOURCE,
                "ground_truth": ground_truth,
                "initial_observation": initial_observation,
            }
        },
    }


def build_rl_datasets(
    trajectories_by_split: Dict[str, List[Dict[str, Any]]],
    output_root: str,
    splits: Sequence[str],
    remote_answer_urls_by_dataset: Dict[str, str],
    remote_answer_timeout_sec: float,
) -> Dict[str, Any]:
    rl_dir = Path(output_root)
    rl_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"paths": {}, "stats": {}}

    for split in splits:
        output_path = rl_dir / f"{split}.jsonl"
        stats: Counter = Counter()
        with output_path.open("w", encoding="utf-8") as out_f:
            for trajectory in trajectories_by_split.get(split, []):
                record = build_rl_record(
                    trajectory=trajectory,
                    remote_answer_urls_by_dataset=remote_answer_urls_by_dataset,
                    remote_answer_timeout_sec=remote_answer_timeout_sec,
                )
                append_jsonl_line(out_f, record)
                stats["episodes"] += 1
                stats[f"dataset::{record['extra_info']['interaction_kwargs']['ground_truth'].get('dataset')}"] += 1
        summary["paths"][split] = str(output_path)
        summary["stats"][split] = dict(stats)
    return summary


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
        if split not in DEFAULT_SPLITS:
            raise ValueError(f"sample_limits_json: unknown split {split!r}, expected one of {DEFAULT_SPLITS}")
        limit = int(value)
        if limit < 0:
            raise ValueError(f"sample_limits_json: limit for {split!r} must be >= 0, got {limit}")
        out[split] = limit
    return out


def _resolve_remote_answer_urls_by_dataset(args: argparse.Namespace) -> Dict[str, str]:
    urls = dict(DEFAULT_REMOTE_ANSWER_URL_BY_DATASET)
    if args.remote_answer_service_url_webqsp:
        urls["webqsp"] = args.remote_answer_service_url_webqsp
    if args.remote_answer_service_url_cwq:
        urls["cwq"] = args.remote_answer_service_url_cwq
    return urls


def _resolve_output_roots(args: argparse.Namespace) -> Dict[str, str]:
    if args.output_root:
        if len(args.datasets) != 1:
            raise ValueError("--output_root only supports a single dataset.")
        return {args.datasets[0]: str(Path(args.output_root).resolve())}
    return {dataset: default_output_root_for_dataset(dataset) for dataset in args.datasets}


def build_all(args: argparse.Namespace) -> Dict[str, Any]:
    output_roots_by_dataset = _resolve_output_roots(args)
    sample_limits = parse_sample_limits_json(getattr(args, "sample_limits_json", None))
    remote_answer_urls_by_dataset = _resolve_remote_answer_urls_by_dataset(args)
    dataset_summaries: Dict[str, Any] = {}

    for dataset in args.datasets:
        output_root = output_roots_by_dataset[dataset]
        trajectories_by_split: Dict[str, List[Dict[str, Any]]] = {}
        split_stats: Dict[str, Dict[str, Any]] = {}

        for split in args.splits:
            eff_sample_limit = (
                sample_limits.get(split, args.sample_limit) if sample_limits is not None else args.sample_limit
            )
            trajectories, stats = collect_trajectories_for_split(
                dataset=dataset,
                split=split,
                top_k=args.top_k,
                merge_paths_per_sample=args.merge_paths_per_sample,
                sample_limit=eff_sample_limit,
                min_answer_hits_to_stop=args.min_answer_hits_to_stop,
                min_answer_recall_to_stop=args.min_answer_recall_to_stop,
                min_connectivity_ratio_to_stop=args.min_connectivity_ratio_to_stop,
                min_selected_triples_to_stop=args.min_selected_triples_to_stop,
                max_selected_triples_soft=args.max_selected_triples_soft,
                max_selected_triples_hard=args.max_selected_triples_hard,
            )
            trajectories_by_split[split] = trajectories
            split_stats[split] = stats

        rl_summary = build_rl_datasets(
            trajectories_by_split=trajectories_by_split,
            output_root=output_root,
            splits=args.splits,
            remote_answer_urls_by_dataset=remote_answer_urls_by_dataset,
            remote_answer_timeout_sec=args.remote_answer_timeout_sec,
        )
        dataset_summaries[dataset] = {
            "dataset": dataset,
            "output_root": output_root,
            "split_stats": split_stats,
            "rl_summary": rl_summary,
            "data_source": DATA_SOURCE,
        }

    return {
        "datasets": list(args.datasets),
        "output_roots_by_dataset": output_roots_by_dataset,
        "dataset_summaries": dataset_summaries,
        "data_source": DATA_SOURCE,
        "remote_answer_urls_by_dataset": remote_answer_urls_by_dataset,
        "remote_answer_timeout_sec": float(args.remote_answer_timeout_sec),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retriever GRPO train/val.jsonl only (no Question Explain)."
    )
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Optional override when building exactly one dataset. Default: rl/data/{dataset}.",
    )
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--merge_paths_per_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--sample_limits_json", type=str, default=None)
    parser.add_argument("--min_answer_hits_to_stop", type=int, default=DEFAULT_MIN_ANSWER_HITS_TO_STOP)
    parser.add_argument("--min_answer_recall_to_stop", type=float, default=DEFAULT_MIN_ANSWER_RECALL_TO_STOP)
    parser.add_argument(
        "--min_connectivity_ratio_to_stop",
        type=float,
        default=DEFAULT_MIN_CONNECTIVITY_RATIO_TO_STOP,
    )
    parser.add_argument("--min_selected_triples_to_stop", type=int, default=DEFAULT_MIN_SELECTED_TRIPLES_TO_STOP)
    parser.add_argument("--max_selected_triples_soft", type=int, default=DEFAULT_MAX_SELECTED_TRIPLES_SOFT)
    parser.add_argument("--max_selected_triples_hard", type=int, default=DEFAULT_MAX_SELECTED_TRIPLES_HARD)
    parser.add_argument("--remote_answer_service_url_webqsp", type=str, default=None)
    parser.add_argument("--remote_answer_service_url_cwq", type=str, default=None)
    parser.add_argument("--remote_answer_timeout_sec", type=float, default=DEFAULT_REMOTE_TIMEOUT_SEC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_all(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
