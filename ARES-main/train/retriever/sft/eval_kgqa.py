#!/usr/bin/env python3
"""KGQA agent inference (SFT prompt, no question_explain).

Reads sub100-style JSON (question, topic_entities, triplets), runs a trained agent model
step-by-step to select a triple sequence, then exports retrieval_result*.pth.

  eval_data/{dataset}/sub100{split}.json
  kgqa_data/{dataset}/inference_result.json
  kgqa_data/{dataset}/retrieval_result_{split}.pth
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_ARES_MAIN_ROOT = _SCRIPT_DIR.parents[2]

DEFAULT_STOP_ACTION = "STOP"
ACTION_PATTERN = re.compile(r"\bSTOP\b|\b\d{1,3}\b", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile("<" + "think" + r">[\s\S]*?" + "<" + "/think" + ">", re.DOTALL)

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


def infer_split_from_subgraphs_path(path: Union[str, Path]) -> str:
    name = Path(path).name.lower()
    if "train" in name:
        return "train"
    if "val" in name:
        return "val"
    return "test"


def retrieval_result_filename(split: str) -> str:
    return "retrieval_result.pth" if split == "test" else f"retrieval_result_{split}.pth"


def default_source_retrieval_path(dataset: str, split: str) -> Path:
    return _ARES_MAIN_ROOT / "retrieve" / "stage2_results" / "inference" / dataset / retrieval_result_filename(split)


def default_export_retrieval_path(dataset: str, split: str) -> Path:
    return _SCRIPT_DIR / "kgqa_data" / dataset / retrieval_result_filename(split)


def default_paths(dataset: str, split: str = "train") -> dict[str, Path]:
    split_suffix = "" if split == "test" else split
    subgraph_name = "sub100.json" if split == "test" else f"sub100{split_suffix}.json"
    return {
        "subgraphs_json": _SCRIPT_DIR / "eval_data" / dataset / subgraph_name,
        "out_json": _SCRIPT_DIR / "kgqa_data" / dataset / "inference_result.json",
        "out_retrieval_pth": default_export_retrieval_path(dataset, split),
    }


def resolve_paths(dataset: str, subgraphs_json: Optional[str]) -> dict[str, Path]:
    split = infer_split_from_subgraphs_path(subgraphs_json or default_paths(dataset)["subgraphs_json"])
    paths = default_paths(dataset, split)
    if subgraphs_json:
        paths["subgraphs_json"] = Path(subgraphs_json)
    return paths


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


def format_candidate_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def format_candidate_triple_text(candidate: Dict[str, Any]) -> str:
    return f"({candidate['head']}, {candidate['relation']}, {candidate['tail']})"


def format_indexed_candidate_triple_text(candidate: Dict[str, Any]) -> str:
    return f"{candidate['idx']}|{candidate['head']}|{candidate['relation']}|{candidate['tail']}"


def build_candidate_triples(sample: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    scored_triples = sample.get("scored_triples") or []
    seen_triples: Set[Tuple[str, str, str]] = set()
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


def build_structured_input(
    trajectory: Dict[str, Any],
    selected_indices: Sequence[int],
    candidate_triples: Sequence[Dict[str, Any]],
    question_explain: Optional[Dict[str, Any]] = None,
    omit_answer_side_info: bool = False,
) -> Dict[str, Any]:
    del question_explain, omit_answer_side_info
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
    current_nodes: Set[str] = set(trajectory.get("topic_entities") or [])
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


def record_to_trajectory(record: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    sid = str(record.get("id") or "")
    triples_field = record.get("triplets") or record.get("triplet") or []
    candidate_triples = build_candidate_triples({"scored_triples": triples_field}, top_k=top_k)
    topic_entities = normalize_entity_list(record.get("topic_entities"))
    return {
        "trajectory_id": sid,
        "dataset": record.get("dataset") or "webqsp",
        "question": normalize_question(record.get("question")),
        "topic_entity": topic_entities[0] if topic_entities else "",
        "topic_entities": topic_entities,
        "candidate_triples": candidate_triples,
    }


def strip_qwen_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


def parse_first_action(decoded: str, candidates_by_idx: Dict[int, Dict[str, Any]]) -> Any:
    text = strip_qwen_think(decoded)
    for tok in ACTION_PATTERN.findall(text):
        if str(tok).upper() == DEFAULT_STOP_ACTION:
            return DEFAULT_STOP_ACTION
        try:
            idx = int(tok)
        except (TypeError, ValueError):
            continue
        if idx in candidates_by_idx:
            return idx
    return None


def _normalize_action_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _canonical_triple_text(text: str) -> str:
    s = _normalize_action_text(text)
    s = re.sub(r"^\[\d+\]\s*", "", s)
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        if len(parts) == 4 and parts[0].isdigit():
            _, h, r, t = parts
            return f"({h}, {r}, {t})"
        if len(parts) == 3:
            h, r, t = parts
            return f"({h}, {r}, {t})"
    return s


def _triple_text_lookup(candidates_by_idx: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for idx in sorted(candidates_by_idx.keys()):
        cand = candidates_by_idx[idx]
        lookup.setdefault(_canonical_triple_text(format_candidate_triple_text(cand)), idx)
        lookup.setdefault(_canonical_triple_text(format_indexed_candidate_triple_text(cand)), idx)
    return lookup


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_sft_v1_action(decoded: str, candidates_by_idx: Dict[int, Dict[str, Any]]) -> Any:
    text = strip_qwen_think(decoded)
    obj = extract_first_json_object(text)
    if isinstance(obj, dict):
        raw_action = obj.get("action")
        if isinstance(raw_action, str):
            act = raw_action.strip().upper()
            if act == "STOP":
                return DEFAULT_STOP_ACTION
            if act == "ADD_TRIPLE":
                triple_id_raw = obj.get("triple_id")
                if isinstance(triple_id_raw, int) and triple_id_raw in candidates_by_idx:
                    return triple_id_raw
                if isinstance(triple_id_raw, str):
                    try:
                        triple_id = int(triple_id_raw.strip())
                    except ValueError:
                        triple_id = None
                    if triple_id is not None and triple_id in candidates_by_idx:
                        return triple_id
                triple_raw = obj.get("triple")
                if isinstance(triple_raw, str) and triple_raw.strip():
                    lookup = _triple_text_lookup(candidates_by_idx)
                    candidate_text = triple_raw.strip()
                    normalized = _canonical_triple_text(candidate_text)
                    if normalized in lookup:
                        return lookup[normalized]
                    m_paren = re.search(r"\([^()]+\)$", candidate_text)
                    if m_paren:
                        np = _canonical_triple_text(m_paren.group(0))
                        if np in lookup:
                            return lookup[np]
    return parse_first_action(decoded, candidates_by_idx)


def _coerce_json_triple_id(obj: Dict[str, Any]) -> Optional[int]:
    triple_id_raw = obj.get("triple_id")
    if isinstance(triple_id_raw, bool):
        return None
    if isinstance(triple_id_raw, int):
        return triple_id_raw
    if isinstance(triple_id_raw, str) and triple_id_raw.strip().isdigit():
        return int(triple_id_raw.strip())
    return None


def parse_oob_add_triple_as_stop(decoded: str, candidates_by_idx: Dict[int, Dict[str, Any]]) -> Optional[int]:
    text = strip_qwen_think(decoded)
    obj = extract_first_json_object(text)
    if not isinstance(obj, dict):
        return None
    raw_action = obj.get("action")
    if not isinstance(raw_action, str) or raw_action.strip().upper() != "ADD_TRIPLE":
        return None
    triple_id = _coerce_json_triple_id(obj)
    if triple_id is None or triple_id in candidates_by_idx:
        return None
    return triple_id


def parse_indexed_action(decoded: str, candidates_by_idx: Dict[int, Dict[str, Any]]) -> Optional[Any]:
    text = strip_qwen_think(decoded).strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    if first_line.upper() == DEFAULT_STOP_ACTION:
        return DEFAULT_STOP_ACTION
    m = re.match(r"^\[(\d+)\]\s*\(", first_line)
    if not m:
        return None
    try:
        idx = int(m.group(1))
    except ValueError:
        return None
    return idx if idx in candidates_by_idx else None


@dataclass
class SampleRuntime:
    sid: str
    trajectory: Dict[str, Any]
    candidates_by_idx: Dict[int, Dict[str, Any]]
    candidate_text_map: Dict[int, str]
    prompt_cache: Dict[Tuple[int, ...], str] = field(default_factory=dict)


@dataclass
class EpisodeState:
    runtime: SampleRuntime
    rollout_idx: int = 0
    selected: List[int] = field(default_factory=list)
    selected_set: Set[int] = field(default_factory=set)
    lines: List[str] = field(default_factory=list)
    duplicate_action_count: int = 0
    stalled_steps: int = 0
    terminated_by_stall: bool = False
    terminated_by_stop: bool = False
    done: bool = False
    step_calls: int = 0
    done_reason: str = ""
    parse_none_raw_output: Optional[str] = None


def build_prompt_cached(tokenizer: Any, ep: EpisodeState) -> str:
    key = tuple(ep.selected)
    cached = ep.runtime.prompt_cache.get(key)
    if cached is not None:
        return cached
    payload = build_structured_input(
        trajectory=ep.runtime.trajectory,
        selected_indices=ep.selected,
        candidate_triples=ep.runtime.trajectory["candidate_triples"],
    )
    user_body = render_sft_structured_prompt(payload)
    messages = [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": f"{SFT_INSTRUCTION}\n{user_body}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ep.runtime.prompt_cache[key] = prompt
    return prompt


def _episode_selected_triplets(ep: EpisodeState) -> List[List[str]]:
    rt = ep.runtime
    return [
        [rt.candidates_by_idx[i]["head"], rt.candidates_by_idx[i]["relation"], rt.candidates_by_idx[i]["tail"]]
        for i in ep.selected
        if i in rt.candidates_by_idx
    ]


def _rollout_debug_row(ep: EpisodeState) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "rollout_idx": ep.rollout_idx,
        "selected_indices": list(ep.selected),
        "selected_triplets": _episode_selected_triplets(ep),
        "action_sequence": list(ep.lines),
        "n_steps": ep.step_calls,
        "done_reason": ep.done_reason,
        "terminated_by_stop": ep.terminated_by_stop,
        "duplicate_action_count": ep.duplicate_action_count,
    }
    if ep.done_reason == "parse_none" and ep.parse_none_raw_output is not None:
        row["parse_none_raw_output"] = ep.parse_none_raw_output
    return row


def union_rollout_outputs(eps: Sequence[EpisodeState]) -> Tuple[List[int], List[List[str]], List[Dict[str, Any]]]:
    union_indices: List[int] = []
    union_idx_set: Set[int] = set()
    rollout_rows: List[Dict[str, Any]] = []
    for ep in sorted(eps, key=lambda e: e.rollout_idx):
        rollout_rows.append(_rollout_debug_row(ep))
        for idx in ep.selected:
            if idx not in union_idx_set:
                union_idx_set.add(idx)
                union_indices.append(idx)
    rt = eps[0].runtime if eps else None
    union_triplets: List[List[str]] = []
    if rt is not None:
        union_triplets = [
            [rt.candidates_by_idx[i]["head"], rt.candidates_by_idx[i]["relation"], rt.candidates_by_idx[i]["tail"]]
            for i in union_indices
            if i in rt.candidates_by_idx
        ]
    return union_indices, union_triplets, rollout_rows


def _build_vllm_llm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": args.model_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": 8192,
    }
    try:
        from vllm import LLM

        sig = inspect.signature(LLM.__init__)
        params = sig.parameters
        if args.num_workers > 0 and "max_num_seqs" in params:
            llm_kwargs["max_num_seqs"] = args.num_workers
        if args.enable_prefix_caching and "enable_prefix_caching" in params:
            llm_kwargs["enable_prefix_caching"] = True
    except (TypeError, ValueError, ImportError):
        pass
    if args.num_workers > 0 and "max_num_seqs" not in llm_kwargs:
        llm_kwargs["max_num_seqs"] = args.num_workers
    if args.enable_prefix_caching and "enable_prefix_caching" not in llm_kwargs:
        llm_kwargs["enable_prefix_caching"] = True
    return llm_kwargs


def _triple_key(triple: Sequence[Any]) -> Tuple[str, str, str]:
    return safe_text(triple[0]), safe_text(triple[1]), safe_text(triple[2])


def _lookup_original_triple(
    key: Tuple[str, str, str],
    original_scored: Sequence[Sequence[Any]],
) -> Optional[Sequence[Any]]:
    for triple in original_scored:
        if len(triple) >= 3 and _triple_key(triple) == key:
            return triple
    return None


def _dedupe_triple_keys_ordered(triple_lists: Sequence[Sequence[Any]]) -> List[Tuple[str, str, str]]:
    seen: Set[Tuple[str, str, str]] = set()
    ordered: List[Tuple[str, str, str]] = []
    for raw in triple_lists:
        if len(raw) < 3:
            continue
        key = _triple_key(raw)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def build_scored_triples_from_selected_triplets(
    selected_triplets: Sequence[Sequence[Any]],
    original_scored: Sequence[Sequence[Any]],
    top_k: int = 200,
) -> Tuple[List[Tuple[Any, ...]], List[float], List[float], List[float], List[float]]:
    selected_keys = _dedupe_triple_keys_ordered(selected_triplets)
    selected_set = set(selected_keys)

    pool_keys = _dedupe_triple_keys_ordered(original_scored)

    final_keys: List[Tuple[str, str, str]] = []
    seen_final: Set[Tuple[str, str, str]] = set()
    for key in selected_keys:
        if key not in seen_final:
            seen_final.add(key)
            final_keys.append(key)
    for key in pool_keys:
        if key in seen_final:
            continue
        seen_final.add(key)
        final_keys.append(key)
        if len(final_keys) >= top_k:
            break
    final_keys = final_keys[:top_k]

    scored: List[Tuple[Any, ...]] = []
    for rank, key in enumerate(final_keys):
        orig = _lookup_original_triple(key, original_scored)
        hop = orig[3] if orig is not None and len(orig) >= 4 else 1
        if key in selected_set:
            score = max(1.0 - rank * 0.001, 0.9)
        elif orig is not None and len(orig) >= 5 and orig[4] is not None:
            score = float(orig[4])
        else:
            score = max(0.5 - 0.001 * len(scored), 0.0)
        scored.append((key[0], key[1], key[2], hop, float(score)))

    final_scores = [float(t[4]) for t in scored]
    return scored, final_scores, list(final_scores), list(final_scores), list(final_scores)


def export_retrieval_result_from_inference(
    inference_json_path: str,
    subgraphs_json_path: str,
    dataset: str,
    out_pth_path: Optional[str] = None,
    source_retrieval_path: Optional[str] = None,
    export_top_k: int = 200,
) -> Path:
    split = infer_split_from_subgraphs_path(subgraphs_json_path)
    out_pth = Path(out_pth_path) if out_pth_path else default_export_retrieval_path(dataset, split)
    source_path = Path(source_retrieval_path) if source_retrieval_path else default_source_retrieval_path(dataset, split)

    source_dict: Dict[str, Any] = {}
    if source_path.is_file():
        loaded = torch.load(source_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            source_dict = loaded

    with open(inference_json_path, encoding="utf-8") as f:
        inference_data = json.load(f)
    with open(subgraphs_json_path, encoding="utf-8") as f:
        subgraphs = json.load(f)
    if not isinstance(subgraphs, dict):
        raise TypeError(f"Expected dict in {subgraphs_json_path}")

    inference_by_id: Dict[str, Dict[str, Any]] = {}
    for row in inference_data.get("per_sample") or []:
        if isinstance(row, dict) and row.get("id") is not None:
            inference_by_id[str(row["id"])] = row

    out_dict: Dict[str, Dict[str, Any]] = {}
    for sid, inference_row in sorted(inference_by_id.items(), key=lambda kv: kv[0]):
        record = subgraphs.get(sid)
        if record is None:
            for key, candidate in subgraphs.items():
                if isinstance(candidate, dict) and str(candidate.get("id") or key) == sid:
                    record = candidate
                    break
        if not isinstance(record, dict):
            continue

        selected_triplets = inference_row.get("selected_triplets") or []
        orig_sample = source_dict.get(sid, {})
        orig_scored = orig_sample.get("scored_triples") or []

        scored, final_scores, stage1_scores, stage2_scores, stage2_logits = build_scored_triples_from_selected_triplets(
            selected_triplets,
            orig_scored,
            top_k=export_top_k,
        )
        topic_entities = normalize_entity_list(record.get("topic_entities") or orig_sample.get("q_entity") or [])
        answer_entities = normalize_entity_list(
            record.get("answer_entities")
            or orig_sample.get("a_entity")
            or orig_sample.get("a_entity_in_graph")
            or []
        )

        out_dict[sid] = {
            "question": normalize_question(record.get("question") or orig_sample.get("question")),
            "scored_triples": scored,
            "q_entity": orig_sample.get("q_entity") or topic_entities,
            "q_entity_in_graph": orig_sample.get("q_entity_in_graph") or topic_entities,
            "a_entity": orig_sample.get("a_entity") or answer_entities,
            "a_entity_in_graph": orig_sample.get("a_entity_in_graph") or answer_entities,
            "max_path_length": int(orig_sample.get("max_path_length") or 1),
            "target_relevant_triples": list(orig_sample.get("target_relevant_triples") or []),
            "stage1_scores": stage1_scores,
            "stage2_scores": stage2_scores,
            "stage2_logits": stage2_logits,
            "final_scores": final_scores,
            "candidate_triples_original_order": scored,
        }

    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, out_pth)
    print(
        f"Wrote retrieval_result ({len(out_dict)} samples, top_k={export_top_k}, "
        f"reordered by selected_triplets) -> {out_pth}"
    )
    return out_pth


def merge_inference_shard_jsons(shard_paths: Sequence[str], out_path: str) -> int:
    merged: Dict[str, Dict[str, Any]] = {}
    summary_meta: Dict[str, Any] = {}
    for shard_path in shard_paths:
        with open(shard_path, encoding="utf-8") as f:
            blob = json.load(f)
        if not summary_meta and isinstance(blob.get("summary"), dict):
            summary_meta = dict(blob["summary"])
        for row in blob.get("per_sample") or []:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            sid = str(row["id"])
            if sid in merged:
                raise SystemExit(f"Duplicate sample id while merging shards: {sid}")
            merged[sid] = row
    out = {
        "summary": {
            **summary_meta,
            "n_samples": len(merged),
            "num_shards": len(shard_paths),
        },
        "per_sample": list(merged.values()),
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Merged {len(shard_paths)} shards ({len(merged)} samples) -> {out_file}")
    return len(merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KGQA agent inference (no question_explain).")
    parser.add_argument("--dataset", type=str, default="webqsp", choices=("webqsp", "cwq"))
    parser.add_argument("--subgraphs_json", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--max_active_batch_size", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--rollout_temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_stalled_steps", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", dest="enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--disable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--out_retrieval_pth", type=str, default=None)
    parser.add_argument("--source_retrieval_pth", type=str, default=None)
    parser.add_argument(
        "--export_retrieval_pth_from",
        type=str,
        default=None,
        help="Convert merged inference JSON to retrieval_result*.pth",
    )
    parser.add_argument("--export_top_k", type=int, default=200, help="Max deduplicated triples per sample in exported pth")
    parser.add_argument("--merge_shards", nargs="+", default=None, help="Merge shard JSON files into --out_json")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    paths = resolve_paths(args.dataset, args.subgraphs_json)
    if args.subgraphs_json is None:
        args.subgraphs_json = str(paths["subgraphs_json"])
    if args.out_json is None:
        args.out_json = str(paths["out_json"])
    if args.out_retrieval_pth is None:
        args.out_retrieval_pth = str(paths["out_retrieval_pth"])
    return args


def main() -> None:
    args = parse_args()

    if args.merge_shards:
        if not args.out_json:
            raise SystemExit("error: --out_json is required with --merge_shards")
        merge_inference_shard_jsons(args.merge_shards, args.out_json)
        return

    if args.export_retrieval_pth_from:
        export_retrieval_result_from_inference(
            inference_json_path=args.export_retrieval_pth_from,
            subgraphs_json_path=args.subgraphs_json,
            dataset=args.dataset,
            out_pth_path=args.out_retrieval_pth,
            source_retrieval_path=args.source_retrieval_pth,
            export_top_k=args.export_top_k,
        )
        return

    if not args.model_path:
        raise SystemExit("error: --model_path is required unless --export_retrieval_pth_from or --merge_shards is set")

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("Invalid shard config")
    if args.n_rollouts < 1:
        raise SystemExit("--n_rollouts must be >= 1")
    if args.max_active_batch_size < 0:
        raise SystemExit("--max_active_batch_size must be >= 0")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")

    with open(args.subgraphs_json, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError("Expected top-level JSON object id -> record")

    items: List[Tuple[str, Dict[str, Any]]] = sorted(payload.items(), key=lambda kv: str(kv[0]))
    if args.max_samples is not None:
        items = items[: args.max_samples]
    if args.num_shards > 1:
        items = [items[i] for i in range(len(items)) if i % args.num_shards == args.shard_index]

    out_path = Path(args.out_json)
    existing_by_id: Dict[str, Dict[str, Any]] = {}
    if args.resume and out_path.is_file():
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            for row in prev.get("per_sample") or []:
                if isinstance(row, dict) and row.get("id") is not None:
                    existing_by_id[str(row["id"])] = row
        except (json.JSONDecodeError, OSError) as e:
            print(f"Resume: could not read {out_path}: {e}", file=sys.stderr)

    from vllm import LLM, SamplingParams

    temp = args.rollout_temperature
    if temp is None:
        temp = 0.0 if args.n_rollouts <= 1 else 0.7

    llm_kwargs = _build_vllm_llm_kwargs(args)
    print(
        f"vLLM max_num_seqs={llm_kwargs.get('max_num_seqs', 'default')}; "
        f"enable_prefix_caching={llm_kwargs.get('enable_prefix_caching', False)}",
        flush=True,
    )
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=temp, top_p=args.top_p)

    runtimes: Dict[str, SampleRuntime] = {}
    rows_out: List[Dict[str, Any]] = []

    for key, record in items:
        if not isinstance(record, dict):
            continue
        sid = str(record.get("id") or key)
        if args.resume and sid in existing_by_id:
            rows_out.append(existing_by_id[sid])
            continue
        traj = record_to_trajectory(record, args.top_k)
        if not traj["candidate_triples"]:
            rows_out.append({"id": sid, "skip_reason": "no_triplets", "selected_indices": [], "selected_triplets": []})
            continue
        candidates_by_idx = {c["idx"]: c for c in traj["candidate_triples"]}
        runtimes[sid] = SampleRuntime(
            sid=sid,
            trajectory=traj,
            candidates_by_idx=candidates_by_idx,
            candidate_text_map={idx: format_indexed_candidate_triple_text(c) for idx, c in candidates_by_idx.items()},
        )

    episodes: List[EpisodeState] = []
    by_sid_eps: Dict[str, List[EpisodeState]] = {}
    for sid, rt in runtimes.items():
        eps: List[EpisodeState] = []
        for r in range(args.n_rollouts):
            ep = EpisodeState(runtime=rt, rollout_idx=r)
            eps.append(ep)
            episodes.append(ep)
        by_sid_eps[sid] = eps

    pbar = tqdm(total=len(episodes), desc=f"kgqa_infer[{args.shard_index}/{args.num_shards}]", unit="episode")
    for _ in range(args.max_steps):
        active = [ep for ep in episodes if not ep.done]
        if not active:
            break
        chunk_size = args.max_active_batch_size if args.max_active_batch_size > 0 else len(active)
        for start in range(0, len(active), chunk_size):
            chunk = active[start : start + chunk_size]
            prompts = [build_prompt_cached(tokenizer, ep) for ep in chunk]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            for ep, out in zip(chunk, outputs):
                ep.step_calls += 1
                decoded = out.outputs[0].text
                action = parse_indexed_action(decoded, ep.runtime.candidates_by_idx)
                if action is None:
                    action = parse_sft_v1_action(decoded, ep.runtime.candidates_by_idx)
                if action is None and parse_oob_add_triple_as_stop(decoded, ep.runtime.candidates_by_idx) is not None:
                    action = DEFAULT_STOP_ACTION

                if action is None:
                    ep.parse_none_raw_output = decoded
                    ep.done, ep.done_reason = True, "parse_none"
                elif action == DEFAULT_STOP_ACTION:
                    ep.lines.append(DEFAULT_STOP_ACTION)
                    ep.terminated_by_stop = True
                    ep.done, ep.done_reason = True, "stop"
                elif action not in ep.runtime.candidates_by_idx:
                    ep.lines.append(str(action))
                    ep.done, ep.done_reason = True, "invalid_action"
                elif action in ep.selected_set:
                    ep.duplicate_action_count += 1
                    ep.stalled_steps += 1
                    if args.max_stalled_steps >= 0 and ep.stalled_steps >= args.max_stalled_steps:
                        ep.terminated_by_stall = True
                        ep.done, ep.done_reason = True, "stalled_after_duplicate"
                else:
                    ep.lines.append(str(action))
                    ep.selected.append(action)
                    ep.selected_set.add(action)
                    ep.stalled_steps = 0

                if ep.done:
                    pbar.update(1)

    for ep in episodes:
        if not ep.done:
            ep.done, ep.done_reason = True, "max_steps"
            pbar.update(1)
    pbar.close()

    for sid, eps in sorted(by_sid_eps.items(), key=lambda kv: kv[0]):
        rt = runtimes[sid]
        union_indices, union_triplets, rollout_rows = union_rollout_outputs(eps)
        first = eps[0] if eps else None
        row: Dict[str, Any] = {
            "id": sid,
            "question": rt.trajectory.get("question", ""),
            "topic_entities": rt.trajectory.get("topic_entities") or [],
            "n_rollouts": args.n_rollouts,
            "rollout_temperature": temp,
            "selected_indices": union_indices,
            "selected_triplets": union_triplets,
            "union_n_selected_triples": len(union_indices),
            "rollouts": rollout_rows,
        }
        if first is not None:
            row["action_sequence"] = list(first.lines)
            row["n_steps"] = first.step_calls
            row["done_reason"] = first.done_reason
            row["terminated_by_stop"] = first.terminated_by_stop
            row["duplicate_action_count"] = first.duplicate_action_count
            if first.done_reason == "parse_none" and first.parse_none_raw_output is not None:
                row["parse_none_raw_output"] = first.parse_none_raw_output
        rows_out.append(row)

    summary: Dict[str, Any] = {
        "subgraphs_json": str(Path(args.subgraphs_json).resolve()),
        "model_path": args.model_path,
        "prompt_format": "sft_v1_json",
        "n_rollouts": args.n_rollouts,
        "rollout_temperature": temp,
        "max_active_batch_size": args.max_active_batch_size,
        "num_workers": args.num_workers,
        "enable_prefix_caching": args.enable_prefix_caching,
        "top_k": args.top_k,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "resume": args.resume,
        "n_samples": len(rows_out),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": rows_out}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
