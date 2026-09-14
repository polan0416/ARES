from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_BASE_MODULE = None

STEP_WEIGHT = 0.05
STOP_WEIGHT = 0.05
F1_WEIGHT = 0.8
CONNECT_WEIGHT = 0.05
FORMAT_WEIGHT = 0.05

INVALID_ACTION_REWARD = -2.0
DISCONNECTED_ACTION_REWARD = -1.5

DELTA_SUPPORT_WEIGHT = 1.5
DELTA_ANSWER_WEIGHT = 1.5
DELTA_CONNECTIVITY_WEIGHT = 0.2
NEW_ENTITY_GAIN_WEIGHT = 0.05

NO_GAIN_CONNECTED_PENALTY = 0.20
SAME_HEAD_REL_PENALTY = 0.08

_THINK_BLOCK_RE = re.compile("<" + "think" + r">[\s\S]*?" + "<" + "/think" + ">", re.DOTALL)
_ACTION_TOKEN_RE = re.compile(r"\bSTOP\b|\b\d+\b", re.IGNORECASE)
_ROLE_MARKERS = {"assistant", "user", "system", "tool"}


def _load_base_module():
    global _BASE_MODULE
    if _BASE_MODULE is not None:
        return _BASE_MODULE
    module_path = Path(__file__).resolve().parent / "build_grpo_data.py"
    spec = importlib.util.spec_from_file_location("webqsp_rl_base_builder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load base builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BASE_MODULE = module
    return module


BASE = _load_base_module()
DEFAULT_STOP_ACTION = BASE.DEFAULT_STOP_ACTION


def _safe_text(value: Any) -> str:
    return BASE.safe_text(value)


def _normalize_entity_list(value: Any) -> List[str]:
    return list(BASE.normalize_entity_list(value))


def _strip_think(text: str) -> str:
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


ADD_TRIPLE_ACTION = "ADD_TRIPLE"
_STRICT_JSON_SOURCES = frozenset({"json_action", "json_action_sft"})


def _coerce_triple_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _normalize_action_value(value: Any) -> Any | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.upper() == DEFAULT_STOP_ACTION:
            return DEFAULT_STOP_ACTION
        if token.upper() == ADD_TRIPLE_ACTION:
            return None
        if token.isdigit():
            return int(token)
    return None


def _parse_action_from_json_object(obj: Dict[str, Any]) -> tuple[Any | None, str]:
    """Parse SFT-style or compact JSON actions. Returns (action, source_tag)."""
    if not isinstance(obj, dict):
        return None, ""

    keys = {str(key) for key in obj.keys()}
    action_raw = obj.get("action")

    if action_raw is not None:
        if isinstance(action_raw, str) and action_raw.strip().upper() == DEFAULT_STOP_ACTION:
            source = "json_action" if keys <= {"action"} else "json_action_fallback"
            return DEFAULT_STOP_ACTION, source
        if isinstance(action_raw, str) and action_raw.strip().upper() == ADD_TRIPLE_ACTION:
            triple_id = _coerce_triple_id(obj.get("triple_id"))
            if triple_id is not None:
                source = "json_action_sft" if keys <= {"action", "triple_id"} else "json_action_sft_fallback"
                return triple_id, source
        normalized = _normalize_action_value(action_raw)
        if normalized is not None:
            source = "json_action" if keys <= {"action"} else "json_action_fallback"
            return normalized, source

    triple_id = _coerce_triple_id(obj.get("triple_id"))
    if triple_id is not None and "action" not in obj:
        source = "json_action_sft" if keys <= {"triple_id"} else "json_action_sft_fallback"
        return triple_id, source

    return None, ""


def _extract_first_json_object(text: str) -> tuple[Any | None, int | None, int | None]:
    clean = text or ""
    decoder = json.JSONDecoder()
    for start, ch in enumerate(clean):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(clean, start)
        except json.JSONDecodeError:
            continue
        return obj, start, end
    return None, None, None


def _parse_segment_action(
    segment: str,
    allow_loose_json: bool,
    ignore_non_action_json: bool = False,
) -> tuple[Any | None, str, str]:
    clean = _strip_think(segment)
    if not clean:
        return None, "", ""

    obj, start, end = _extract_first_json_object(clean)
    if isinstance(obj, dict):
        action, source = _parse_action_from_json_object(obj)
        if action is not None and start is not None and end is not None:
            segment_text = clean[start:end].strip()
            has_outer_text = bool(clean[:start].strip() or clean[end:].strip())
            if source in _STRICT_JSON_SOURCES and not has_outer_text:
                return action, source, segment_text
            if allow_loose_json:
                fallback_source = source if source.endswith("_fallback") else f"{source or 'json_action'}_fallback"
                return action, fallback_source, clean
        elif ignore_non_action_json and start == 0 and end == len(clean):
            return None, "", ""

    token_match = _ACTION_TOKEN_RE.search(clean)
    if token_match is None:
        return None, "", ""
    action = _normalize_action_value(token_match.group(0))
    if action is None:
        return None, "", ""
    return action, "token_fallback", token_match.group(0).strip()


def _extract_assistant_segments(solution_str: str) -> tuple[List[str], bool]:
    text = solution_str or ""
    if not text.strip():
        return [], False

    segments: List[str] = []
    current_role: str | None = None
    current_lines: List[str] = []
    saw_role_markers = False

    def flush() -> None:
        nonlocal current_lines
        if current_role == "assistant":
            segment = "\n".join(current_lines).strip()
            if segment:
                segments.append(segment)
        current_lines = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if lowered in _ROLE_MARKERS:
            saw_role_markers = True
            flush()
            current_role = lowered
            continue
        current_lines.append(raw_line)
    flush()
    return segments, saw_role_markers


def extract_action_trace(solution_str: str, max_actions: int = 256) -> Dict[str, Any]:
    text = solution_str or ""
    actions: List[Any] = []
    action_segments: List[str] = []
    action_sources: List[str] = []
    assistant_segments, saw_role_markers = _extract_assistant_segments(text)

    if saw_role_markers:
        for segment in assistant_segments:
            action, source, parsed_segment = _parse_segment_action(
                segment,
                allow_loose_json=True,
                ignore_non_action_json=True,
            )
            if action is None:
                continue
            actions.append(action)
            action_segments.append(parsed_segment or _safe_text(segment))
            action_sources.append(source)
            if len(actions) >= max_actions:
                break
        return {
            "actions": actions,
            "action_segments": action_segments,
            "action_sources": action_sources,
            "used_role_markers": True,
        }

    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        action, source, parsed_segment = _parse_segment_action(
            line,
            allow_loose_json=True,
            ignore_non_action_json=True,
        )
        if action is None:
            continue
        actions.append(action)
        action_segments.append(parsed_segment or line)
        action_sources.append(source)
        if len(actions) >= max_actions:
            break

    if actions:
        return {
            "actions": actions,
            "action_segments": action_segments,
            "action_sources": action_sources,
            "used_role_markers": False,
        }

    fallback_actions = BASE.parse_action_sequence(text, max_actions=max_actions)
    return {
        "actions": fallback_actions,
        "action_segments": [_safe_text(action) for action in fallback_actions],
        "action_sources": ["token_fallback" for _ in fallback_actions],
        "used_role_markers": False,
    }


def normalize_ground_truth(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    candidate_triples = BASE.normalize_reward_candidate_triples(ground_truth.get("candidate_triples") or [])
    return {
        **ground_truth,
        "candidate_triples": candidate_triples,
        "gold_path_indices": [int(x) for x in (ground_truth.get("gold_path_indices") or []) if str(x).isdigit()],
        "support_union_indices": [
            int(x)
            for x in (ground_truth.get("support_union_indices") or ground_truth.get("gold_path_indices") or [])
            if str(x).isdigit()
        ],
        "answer_entities": _normalize_entity_list(ground_truth.get("answer_entities") or ground_truth.get("gold_answers") or []),
        "gold_answers": _normalize_entity_list(ground_truth.get("gold_answers") or ground_truth.get("answer_entities") or []),
        "topic_entity": _safe_text(ground_truth.get("topic_entity")),
        "question": _safe_text(ground_truth.get("question")),
    }


def extract_accepted_action_history(solution_str: str) -> List[Any] | None:
    """Read interaction user feedback embedded in a multiturn rollout decode."""
    text = solution_str or ""
    histories: List[List[Any]] = []
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        history = obj.get("accepted_action_history")
        if isinstance(history, list):
            histories.append(history)
    if not histories:
        return None
    return histories[-1]


def _count_triple_actions(actions: Sequence[Any] | None) -> int:
    return sum(1 for action in (actions or []) if isinstance(action, int))


def _resolve_retriever_solution_source(
    solution_str: str,
    extra_info: Dict[str, Any] | None,
) -> str:
    extra = extra_info if isinstance(extra_info, dict) else {}
    full_rollout = extra.get("rollout_response_str_full")
    if isinstance(full_rollout, str) and full_rollout.strip():
        _, saw_markers = _extract_assistant_segments(full_rollout)
        if saw_markers:
            return "full_rollout_decode"
    _, saw_markers = _extract_assistant_segments(solution_str)
    if saw_markers:
        return "solution_str_decode"
    assistant_actions = extract_action_trace(solution_str).get("actions") or []
    if _count_triple_actions(assistant_actions) > 0 or assistant_actions:
        return "assistant_decode"
    return "none"


def resolve_retriever_solution_str(
    solution_str: str,
    extra_info: Dict[str, Any] | None = None,
) -> str:
    """Replay from full multiturn rollout decode (grpov2-compatible), not rebuilt action indices."""
    extra = extra_info if isinstance(extra_info, dict) else {}
    candidates: List[str] = []
    full_rollout = extra.get("rollout_response_str_full")
    if isinstance(full_rollout, str) and full_rollout.strip():
        candidates.append(full_rollout)
    if isinstance(solution_str, str) and solution_str.strip():
        candidates.append(solution_str)
    for text in candidates:
        _, saw_role_markers = _extract_assistant_segments(text)
        if saw_role_markers:
            return text
    return candidates[0] if candidates else (solution_str or "")


def diagnose_triple_transmission(
    *,
    solution_str: str,
    extra_info: Dict[str, Any] | None,
    replay: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify empty remote triples: no accepted selection vs accepted but lost in reward replay."""
    extra = extra_info if isinstance(extra_info, dict) else {}

    rollout_history = extra.get("accepted_action_history")
    if not isinstance(rollout_history, list):
        rollout_history = None

    full_rollout = extra.get("rollout_response_str_full")
    full_history = None
    if isinstance(full_rollout, str) and full_rollout.strip():
        full_history = extract_accepted_action_history(full_rollout)

    assistant_actions = list(extract_action_trace(solution_str).get("actions") or [])
    n_rollout_accepted = _count_triple_actions(rollout_history)
    n_full_history_accepted = _count_triple_actions(full_history)
    n_assistant_parsed = _count_triple_actions(assistant_actions)
    n_replay_selected = len(replay.get("selected_indices") or [])
    n_remote_triples = len(replay.get("selected_triples") or [])
    resolve_source = _resolve_retriever_solution_source(solution_str, extra)

    if n_rollout_accepted > 0 and n_replay_selected == 0:
        diagnosis = "accepted_but_replay_empty"
    elif n_rollout_accepted > 0 and n_remote_triples == 0:
        diagnosis = "accepted_but_remote_empty"
    elif n_rollout_accepted == 0 and n_replay_selected == 0:
        if n_assistant_parsed > 0 or n_full_history_accepted > 0:
            diagnosis = "parsed_not_accepted"
        else:
            diagnosis = "no_selection"
    elif n_replay_selected > 0:
        diagnosis = "ok"
    else:
        diagnosis = "unknown"

    return {
        "triple_transmission_diagnosis": diagnosis,
        "triple_resolve_source": resolve_source,
        "n_rollout_accepted_triples": n_rollout_accepted,
        "n_full_decode_history_triples": n_full_history_accepted,
        "n_assistant_parsed_triples": n_assistant_parsed,
        "n_replay_selected_triples": n_replay_selected,
        "n_remote_triples": n_remote_triples,
        "has_rollout_history_field": 1.0 if isinstance(rollout_history, list) else 0.0,
        "diag_no_selection": 1.0 if diagnosis == "no_selection" else 0.0,
        "diag_parsed_not_accepted": 1.0 if diagnosis == "parsed_not_accepted" else 0.0,
        "diag_accepted_lost_in_reward": 1.0
        if diagnosis in {"accepted_but_replay_empty", "accepted_but_remote_empty"}
        else 0.0,
        "diag_ok": 1.0 if diagnosis == "ok" else 0.0,
    }


def _candidate_maps(ground_truth: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], set[int]]:
    candidates = BASE.normalize_reward_candidate_triples(ground_truth.get("candidate_triples") or [])
    by_idx = {int(candidate["idx"]): candidate for candidate in candidates}
    return by_idx, set(by_idx.keys())


def _is_first_step_connected(candidate: Dict[str, Any], topic_entity: str) -> bool:
    topic = _safe_text(topic_entity)
    if not topic:
        return True
    return topic in BASE.triple_entities(candidate)


def _is_connected_action(candidate: Dict[str, Any], selected_candidates: Sequence[Dict[str, Any]], topic_entity: str) -> bool:
    if not selected_candidates:
        return _is_first_step_connected(candidate, topic_entity)
    return BASE.is_connected_action(candidate, selected_candidates)


def _selected_entity_set(
    selected_indices: Sequence[int],
    candidates_by_idx: Dict[int, Dict[str, Any]],
) -> set[str]:
    entities: set[str] = set()
    for idx in selected_indices:
        candidate = candidates_by_idx.get(idx)
        if candidate is not None:
            entities |= BASE.triple_entities(candidate)
    return entities


def _coverage_tuple(
    selected_indices: Sequence[int],
    candidates_by_idx: Dict[int, Dict[str, Any]],
    support_union_indices: Sequence[int],
    answer_entities: Sequence[str],
) -> Tuple[float, float, float]:
    support = float(BASE.support_recall(selected_indices, support_union_indices))
    answer = float(BASE.answer_hit_stats(selected_indices, candidates_by_idx, answer_entities)[1])
    connectivity = float(BASE.connectivity_ratio(selected_indices, candidates_by_idx))
    return support, answer, connectivity


def _position_discount(rank: int) -> float:
    rank = max(1, int(rank))
    return 1.0 / math.log2(rank + 1.0)


def _has_same_head_relation(
    candidate: Dict[str, Any],
    selected_candidates: Sequence[Dict[str, Any]],
) -> bool:
    head = _safe_text(candidate.get("head"))
    relation = _safe_text(candidate.get("relation"))
    for prev in selected_candidates:
        if _safe_text(prev.get("head")) == head and _safe_text(prev.get("relation")) == relation:
            return True
    return False


def _selected_triples(selected_indices: Sequence[int], candidates_by_idx: Dict[int, Dict[str, Any]]) -> List[List[str]]:
    triples: List[List[str]] = []
    for idx in selected_indices:
        candidate = candidates_by_idx.get(int(idx))
        if candidate is None:
            continue
        triples.append([candidate["head"], candidate["relation"], candidate["tail"]])
    return triples


def _stop_metrics(
    selected_indices: Sequence[int],
    candidates_by_idx: Dict[int, Dict[str, Any]],
    support_union_indices: Sequence[int],
    answer_entities: Sequence[str],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    answer_entities_hit, answer_recall = BASE.answer_hit_stats(selected_indices, candidates_by_idx, answer_entities)
    connectivity = BASE.connectivity_ratio(selected_indices, candidates_by_idx)
    support = BASE.support_recall(selected_indices, support_union_indices)
    min_answer_hits = int(ground_truth.get("min_answer_hits_to_stop", BASE.DEFAULT_MIN_ANSWER_HITS_TO_STOP))
    min_answer_recall = float(
        ground_truth.get("min_answer_recall_to_stop", BASE.DEFAULT_MIN_ANSWER_RECALL_TO_STOP)
    )
    min_connectivity = float(
        ground_truth.get("min_connectivity_ratio_to_stop", BASE.DEFAULT_MIN_CONNECTIVITY_RATIO_TO_STOP)
    )
    min_selected = int(ground_truth.get("min_selected_triples_to_stop", BASE.DEFAULT_MIN_SELECTED_TRIPLES_TO_STOP))
    coverage_ready = len(answer_entities_hit) >= min_answer_hits or answer_recall >= min_answer_recall
    graph_ready = len(selected_indices) >= min_selected and connectivity >= min_connectivity
    stop_allowed = coverage_ready and graph_ready
    base_stop_reward = -0.3 + 0.8 * answer_recall + 0.5 * support + 0.1 * connectivity
    illegal_stop_penalty = 3.0 if not stop_allowed else 0.0
    incomplete_answer_penalty = 2.5 * (1.0 - answer_recall) if answer_entities and answer_recall < 1.0 else 0.0
    stop_reward = base_stop_reward - illegal_stop_penalty - incomplete_answer_penalty
    return {
        "answer_entities_hit": answer_entities_hit,
        "answer_recall": answer_recall,
        "connectivity_ratio": connectivity,
        "support_recall": support,
        "stop_allowed": stop_allowed,
        "stop_reward": stop_reward,
    }


def replay_episode(ground_truth: Dict[str, Any], solution_str: str) -> Dict[str, Any]:
    gt = normalize_ground_truth(ground_truth)
    candidates_by_idx, allowed_indices = _candidate_maps(gt)
    gold_path_indices = list(gt.get("gold_path_indices") or [])
    support_union_indices = list(gt.get("support_union_indices") or gold_path_indices)
    topic_entity = _safe_text(gt.get("topic_entity"))
    answer_entities = _normalize_entity_list(gt.get("answer_entities") or gt.get("gold_answers") or [])
    max_soft, _ = BASE.resolve_max_selected_triple_limits(gt)

    action_trace = extract_action_trace(solution_str)
    actions = list(action_trace.get("actions") or [])
    selected_indices: List[int] = []
    selected_set: set[int] = set()
    selected_candidates: List[Dict[str, Any]] = []
    step_rewards: List[float] = []
    step_details: List[Dict[str, Any]] = []
    stop_seen = False
    stop_reward = 0.0
    stop_metrics: Dict[str, Any] = {
        "answer_entities_hit": [],
        "answer_recall": 0.0,
        "connectivity_ratio": 0.0,
        "support_recall": 0.0,
        "stop_allowed": False,
        "stop_reward": 0.0,
    }

    for step_index, action in enumerate(actions):
        if action == DEFAULT_STOP_ACTION:
            stop_seen = True
            stop_metrics = _stop_metrics(selected_indices, candidates_by_idx, support_union_indices, answer_entities, gt)
            stop_reward = float(stop_metrics["stop_reward"])
            step_details.append(
                {
                    "step_index": step_index,
                    "action": DEFAULT_STOP_ACTION,
                    "reward": stop_reward,
                    "reason": "stop_action",
                    **stop_metrics,
                }
            )
            break

        if not isinstance(action, int) or action not in allowed_indices:
            step_rewards.append(INVALID_ACTION_REWARD)
            step_details.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "reward": INVALID_ACTION_REWARD,
                    "reason": "invalid_action",
                }
            )
            continue

        if action in selected_set:
            step_rewards.append(INVALID_ACTION_REWARD)
            step_details.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "reward": INVALID_ACTION_REWARD,
                    "reason": "duplicate_action",
                }
            )
            continue

        candidate = candidates_by_idx[action]
        is_connected = _is_connected_action(candidate, selected_candidates, topic_entity)

        prev_support, prev_answer, prev_connectivity = _coverage_tuple(
            selected_indices, candidates_by_idx, support_union_indices, answer_entities
        )
        prev_entities = _selected_entity_set(selected_indices, candidates_by_idx)
        same_head_relation = _has_same_head_relation(candidate, selected_candidates)

        selected_indices.append(action)
        selected_set.add(action)
        selected_candidates.append(candidate)

        if not is_connected:
            step_rewards.append(DISCONNECTED_ACTION_REWARD)
            step_details.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "reward": DISCONNECTED_ACTION_REWARD,
                    "reason": "disconnected_action",
                    "candidate": candidate,
                    "selected_count": len(selected_indices),
                }
            )
            continue

        curr_support, curr_answer, curr_connectivity = _coverage_tuple(
            selected_indices, candidates_by_idx, support_union_indices, answer_entities
        )
        curr_entities = _selected_entity_set(selected_indices, candidates_by_idx)

        delta_support = max(0.0, curr_support - prev_support)
        delta_answer = max(0.0, curr_answer - prev_answer)
        delta_connectivity = max(0.0, curr_connectivity - prev_connectivity)
        new_entity_gain = float(len(curr_entities - prev_entities)) / 2.0

        marginal_gain = (
            DELTA_SUPPORT_WEIGHT * delta_support
            + DELTA_ANSWER_WEIGHT * delta_answer
            + DELTA_CONNECTIVITY_WEIGHT * delta_connectivity
            + NEW_ENTITY_GAIN_WEIGHT * new_entity_gain
        )

        redundancy_penalty = 0.0
        if (
            delta_support <= 1e-12
            and delta_answer <= 1e-12
            and delta_connectivity <= 1e-12
            and new_entity_gain <= 1e-12
        ):
            redundancy_penalty += NO_GAIN_CONNECTED_PENALTY

        if same_head_relation and delta_support <= 1e-12 and delta_answer <= 1e-12:
            redundancy_penalty += SAME_HEAD_REL_PENALTY

        extra_over_soft = max(0, len(selected_indices) - max_soft)
        soft_overflow_penalty = -0.05 * float(extra_over_soft)
        position_discount = _position_discount(len(selected_indices))
        final_step_reward = position_discount * marginal_gain - redundancy_penalty + soft_overflow_penalty

        if delta_support > 1e-12:
            reason = "support_gain"
        elif delta_answer > 1e-12:
            reason = "answer_gain"
        elif new_entity_gain > 1e-12:
            reason = "frontier_expand"
        else:
            reason = "redundant_connected"

        step_rewards.append(final_step_reward)
        step_details.append(
            {
                "step_index": step_index,
                "action": action,
                "reward": final_step_reward,
                "reason": reason,
                "candidate": candidate,
                "selected_count": len(selected_indices),
                "delta_support_recall": delta_support,
                "delta_answer_recall": delta_answer,
                "delta_connectivity": delta_connectivity,
                "new_entity_gain": new_entity_gain,
                "position_discount": position_discount,
                "redundancy_penalty": redundancy_penalty,
                "same_head_relation": same_head_relation,
                "soft_overflow_penalty": soft_overflow_penalty,
            }
        )

    if not stop_seen:
        stop_metrics = _stop_metrics(selected_indices, candidates_by_idx, support_union_indices, answer_entities, gt)

    return {
        "trajectory_id": gt.get("trajectory_id"),
        "sample_id": gt.get("sample_id"),
        "dataset": gt.get("dataset"),
        "split": gt.get("split"),
        "question": gt.get("question"),
        "topic_entity": topic_entity,
        "gold_answers": list(gt.get("gold_answers") or answer_entities),
        "actions": actions,
        "selected_indices": selected_indices,
        "selected_triples": _selected_triples(selected_indices, candidates_by_idx),
        "step_rewards": step_rewards,
        "step_reward_sum": float(sum(step_rewards)),
        "step_details": step_details,
        **stop_metrics,
        # Keep after **stop_metrics: only credit stop_reward when STOP was actually taken.
        "stop_reward": stop_reward,
        "stop_seen": stop_seen,
    }


def build_final_reward(
    replay: Dict[str, Any],
    answer_f1: float,
    answer_metrics: Dict[str, Any],
    remote_answer_ok: bool = True,
    format_reward: float = 0.0,
) -> Dict[str, Any]:
    # score = 0.8*f1 + 0.05*connect + 0.05*stop + 0.05*step + 0.05*format
    step_reward_sum = float(replay.get("step_reward_sum", 0.0))
    stop_reward = float(replay.get("stop_reward", 0.0))
    connectivity_ratio = float(replay.get("connectivity_ratio", 0.0))
    format_reward = float(format_reward)

    f1_weight_applied = F1_WEIGHT if remote_answer_ok else 0.0
    format_weight_applied = FORMAT_WEIGHT if remote_answer_ok else 0.0

    weighted_f1_reward = f1_weight_applied * float(answer_f1)
    weighted_connect_reward = CONNECT_WEIGHT * connectivity_ratio
    weighted_stop_reward = STOP_WEIGHT * stop_reward
    weighted_step_reward = STEP_WEIGHT * step_reward_sum
    weighted_format_reward = format_weight_applied * format_reward

    score = (
        weighted_f1_reward
        + weighted_connect_reward
        + weighted_stop_reward
        + weighted_step_reward
        + weighted_format_reward
    )
    return {
        **replay,
        **answer_metrics,
        "remote_answer_ok": bool(remote_answer_ok),
        "f1_reward": float(answer_f1) if remote_answer_ok else 0.0,
        "connect_reward": connectivity_ratio,
        "score": score,
        "format_reward": format_reward,
        "reward_weights": {
            "f1": f1_weight_applied,
            "connect": CONNECT_WEIGHT,
            "stop": STOP_WEIGHT,
            "step": STEP_WEIGHT,
            "format": format_weight_applied,
        },
        "weighted_f1_reward": weighted_f1_reward,
        "weighted_connect_reward": weighted_connect_reward,
        "weighted_stop_reward": weighted_stop_reward,
        "weighted_step_reward": weighted_step_reward,
        "weighted_format_reward": weighted_format_reward,
    }
