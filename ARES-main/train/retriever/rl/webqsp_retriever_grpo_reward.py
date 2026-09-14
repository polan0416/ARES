from __future__ import annotations

import json
from typing import Any, Dict

from rl.format_reward import compute_answer_format_reward
from rl.remote_answer_client import RemoteAnswerClient
from rl.reward_logic import (
    build_final_reward,
    diagnose_triple_transmission,
    replay_episode,
    resolve_retriever_solution_str,
)

DATA_SOURCE = "webqsp_retriever_grpo"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _reward_extra_value_for_batch(value: Any) -> Any:
    """verl agent_loop stacks reward_extra_info with np.array(...); nested dicts / ragged lists must be JSON strings."""
    if value is None:
        # verl process_validation_metrics calls np.mean on scalar extras; None crashes.
        return 0.0
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist(), ensure_ascii=False, default=str)
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return value



def _sanitize_reward_result_for_verl_batching(result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in result.items():
        if key == "score":
            out[key] = float(value) if value is not None else 0.0
        else:
            out[key] = _reward_extra_value_for_batch(value)
    return out



def _resolve_ground_truth(
    ground_truth: Dict[str, Any] | None,
    extra_info: Dict[str, Any] | None,
) -> Dict[str, Any]:
    gt = ground_truth if isinstance(ground_truth, dict) and ground_truth else None
    if gt is None and isinstance(extra_info, dict):
        interaction_kwargs = extra_info.get("interaction_kwargs") or {}
        if isinstance(interaction_kwargs, dict) and isinstance(interaction_kwargs.get("ground_truth"), dict):
            gt = interaction_kwargs["ground_truth"]
    if not isinstance(gt, dict) or not gt:
        raise ValueError("Missing ground_truth for webqsp retriever GRPO reward.")
    return gt



def _extract_remote_answer_metrics(answer_result: Dict[str, Any]) -> Dict[str, Any]:
    remote_ok = bool(answer_result.get("ok"))
    if not remote_ok:
        return {
            "prediction_lines": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_predictions": 0.0,
            "num_predictions": 0,
            "matched_gold": 0.0,
            "num_gold": 0,
            "hit": 0.0,
            "hit_any": 0.0,
            "answer_format_valid": 0.0,
            "no_answer": True,
            "hal_score": 0.0,
            "remote_answer_ok": 0.0,
            "remote_answer_error": str(answer_result.get("error") or "remote_answer_failed"),
        }

    corrected_metrics = answer_result.get("corrected_metrics") or {}
    try:
        reward_metric_value = float(answer_result.get("reward_metric_value", corrected_metrics.get("f1", 0.0)) or 0.0)
    except (TypeError, ValueError):
        reward_metric_value = 0.0
    return {
        "prediction_lines": corrected_metrics.get("prediction_lines") or [],
        "precision": _coerce_float(corrected_metrics.get("precision")),
        "recall": _coerce_float(corrected_metrics.get("recall")),
        "f1": reward_metric_value,
        "matched_predictions": _coerce_float(corrected_metrics.get("matched_predictions")),
        "num_predictions": _coerce_int(corrected_metrics.get("num_predictions")),
        "matched_gold": _coerce_float(corrected_metrics.get("matched_gold")),
        "num_gold": _coerce_int(corrected_metrics.get("num_gold")),
        "hit": _coerce_float(corrected_metrics.get("hit")),
        "hit_any": _coerce_float(corrected_metrics.get("hit_any")),
        "answer_format_valid": _coerce_float(
            answer_result.get("answer_format_valid", corrected_metrics.get("answer_format_valid"))
        ),
        "no_answer": bool(corrected_metrics.get("no_answer", False)),
        "hal_score": _coerce_float(corrected_metrics.get("hal_score")),
        "remote_answer_ok": 1.0,
        "remote_answer_error": "",
    }



def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any] | None,
    extra_info: Dict[str, Any] | None = None,
    remote_answer_timeout_sec: float | None = None,
    **_: Any,
) -> Dict[str, Any]:
    if data_source != DATA_SOURCE:
        raise ValueError(f"Unsupported data_source for WebQSP retriever GRPO reward: {data_source}")

    gt = _resolve_ground_truth(ground_truth=ground_truth, extra_info=extra_info)
    effective_solution_str = resolve_retriever_solution_str(solution_str, extra_info=extra_info)
    replay = replay_episode(ground_truth=gt, solution_str=effective_solution_str)
    transmission_diag = diagnose_triple_transmission(
        solution_str=solution_str,
        extra_info=extra_info,
        replay=replay,
    )

    client = RemoteAnswerClient.from_ground_truth(
        gt,
        timeout_sec_override=remote_answer_timeout_sec,
    )
    answer_result = client.answer(
        question=str(gt.get("question") or ""),
        question_explain=gt.get("question_explain") or {},
        topic_entity=str(gt.get("topic_entity") or ""),
        triples=replay.get("selected_triples") or [],
        trajectory_id=str(gt.get("trajectory_id") or ""),
        sample_id=str(gt.get("sample_id") or gt.get("trajectory_id") or ""),
        gold_answers=gt.get("gold_answers") or gt.get("answer_entities") or [],
        candidate_triples=gt.get("candidate_triples") or [],
        prompt_mode=str(gt.get("prompt_mode") or "scored_200"),
        llm_mode=str(gt.get("llm_mode") or "sys_icl_dc"),
        eval_mode=str(gt.get("eval_mode") or "kbqa_scored_200_corrected"),
        score_threshold=float(gt.get("score_threshold", 0.0) or 0.0),
        a_entity_in_graph=gt.get("a_entity_in_graph"),
        transmission_diagnostic=transmission_diag,
    )

    remote_ok = bool(answer_result.get("ok"))
    answer_metrics = _extract_remote_answer_metrics(answer_result)
    answer_f1 = answer_metrics["f1"] if remote_ok else 0.0
    if remote_ok:
        prediction_text = str(
            answer_result.get("prediction_text")
            or answer_result.get("raw_output")
            or ""
        )
        prediction_lines = answer_metrics.get("prediction_lines") or []
        if not isinstance(prediction_lines, list):
            prediction_lines = []
        format_info = compute_answer_format_reward(prediction_text, prediction_lines)
        format_reward = float(format_info.get("format_reward", 0.0))
    else:
        format_info = {"format_reward": 0.0, "format_issues": ["remote_answer_failed"]}
        format_reward = 0.0

    result = build_final_reward(
        replay=replay,
        answer_f1=answer_f1,
        answer_metrics=answer_metrics,
        remote_answer_ok=remote_ok,
        format_reward=format_reward,
    )
    result.update(
        {
            "trajectory_id": replay.get("trajectory_id") or gt.get("trajectory_id"),
            "n_selected_triples": len(replay.get("selected_indices") or []),
            "remote_answer": answer_result,
            "remote_answer_ok": 1.0 if remote_ok else 0.0,
            "remote_answer_error": answer_metrics.get("remote_answer_error") or "",
            "solution_str": solution_str,
            "ground_truth_question": gt.get("question"),
            "ground_truth_topic_entity": gt.get("topic_entity"),
            "question_explain": gt.get("question_explain") or {},
            "selected_triples_json": json.dumps(replay.get("selected_triples") or [], ensure_ascii=False),
            "reward_answer_metric_name": "corrected_f1" if remote_ok else "corrected_f1_skipped_remote_failed",
            "reward_answer_metric_value": answer_f1 if remote_ok else 0.0,
            "corrected_metrics_json": answer_result.get("corrected_metrics") or {},
            "kbqa_prompt_mode": gt.get("prompt_mode") or "scored_200",
            "kbqa_eval_semantics": gt.get("eval_mode") or "kbqa_scored_200_corrected",
            "answer_prediction_text": (
                str(answer_result.get("prediction_text") or answer_result.get("raw_output") or "")
                if remote_ok
                else ""
            ),
            "format_issues": format_info.get("format_issues") or [],
            **transmission_diag,
        }
    )
    return _sanitize_reward_result_for_verl_batching(result)
