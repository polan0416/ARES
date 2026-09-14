from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Sequence
from urllib import error, request

DEFAULT_REMOTE_ANSWER_URL = "http://10.87.135.152:8004/answer"
DEFAULT_CWQ_REMOTE_ANSWER_URL = "http://10.87.135.152:8002/answer"
DEFAULT_REMOTE_ANSWER_URL_BY_DATASET = {
    "webqsp": DEFAULT_REMOTE_ANSWER_URL,
    "cwq": DEFAULT_CWQ_REMOTE_ANSWER_URL,
}
DEFAULT_REMOTE_TIMEOUT_SEC = 60.0
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "remote_answer_client.jsonl"


def resolve_remote_answer_service_url(dataset: str, override: str | None = None) -> str:
    """Resolve frozen answerer URL by dataset, with optional explicit override."""
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
    return DEFAULT_REMOTE_ANSWER_URL_BY_DATASET.get(str(dataset or "webqsp"), DEFAULT_REMOTE_ANSWER_URL)


def _coerce_positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_timeout_sec(
    ground_truth: Dict[str, Any] | None = None,
    timeout_sec_override: float | None = None,
) -> float:
    """Priority: explicit override > env WEBQSP_REMOTE_ANSWER_TIMEOUT_SEC > jsonl ground_truth > default."""
    if timeout_sec_override is not None and str(timeout_sec_override).strip() != "":
        return _coerce_positive_float(timeout_sec_override, DEFAULT_REMOTE_TIMEOUT_SEC)
    env_value = os.environ.get("WEBQSP_REMOTE_ANSWER_TIMEOUT_SEC")
    if env_value is not None and str(env_value).strip() != "":
        return _coerce_positive_float(env_value, DEFAULT_REMOTE_TIMEOUT_SEC)
    if isinstance(ground_truth, dict):
        gt_value = ground_truth.get("remote_answer_timeout_sec")
        if gt_value is not None and str(gt_value).strip() != "":
            return _coerce_positive_float(gt_value, DEFAULT_REMOTE_TIMEOUT_SEC)
    return DEFAULT_REMOTE_TIMEOUT_SEC


class RemoteAnswerClient:
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, service_url: str, timeout_sec: float = DEFAULT_REMOTE_TIMEOUT_SEC, log_path: Path = DEFAULT_LOG_PATH):
        self.service_url = (service_url or DEFAULT_REMOTE_ANSWER_URL).strip()
        self.timeout_sec = float(timeout_sec)
        self.log_path = log_path

    @classmethod
    def from_ground_truth(
        cls,
        ground_truth: Dict[str, Any],
        timeout_sec_override: float | None = None,
    ) -> "RemoteAnswerClient":
        service_url = resolve_remote_answer_service_url(
            dataset=str(ground_truth.get("dataset") or "webqsp"),
            override=ground_truth.get("remote_answer_service_url"),
        )
        timeout_sec = _resolve_timeout_sec(ground_truth, timeout_sec_override=timeout_sec_override)
        return cls(service_url=str(service_url), timeout_sec=timeout_sec)

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _log(self, payload: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def answer(
        self,
        question: str,
        question_explain: Dict[str, Any],
        topic_entity: str,
        triples: Sequence[Sequence[str]],
        trajectory_id: str = "",
        sample_id: str = "",
        gold_answers: Sequence[str] | None = None,
        candidate_triples: Sequence[Dict[str, Any]] | None = None,
        prompt_mode: str = "scored_200",
        llm_mode: str = "sys_icl_dc",
        eval_mode: str = "kbqa_scored_200_corrected",
        score_threshold: float = 0.0,
        a_entity_in_graph: bool | None = None,
        transmission_diagnostic: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "sample_id": sample_id or trajectory_id or "",
            "question": question or "",
            "question_explain": question_explain or {},
            "topic_entity": topic_entity or "",
            "triples": list(triples or []),
            "trajectory_id": trajectory_id or "",
            "gold_answers": list(gold_answers or []),
            "candidate_triples": list(candidate_triples or []),
            "prompt_mode": prompt_mode or "scored_200",
            "llm_mode": llm_mode or "sys_icl_dc",
            "eval_mode": eval_mode or "kbqa_scored_200_corrected",
            "score_threshold": float(score_threshold or 0.0),
            "a_entity_in_graph": a_entity_in_graph,
        }
        cache_key = self._cache_key(payload)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        started_at = time.time()
        req = request.Request(
            self.service_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                result = {
                    "ok": True,
                    "cached": False,
                    "service_url": self.service_url,
                    "timeout_sec": self.timeout_sec,
                    "status_code": getattr(response, "status", 200),
                    "latency_ms": round((time.time() - started_at) * 1000.0, 3),
                    **parsed,
                }
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            result = {
                "ok": False,
                "cached": False,
                "service_url": self.service_url,
                "timeout_sec": self.timeout_sec,
                "status_code": None,
                "latency_ms": round((time.time() - started_at) * 1000.0, 3),
                "error": str(exc),
                "raw_output": "",
                "parsed_answer": "",
            }

        if result.get("ok"):
            with self._cache_lock:
                self._cache[cache_key] = dict(result)
        log_entry: Dict[str, Any] = {"request": payload, "response": result}
        if transmission_diagnostic:
            log_entry["transmission_diagnostic"] = transmission_diagnostic
        self._log(log_entry)
        return result
