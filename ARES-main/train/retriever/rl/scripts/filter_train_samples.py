#!/usr/bin/env python3
"""Identify low-quality GRPO training samples from remote_answer_client.jsonl and filter train.jsonl.

Typical workflow:
  1. Run a short GRPO probe (or use an existing run log).
  2. Analyze jsonl to collect sample_id blacklist.
  3. Write a filtered train.jsonl and point cwq_retriever_grpo.yaml at it.

Example:
  python filter_train_samples.py analyze \\
    --jsonl ../logs/remote_answer_client.jsonl \\
    --step-min 53 --step-max 57 \\
    --rule all_rollouts_zero \\
    --output ../data/cwq/exclude_step53_57.txt

  python filter_train_samples.py filter \\
    --input ../data/cwq/train.jsonl \\
    --exclude ../data/cwq/exclude_step53_57.txt \\
    --output ../data/cwq/train_filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


def _sample_id_from_record(obj: Dict[str, Any]) -> str:
    req = obj.get("request") or {}
    sid = req.get("sample_id") or req.get("trajectory_id")
    if sid:
        return str(sid)
    resp = obj.get("response") or {}
    sid = resp.get("sample_id") or resp.get("trajectory_id")
    return str(sid or "")


def _f1_from_record(obj: Dict[str, Any]) -> float:
    resp = obj.get("response") or {}
    cm = resp.get("corrected_metrics") or {}
    if isinstance(cm, dict) and cm.get("f1") is not None:
        try:
            return float(cm["f1"])
        except (TypeError, ValueError):
            pass
    try:
        return float(resp.get("reward_metric_value") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _diag_from_record(obj: Dict[str, Any]) -> str:
    diag = obj.get("transmission_diagnostic") or {}
    return str(diag.get("triple_transmission_diagnosis") or "")


def _gt_sample_id_from_train_row(row: Dict[str, Any]) -> str:
    ex = row.get("extra_info") or {}
    ik = ex.get("interaction_kwargs") or {}
    gt = ik.get("ground_truth") or {}
    return str(gt.get("sample_id") or gt.get("inference_sample_id") or "")


def iter_jsonl_records(
    path: Path,
    *,
    chunk_size: int,
    step_min: Optional[int],
    step_max: Optional[int],
) -> Iterable[Tuple[int, int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            step = (line_no - 1) // chunk_size
            if step_min is not None and step < step_min:
                continue
            if step_max is not None and step > step_max:
                break
            yield line_no, step, json.loads(line)


def aggregate_by_sample(
    path: Path,
    *,
    chunk_size: int,
    step_min: Optional[int],
    step_max: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    by_sample: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "f1_values": [],
            "diags": [],
            "questions": [],
            "steps": set(),
            "line_nos": [],
        }
    )
    for line_no, step, obj in iter_jsonl_records(
        path, chunk_size=chunk_size, step_min=step_min, step_max=step_max
    ):
        sid = _sample_id_from_record(obj)
        if not sid:
            continue
        rec = by_sample[sid]
        rec["f1_values"].append(_f1_from_record(obj))
        rec["diags"].append(_diag_from_record(obj))
        req = obj.get("request") or {}
        q = str(req.get("question") or "")
        if q and q not in rec["questions"]:
            rec["questions"].append(q)
        rec["steps"].add(step)
        rec["line_nos"].append(line_no)
    return dict(by_sample)


def select_bad_sample_ids(
    by_sample: Dict[str, Dict[str, Any]],
    *,
    rule: str,
    min_rollouts: int,
    max_mean_f1: float,
    min_zero_rollout_frac: float,
    diag_any: Optional[Set[str]],
) -> List[Tuple[str, Dict[str, Any]]]:
    selected: List[Tuple[str, Dict[str, Any]]] = []
    for sid, rec in by_sample.items():
        f1s = rec["f1_values"]
        n = len(f1s)
        if n < min_rollouts:
            continue
        mean_f1 = sum(f1s) / n if n else 0.0
        zero_frac = sum(1 for x in f1s if x <= 0.0) / n if n else 1.0
        all_zero = n > 0 and all(x <= 0.0 for x in f1s)
        diags = set(rec["diags"])
        if diag_any and not (diags & diag_any):
            continue

        hit = False
        if rule == "all_rollouts_zero":
            hit = all_zero
        elif rule == "mean_f1_below":
            hit = mean_f1 <= max_mean_f1
        elif rule == "high_zero_frac":
            hit = zero_frac >= min_zero_rollout_frac
        else:
            raise ValueError(f"unknown rule: {rule}")

        if hit:
            rec = dict(rec)
            rec["mean_f1"] = mean_f1
            rec["zero_frac"] = zero_frac
            rec["n_rollouts"] = n
            rec["steps"] = sorted(rec["steps"])
            selected.append((sid, rec))
    selected.sort(key=lambda x: (x[1]["mean_f1"], -x[1]["n_rollouts"], x[0]))
    return selected


def cmd_analyze(args: argparse.Namespace) -> int:
    path = Path(args.jsonl).resolve()
    if not path.is_file():
        print(f"jsonl not found: {path}", file=sys.stderr)
        return 1

    diag_any = None
    if args.diag_any:
        diag_any = {x.strip() for x in args.diag_any.split(",") if x.strip()}

    by_sample = aggregate_by_sample(
        path,
        chunk_size=args.chunk_size,
        step_min=args.step_min,
        step_max=args.step_max,
    )
    bad = select_bad_sample_ids(
        by_sample,
        rule=args.rule,
        min_rollouts=args.min_rollouts,
        max_mean_f1=args.max_mean_f1,
        min_zero_rollout_frac=args.min_zero_frac,
        diag_any=diag_any,
    )

    out_path = Path(args.output).resolve() if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out_f:
            for sid, _ in bad:
                out_f.write(sid + "\n")

    report = {
        "jsonl": str(path),
        "chunk_size": args.chunk_size,
        "step_min": args.step_min,
        "step_max": args.step_max,
        "rule": args.rule,
        "min_rollouts": args.min_rollouts,
        "unique_samples_in_window": len(by_sample),
        "bad_samples": len(bad),
        "output": str(out_path) if out_path else None,
        "examples": [
            {
                "sample_id": sid,
                "mean_f1": rec["mean_f1"],
                "zero_frac": rec["zero_frac"],
                "n_rollouts": rec["n_rollouts"],
                "steps": rec["steps"],
                "diags": sorted(set(rec["diags"])),
                "question": (rec["questions"][0][:120] if rec["questions"] else ""),
            }
            for sid, rec in bad[: min(20, len(bad))]
        ],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_exclude_ids(paths: List[str]) -> Set[str]:
    ids: Set[str] = set()
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid and not sid.startswith("#"):
                    ids.add(sid)
    return ids


def cmd_filter(args: argparse.Namespace) -> int:
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    exclude_ids = _load_exclude_ids(args.exclude)

    kept = 0
    dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            row = json.loads(line)
            sid = _gt_sample_id_from_train_row(row)
            if sid in exclude_ids:
                dropped += 1
                continue
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "exclude_count": len(exclude_ids),
        "kept": kept,
        "dropped": dropped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="scan remote_answer_client.jsonl and write exclude list")
    p_analyze.add_argument("--jsonl", required=True)
    p_analyze.add_argument("--output", required=True, help="one sample_id per line")
    p_analyze.add_argument("--report", default=None, help="optional JSON report path")
    p_analyze.add_argument("--chunk-size", type=int, default=256, help="lines per training step (batch*rollout.n)")
    p_analyze.add_argument("--step-min", type=int, default=None)
    p_analyze.add_argument("--step-max", type=int, default=None)
    p_analyze.add_argument(
        "--rule",
        choices=("all_rollouts_zero", "mean_f1_below", "high_zero_frac"),
        default="all_rollouts_zero",
    )
    p_analyze.add_argument("--min-rollouts", type=int, default=4)
    p_analyze.add_argument("--max-mean-f1", type=float, default=0.0)
    p_analyze.add_argument("--min-zero-frac", type=float, default=1.0, help="for rule=high_zero_frac")
    p_analyze.add_argument(
        "--diag-any",
        default=None,
        help="comma-separated transmission diagnoses; sample must hit at least one (e.g. no_selection,ok)",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_filter = sub.add_parser("filter", help="drop excluded sample_ids from train.jsonl")
    p_filter.add_argument("--input", required=True)
    p_filter.add_argument("--output", required=True)
    p_filter.add_argument("--exclude", nargs="+", required=True, help="text file(s), one sample_id per line")
    p_filter.set_defaults(func=cmd_filter)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
