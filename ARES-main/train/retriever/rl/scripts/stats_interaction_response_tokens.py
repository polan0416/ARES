#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from transformers import AutoTokenizer


def _gt_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    ex = row.get("extra_info") or {}
    ik = ex.get("interaction_kwargs") or {}
    gt = ik.get("ground_truth")
    if not isinstance(gt, dict):
        raise ValueError("missing ground_truth")
    return gt


def _candidates_by_idx(gt: Dict[str, Any]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for c in gt.get("candidate_triples") or []:
        i = int(c["idx"])
        t = c.get("triple") or []
        if len(t) >= 3:
            out[i] = {"head": str(t[0]), "relation": str(t[1]), "tail": str(t[2])}
    return out


def _next_observation(
    selected_indices: Sequence[int], candidates_by_idx: Dict[int, Dict[str, str]]
) -> Dict[str, Any]:
    current_nodes: set[str] = set()
    selected_candidates = [candidates_by_idx[i] for i in selected_indices if i in candidates_by_idx]
    for cand in selected_candidates:
        current_nodes.add(cand["head"])
        current_nodes.add(cand["tail"])
    return {
        "action": selected_indices[-1] if selected_indices else None,
        "selected_indices": list(selected_indices),
        "current_nodes": sorted(current_nodes),
    }


def _accepted_payload(actions_hist: List[Any], obs: Dict[str, Any]) -> str:
    payload = {"event": "accepted", "accepted_action_history": list(actions_hist), "observation": obs}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _stop_payload(actions_hist: List[Any], stop_reward: float) -> str:
    payload = {
        "event": "stop",
        "accepted_action_history": list(actions_hist),
        "done": True,
        "stop_reward": float(stop_reward),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _simulate_suffix_total_tokens(
    tok: Any,
    initial_msgs: List[Dict[str, Any]],
    path_indices: List[int],
    candidates_by_idx: Dict[int, Dict[str, str]],
) -> Tuple[int, int, List[int]]:
    msgs: List[Dict[str, Any]] = [dict(x) for x in initial_msgs]
    base = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    n0 = len(base)

    def full_len() -> int:
        return len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False))

    cumulative: List[int] = []
    actions_hist: List[Any] = []
    selected: List[int] = []

    for idx in path_indices:
        assistant = json.dumps({"action": idx}, separators=(",", ":"))
        msgs.append({"role": "assistant", "content": assistant})
        actions_hist.append(idx)
        selected.append(idx)
        user = _accepted_payload(actions_hist, _next_observation(selected, candidates_by_idx))
        msgs.append({"role": "user", "content": user})
        cumulative.append(full_len() - n0)

    path_only_suffix = full_len() - n0
    msgs.append({"role": "assistant", "content": json.dumps({"action": "STOP"}, separators=(",", ":"))})
    actions_hist.append("STOP")
    user_stop = _stop_payload(actions_hist, stop_reward=0.0)
    msgs.append({"role": "user", "content": user_stop})
    total_with_stop = full_len() - n0
    cumulative.append(total_with_stop)

    return path_only_suffix, total_with_stop, cumulative


def _first_k_path(gt: Dict[str, Any], k: int | None) -> List[int]:
    gold = list(gt.get("gold_path_indices") or [])
    if k is not None:
        gold = gold[:k]
    return [int(x) for x in gold]


def main() -> None:
    ap = argparse.ArgumentParser()
    _rl_dir = Path(__file__).resolve().parent.parent
    ap.add_argument(
        "--model",
        default=str(Path("~/.cache/huggingface/model/Qwen3-4B").expanduser()),
    )
    ap.add_argument("--val", default=str(_rl_dir / "data/webqsp/val.jsonl"))
    ap.add_argument("--train", default=str(_rl_dir / "data/webqsp/train.jsonl"))
    ap.add_argument("--n-val", type=int, default=221, help="val size or sample count")
    ap.add_argument("--n-train-sample", type=int, default=300, help="random sample size from train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="2048,4096", help="comma-separated budgets; report first exceed step")
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def run_rows(rows: List[Dict[str, Any]], label: str) -> None:
        first_at: Dict[int, List[int]] = {b: [] for b in budgets}
        never_exceed: Dict[int, int] = {b: 0 for b in budgets}
        steps_full: List[int] = []
        suffix_at_stop: List[int] = []

        for row in rows:
            try:
                gt = _gt_from_row(row)
                cand = _candidates_by_idx(gt)
                path = _first_k_path(gt, k=None)
                if not path:
                    continue
                prompt = row.get("prompt")
                if not isinstance(prompt, list):
                    continue
                _, total_stop, cum = _simulate_suffix_total_tokens(tok, prompt, path, cand)
                steps_full.append(len(path))
                suffix_at_stop.append(total_stop)
                for b in budgets:
                    exceeded = False
                    for i, c in enumerate(cum, start=1):
                        if c > b:
                            first_at[b].append(i)
                            exceeded = True
                            break
                    if not exceeded:
                        never_exceed[b] += 1
            except Exception:
                continue

        print(f"\n=== {label} (valid samples {len(suffix_at_stop)}) ===")
        if suffix_at_stop:
            print(
                f"  gold_path steps: min/mean/max = {min(steps_full)} / "
                f"{statistics.mean(steps_full):.2f} / {max(steps_full)}"
            )
            print(
                f"  suffix length (incl. STOP turn, Qwen3 chat re-encode delta): "
                f"min/mean/max = {min(suffix_at_stop)} / "
                f"{statistics.mean(suffix_at_stop):.1f} / {max(suffix_at_stop)}"
            )
        n = len(suffix_at_stop) or 1
        for b in budgets:
            xs = first_at[b]
            ne = never_exceed[b]
            print(f"  budget {b}: never-exceed ratio = {ne}/{n} ({100.0 * ne / n:.1f}%)")
            if xs:
                print(
                    f"    among exceeded samples, first-exceed turn min/mean/p90 = "
                    f"{min(xs)} / {statistics.mean(xs):.2f} / {_pctl(xs, 90):.1f} "
                    f"(n={len(xs)})"
                )

    val_path = Path(args.val)
    val_lines = val_path.read_text().splitlines()
    if args.n_val < len(val_lines):
        rng = random.Random(args.seed)
        pick = rng.sample(range(len(val_lines)), args.n_val)
        val_rows = [json.loads(val_lines[i]) for i in pick]
    else:
        val_rows = [json.loads(x) for x in val_lines]

    run_rows(val_rows, f"val (n={len(val_rows)})")

    train_path = Path(args.train)
    train_lines = train_path.read_text().splitlines()
    pick = random.Random(args.seed).sample(range(len(train_lines)), min(args.n_train_sample, len(train_lines)))
    train_rows = [json.loads(train_lines[i]) for i in pick]
    run_rows(train_rows, f"train random sample (n={len(train_rows)})")


def _pctl(xs: List[float], p: float) -> float:
    ys = sorted(xs)
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    return ys[f] + (k - f) * (ys[c] - ys[f])


if __name__ == "__main__":
    main()
