from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

from tqdm import tqdm
from vllm import LLM, SamplingParams

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data, unique_preserve_order
from prompts import icl_user_prompt, icl_ass_prompt, icl_sys_prompt, icl_cot_prompt
from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original

_VLLM_PATCHED = False


def _patch_vllm() -> None:
    global _VLLM_PATCHED
    if _VLLM_PATCHED:
        return
    try:
        import vllm.config as vllm_config

        _orig = vllm_config.ModelConfig._init_multimodal_config

        def _safe(self, limit_mm_per_prompt):
            arch = getattr(self.hf_config, "architectures", None)
            if arch is None or not isinstance(arch, (list, tuple)):
                fixed = ["LlamaForCausalLM"] if getattr(self.hf_config, "model_type", None) == "llama" else ([] if arch is None else [arch])
                try:
                    self.hf_config.architectures = fixed
                except Exception:
                    object.__setattr__(self.hf_config, "architectures", fixed)
            return _orig(self, limit_mm_per_prompt)

        vllm_config.ModelConfig._init_multimodal_config = _safe
        _VLLM_PATCHED = True
    except Exception as e:
        print(f"[main_vllm] vLLM patch skipped: {e}")


def _filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        from vllm.engine.arg_utils import EngineArgs

        supported = inspect.signature(EngineArgs.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in supported and v is not None}
    except Exception:
        return {k: v for k, v in kwargs.items() if v is not None}


def llm_init(cfg: dict[str, Any]):
    _patch_vllm()
    try:
        logging.getLogger("vllm").setLevel(logging.WARNING)
    except Exception:
        pass
    max_len = int(cfg["max_seq_len_to_capture"])
    num_workers = int(cfg["num_workers"])
    kwargs = _filter_kwargs(
        {
            "model": cfg["model_name"],
            "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
            "trust_remote_code": True,
            "gpu_memory_utilization": float(
                cfg.get("gpu_memory_utilization") or os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.88")
            ),
            "seed": int(cfg["seed"]),
            "max_seq_len_to_capture": max_len,
            "max_model_len": max_len,
            "max_num_seqs": num_workers if num_workers > 1 else None,
            "enforce_eager": True if cfg.get("enforce_eager") else None,
        }
    )
    client = LLM(**kwargs)
    params = SamplingParams(
        temperature=float(cfg["temperature"]),
        max_tokens=int(cfg["max_tokens"]),
        frequency_penalty=float(cfg["frequency_penalty"]),
    )
    return partial(client.chat, sampling_params=params, use_tqdm=False)


def build_conversation(prompts: dict) -> list[dict]:
    return [
        {"role": "system", "content": prompts["sys_query"]},
        {"role": "user", "content": icl_user_prompt},
        {"role": "assistant", "content": icl_ass_prompt},
        {"role": "user", "content": prompts["user_query"]},
    ]


def needs_dc(text: str) -> bool:
    low = text.lower()
    return "ans:" not in low or "ans: not available" in low or "ans: no information available" in low


def infer_batch(llm, batch: list[dict]) -> list[str]:
    conversations = [build_conversation(p) for p in batch]
    outputs = llm(messages=conversations)
    answers = [o.outputs[0].text for o in outputs]

    dc_convs, dc_idx = [], []
    for i, (conv, ans) in enumerate(zip(conversations, answers)):
        if not needs_dc(ans):
            continue
        dc = list(conv)
        dc.append({"role": "assistant", "content": ans})
        dc.append({"role": "user", "content": batch[i]["cot_query"]})
        dc_convs.append(dc)
        dc_idx.append(i)
    if dc_convs:
        for i, out in zip(dc_idx, llm(messages=dc_convs)):
            answers[i] = out.outputs[0].text
    return answers


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def parse_ans_lines(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        low = line.lower()
        if "ans:" in low and "ans: not available" not in low and "ans: no information available" not in low:
            out.append(line.strip())
    return unique_preserve_order(out)


def union_predictions(pred_runs: list[list[dict]]) -> list[dict]:
    by_run = [{x["id"]: x for x in run} for run in pred_runs]
    merged = []
    for row0 in pred_runs[0]:
        ans: list[str] = []
        for by_id in by_run:
            ans.extend(parse_ans_lines(by_id.get(row0["id"], {}).get("prediction", "")))
        merged.append(
            {
                "id": row0["id"],
                "question": row0["question"],
                "ground_truth": row0["ground_truth"],
                "prediction": "\n".join(unique_preserve_order(ans)),
            }
        )
    return merged


def eval_hit_f1(pred_path: Path, topk: int) -> tuple[float, float]:
    eval_path = pred_path.with_name(f"scored_{topk}-{pred_path.stem}-predictions.jsonl")
    if eval_path != pred_path:
        with open(pred_path) as src, open(eval_path, "w") as dst:
            dst.write(src.read())
    corrected = quiet(eval_results_corrected, str(eval_path), cal_f1=True, subset=True)
    f1 = corrected[1] if len(corrected) == 14 else corrected[2]
    hit, *_ = quiet(eval_results_original, str(eval_path), cal_f1=True, subset=True)
    return hit, f1


def pred_resume_path(cfg: dict[str, Any], run_index: int) -> Path:
    n = int(cfg["num_runs"])
    suffix = f"-run{run_index + 1}" if n > 1 else ""
    return Path(cfg["output_dir"]) / (
        f"{cfg['pth_tag']}{suffix}-{cfg['prompt_mode']}-sys_icl_dc-"
        f"{cfg['frequency_penalty']}-thres_{cfg['thres']}-{cfg['split']}-predictions-resume.jsonl"
    )


def run_one(cfg: dict[str, Any], llm, run_index: int) -> Path:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = pred_resume_path(cfg, run_index)

    data = quiet(
        get_data,
        cfg["dataset_name"],
        cfg["base_pred_path"],
        cfg["retrieval_path"],
        cfg["split"],
        cfg["prompt_mode"],
    )
    data = quiet(
        get_prompts_for_data,
        data,
        cfg["prompt_mode"],
        icl_sys_prompt,
        icl_cot_prompt,
        float(cfg["thres"]),
    )

    start = sum(1 for _ in open(resume)) if resume.exists() else 0
    todo = data[start:]
    bs = max(1, int(cfg["num_workers"]))

    with open(resume, "a") as f:
        for i in tqdm(range(0, len(todo), bs), desc="Inference", leave=False):
            chunk = todo[i : i + bs]
            for qa, ans in zip(chunk, infer_batch(llm, chunk)):
                for k in ("graph", "good_paths_rog", "good_triplets_rog", "scored_triplets"):
                    qa.pop(k, None)
                qa["prediction"] = ans
                f.write(json.dumps(qa) + "\n")

    final = resume.with_name(resume.stem.replace("-resume", "") + resume.suffix)
    os.rename(resume, final)
    return final


def run_pipeline(cfg: dict[str, Any], reuse: list[str] | None) -> None:
    n = int(cfg["num_runs"])
    if reuse is not None and len(reuse) != n:
        raise ValueError(f"--reuse_predictions needs {n} files, got {len(reuse)}")

    paths: list[Path] = []
    llm = None
    try:
        if reuse is None:
            llm = llm_init(cfg)
        for i in range(n):
            if reuse is not None:
                p = Path(reuse[i])
                if not p.is_file():
                    raise FileNotFoundError(p)
            else:
                p = run_one(cfg, llm, i)
            paths.append(p)
    finally:
        if llm is not None:
            del llm

    topk = int(cfg["prompt_mode"].split("_")[1]) if "scored" in cfg["prompt_mode"] else 100
    union_path = Path(cfg["output_dir"]) / f"pred_union_{cfg['pth_tag']}_first{n}.jsonl"
    save_jsonl(union_path, union_predictions([load_jsonl(p) for p in paths]))
    hit, f1 = eval_hit_f1(union_path, topk)
    print(f"Hit: {hit}")
    print(f"Macro F1: {f1}")


def main():
    p = argparse.ArgumentParser(description="Hit / Macro F1")
    p.add_argument("-d", "--dataset_name", default="cwq")
    p.add_argument("--prompt_mode", default="scored_200")
    p.add_argument("-p", "--retrieval_path", required=True, help="results.pth")
    p.add_argument("--reuse_predictions", nargs="+", default=None, help="skip inference")
    p.add_argument("--num_runs", type=int, default=2)
    p.add_argument("-m", "--model_name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2)
    p.add_argument("--max_tokens", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--frequency_penalty", type=float, default=0.16)
    p.add_argument("--thres", type=float, default=0.0)
    p.add_argument("--gpu_memory_utilization", type=float, default=None)
    p.add_argument("--enforce_eager", action="store_true")
    args = p.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    if not Path(args.retrieval_path).is_file():
        print(f"[error] missing: {args.retrieval_path}", file=sys.stderr)
        sys.exit(1)

    cfg = {
        "dataset_name": args.dataset_name,
        "prompt_mode": args.prompt_mode,
        "model_name": args.model_name,
        "split": args.split,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_seq_len_to_capture": args.max_seq_len_to_capture,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "frequency_penalty": args.frequency_penalty,
        "thres": args.thres,
        "retrieval_path": args.retrieval_path,
        "pth_tag": Path(args.retrieval_path).stem,
        "base_pred_path": (
            f"./results/KGQA/{args.dataset_name}/RoG/{args.split}/"
            f"results_gen_rule_path_RoG-{args.dataset_name}_RoG_{args.split}_predictions_3_False_jsonl/predictions.jsonl"
        ),
        "output_dir": args.output_dir,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "num_workers": args.num_workers,
        "num_runs": args.num_runs,
    }
    run_pipeline(cfg, args.reuse_predictions)


if __name__ == "__main__":
    main()
