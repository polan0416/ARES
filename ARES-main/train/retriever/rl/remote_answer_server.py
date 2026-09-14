from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib import error, request

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

_THIS_DIR = Path(__file__).resolve().parent
_ARES_MAIN_ROOT = _THIS_DIR.parents[2]
_ARES_ROOT = _THIS_DIR.parents[3]
_REASON_DIR = _ARES_MAIN_ROOT / "reason"
if str(_ARES_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARES_MAIN_ROOT))
if str(_REASON_DIR) not in sys.path:
    sys.path.insert(0, str(_REASON_DIR))

from reason.main import build_single_reason_prompt_record
from reason.metrics.evaluate_results_corrected import eval_single_prediction_corrected, get_pred

DEFAULT_BASE_MODEL_PATH = str(Path("~/.cache/huggingface/model/Meta-Llama-3.1-8B-Instruct").expanduser())
DEFAULT_ADAPTER_PATH = str(_ARES_ROOT / "checkpoint/webqsp/kgqa_answer/llama31_8b_sft")
DEFAULT_MODEL_NAME = str(_ARES_ROOT / "checkpoint/webqsp/kgqa_answer/llama31_8b_sft_merged")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_PROMPT_MODE = "scored_200"
DEFAULT_LLM_MODE = "sys_icl_dc"
DEFAULT_EVAL_MODE = "kbqa_scored_200_corrected"
DEFAULT_VLLM_API_BASE = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.88
DEFAULT_VLLM_MAX_NUM_SEQS = 6
DEFAULT_VLLM_MAX_MODEL_LEN = 4096
DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS = 8192


class AnswerGenerator(Protocol):
    backend: str
    base_model_path: str
    adapter_path: str
    model_name: str
    prompt_mode: str
    llm_mode: str
    max_new_tokens: int
    device: str
    dtype: str

    def generate_reason_aligned(
        self,
        sample_id: str,
        question: str,
        question_explain: Dict[str, Any],
        scored_triplets: Sequence[Sequence[Any]],
        gold_answers: Sequence[str],
        prompt_mode: str,
        llm_mode: str,
        score_threshold: float,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]: ...


def _apply_chat_template(tokenizer: Any, messages: Sequence[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)


def _needs_dc_second_turn(raw_output: str, llm_mode: str) -> bool:
    if "dc" not in (llm_mode or ""):
        return False
    lowered = (raw_output or "").lower()
    return (
        "ans:" not in lowered
        or "ans: not available" in lowered
        or "ans: no information available" in lowered
    )


@dataclass
class ReasonAlignedMixin:
    """Shared KBQA prompt building and sys_icl_dc second-turn logic."""

    base_model_path: str
    model_name: str
    prompt_mode: str
    llm_mode: str
    max_new_tokens: int
    backend: str
    adapter_path: str = ""
    device: str = "unknown"
    dtype: str = "unknown"

    def _chat_generate(self, messages: Sequence[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    def generate_reason_aligned(
        self,
        sample_id: str,
        question: str,
        question_explain: Dict[str, Any],
        scored_triplets: Sequence[Sequence[Any]],
        gold_answers: Sequence[str],
        prompt_mode: str,
        llm_mode: str,
        score_threshold: float,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt_record = build_single_reason_prompt_record(
            sample_id=sample_id,
            question=question,
            scored_triplets=scored_triplets,
            prompt_mode=prompt_mode,
            model_name=self.model_name,
            llm_mode=llm_mode,
            thres=score_threshold,
            question_explain=question_explain,
            ground_truth=gold_answers,
        )
        messages: List[Dict[str, str]] = []
        if llm_mode and "sys" in llm_mode:
            messages.append({"role": "system", "content": prompt_record.get("sys_query") or ""})
        messages.append({"role": "user", "content": prompt_record.get("user_query") or ""})

        started_at = time.time()
        raw_output = self._chat_generate(messages=messages, max_new_tokens=max_new_tokens)
        cot_prediction = ""
        if _needs_dc_second_turn(raw_output, llm_mode):
            dc_messages = list(messages)
            dc_messages.append({"role": "assistant", "content": raw_output})
            dc_messages.append({"role": "user", "content": prompt_record.get("cot_query") or ""})
            cot_prediction = self._chat_generate(messages=dc_messages, max_new_tokens=max_new_tokens)
            raw_output = cot_prediction

        latency_ms = round((time.time() - started_at) * 1000.0, 3)
        return {
            "raw_output": raw_output,
            "prediction_text": raw_output,
            "prediction_lines": get_pred(raw_output or ""),
            "cot_prediction": cot_prediction,
            "prompt_record": prompt_record,
            "latency_ms": latency_ms,
            "prompt_char_count": len(prompt_record.get("all_query") or prompt_record.get("user_query") or ""),
        }


@dataclass
class HfAnswerGenerator(ReasonAlignedMixin):
    adapter_path: str = DEFAULT_ADAPTER_PATH

    def __post_init__(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.backend = "hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = "torch.bfloat16" if torch.cuda.is_available() else "torch.float32"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self.model = model.to(self.device)
        self._lock = threading.Lock()

    def _chat_generate(self, messages: Sequence[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        import torch

        prompt = _apply_chat_template(self.tokenizer, messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        limit = int(max_new_tokens or self.max_new_tokens)
        with self._lock:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=limit,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@dataclass
class VllmApiAnswerGenerator(ReasonAlignedMixin):
    """Call a vLLM OpenAI-compatible server (vllm serve). No app-level lock; vLLM schedules requests."""

    vllm_api_base: str = DEFAULT_VLLM_API_BASE
    vllm_served_model_name: str = ""
    request_timeout_sec: float = 300.0

    def __post_init__(self) -> None:
        from transformers import AutoTokenizer

        self.backend = "vllm-api"
        self.device = "vllm"
        self.dtype = "bfloat16"
        self.adapter_path = ""
        self.vllm_api_base = (self.vllm_api_base or DEFAULT_VLLM_API_BASE).rstrip("/")
        self.vllm_served_model_name = self.vllm_served_model_name or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _openai_chat_completion(
        self,
        messages: Sequence[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.vllm_served_model_name,
            "messages": list(messages),
            "max_tokens": int(max_new_tokens or self.max_new_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        url = f"{self.vllm_api_base}/chat/completions"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.request_timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM chat completion failed ({exc.code}): {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"vLLM chat completion unreachable at {url}: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {body}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip()

    def _chat_generate(self, messages: Sequence[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        return self._openai_chat_completion(messages=messages, max_new_tokens=max_new_tokens)


@dataclass
class VllmInlineAnswerGenerator(ReasonAlignedMixin):
    """In-process vLLM LLM engine (single process, no separate vllm serve)."""

    gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_VLLM_MAX_NUM_SEQS
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    max_num_batched_tokens: int = DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS
    enable_prefix_caching: bool = True

    def __post_init__(self) -> None:
        from vllm import LLM

        self.backend = "vllm-inline"
        self.device = "cuda"
        self.dtype = "bfloat16"
        self.adapter_path = ""
        self.llm = LLM(
            model=self.model_name,
            tokenizer=self.base_model_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            max_model_len=int(self.max_model_len),
            max_num_seqs=int(self.max_num_seqs),
            max_num_batched_tokens=int(self.max_num_batched_tokens),
            enable_prefix_caching=bool(self.enable_prefix_caching),
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _chat_generate(self, messages: Sequence[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
        from vllm import SamplingParams

        prompt = _apply_chat_template(self.tokenizer, messages)
        limit = int(max_new_tokens or self.max_new_tokens)
        outputs = self.llm.generate(
            [prompt],
            SamplingParams(max_tokens=limit, temperature=0.0, top_p=1.0),
        )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()


class AnswerRequest(BaseModel):
    sample_id: str = ""
    question: str = ""
    question_explain: Dict[str, Any] = Field(default_factory=dict)
    topic_entity: str = ""
    triples: List[List[str]] = Field(default_factory=list)
    candidate_triples: List[Dict[str, Any]] = Field(default_factory=list)
    gold_answers: List[str] = Field(default_factory=list)
    trajectory_id: str = ""
    prompt_mode: str = DEFAULT_PROMPT_MODE
    llm_mode: str = DEFAULT_LLM_MODE
    eval_mode: str = DEFAULT_EVAL_MODE
    score_threshold: float = 0.0
    a_entity_in_graph: Optional[bool] = None
    max_new_tokens: Optional[int] = None


def _normalize_triple(triple: Sequence[Any]) -> Optional[Tuple[str, str, str]]:
    if not isinstance(triple, (list, tuple)) or len(triple) < 3:
        return None
    head = str(triple[0]).strip()
    relation = str(triple[1]).strip()
    tail = str(triple[2]).strip()
    if not head or not relation or not tail:
        return None
    return head, relation, tail


def _candidate_score_map(candidate_triples: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    score_map: Dict[Tuple[str, str, str], float] = {}
    for rank, candidate in enumerate(candidate_triples or [], start=1):
        tri = _normalize_triple(candidate.get("triple") or [candidate.get("head"), candidate.get("relation"), candidate.get("tail")])
        if tri is None:
            continue
        raw_score = candidate.get("score", candidate.get("original_rank", rank))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = float(rank)
        prev = score_map.get(tri)
        score_map[tri] = score if prev is None else max(prev, score)
    return score_map


def build_retrieval_evidence(
    selected_triples: Sequence[Sequence[Any]],
    candidate_triples: Sequence[Dict[str, Any]],
    default_score: float = 1.0,
) -> List[Tuple[str, str, str, float]]:
    score_map = _candidate_score_map(candidate_triples)
    seen: set[Tuple[str, str, str]] = set()
    out: List[Tuple[str, str, str, float]] = []
    for triple in selected_triples or []:
        tri = _normalize_triple(triple)
        if tri is None or tri in seen:
            continue
        seen.add(tri)
        score = score_map.get(tri, default_score)
        out.append((tri[0], tri[1], tri[2], float(score)))
    return out


class AnswerServer:
    def __init__(self, generator: AnswerGenerator):
        self.generator = generator
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/health")
        def health() -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "ok": True,
                "backend": self.generator.backend,
                "device": self.generator.device,
                "dtype": self.generator.dtype,
                "base_model_path": self.generator.base_model_path,
                "adapter_path": self.generator.adapter_path,
                "model_name": self.generator.model_name,
                "prompt_mode": self.generator.prompt_mode,
                "llm_mode": self.generator.llm_mode,
            }
            if isinstance(self.generator, VllmApiAnswerGenerator):
                payload["vllm_api_base"] = self.generator.vllm_api_base
                payload["vllm_served_model_name"] = self.generator.vllm_served_model_name
            if isinstance(self.generator, VllmInlineAnswerGenerator):
                payload["vllm_gpu_memory_utilization"] = self.generator.gpu_memory_utilization
                payload["vllm_max_num_seqs"] = self.generator.max_num_seqs
                payload["vllm_max_model_len"] = self.generator.max_model_len
            return payload

        @self.app.post("/answer")
        def answer(answer_request: AnswerRequest) -> Dict[str, Any]:
            retrieval_evidence = build_retrieval_evidence(
                selected_triples=answer_request.triples,
                candidate_triples=answer_request.candidate_triples,
            )
            result = self.generator.generate_reason_aligned(
                sample_id=answer_request.sample_id or answer_request.trajectory_id,
                question=answer_request.question,
                question_explain=answer_request.question_explain,
                scored_triplets=retrieval_evidence,
                gold_answers=answer_request.gold_answers,
                prompt_mode=answer_request.prompt_mode or self.generator.prompt_mode,
                llm_mode=answer_request.llm_mode or self.generator.llm_mode,
                score_threshold=float(answer_request.score_threshold or 0.0),
                max_new_tokens=answer_request.max_new_tokens,
            )
            metrics = eval_single_prediction_corrected(
                prediction_text=result["prediction_text"],
                answer=answer_request.gold_answers,
                question=answer_request.question,
                retrieved_triplets=[(tri[0], tri[1], tri[2]) for tri in retrieval_evidence],
                a_entity_in_graph=answer_request.a_entity_in_graph,
            )
            return {
                "ok": True,
                "trajectory_id": answer_request.trajectory_id,
                "sample_id": answer_request.sample_id,
                "topic_entity": answer_request.topic_entity,
                "n_triples": len(answer_request.triples or []),
                "prompt_mode_used": answer_request.prompt_mode or self.generator.prompt_mode,
                "llm_mode_used": answer_request.llm_mode or self.generator.llm_mode,
                "eval_mode_used": answer_request.eval_mode,
                "retrieval_evidence_used": retrieval_evidence,
                "answer_format_valid": metrics["answer_format_valid"],
                "corrected_metrics": metrics,
                "reward_metric_value": metrics["f1"],
                **result,
            }


def create_generator(args: argparse.Namespace) -> AnswerGenerator:
    if args.backend == "hf":
        return HfAnswerGenerator(
            base_model_path=args.base_model_path,
            adapter_path=args.adapter_path,
            model_name=args.model_name,
            prompt_mode=args.prompt_mode,
            llm_mode=args.llm_mode,
            max_new_tokens=args.max_new_tokens,
        )
    if args.backend == "vllm":
        if args.vllm_inline:
            return VllmInlineAnswerGenerator(
                base_model_path=args.base_model_path,
                model_name=args.model_name,
                prompt_mode=args.prompt_mode,
                llm_mode=args.llm_mode,
                max_new_tokens=args.max_new_tokens,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_num_seqs=args.vllm_max_num_seqs,
                max_model_len=args.vllm_max_model_len,
                max_num_batched_tokens=args.vllm_max_num_batched_tokens,
                enable_prefix_caching=not args.disable_prefix_caching,
            )
        return VllmApiAnswerGenerator(
            base_model_path=args.base_model_path,
            model_name=args.model_name,
            prompt_mode=args.prompt_mode,
            llm_mode=args.llm_mode,
            max_new_tokens=args.max_new_tokens,
            vllm_api_base=args.vllm_api_base,
            vllm_served_model_name=args.vllm_served_model_name or args.model_name,
            request_timeout_sec=args.vllm_request_timeout_sec,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the frozen WebQSP answer model over HTTP.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--backend",
        type=str,
        choices=("hf", "vllm"),
        default="hf",
        help="hf: HuggingFace+PEFT in-process; vllm: vLLM OpenAI API client (default) or --vllm-inline",
    )
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt_mode", type=str, default=DEFAULT_PROMPT_MODE)
    parser.add_argument("--llm_mode", type=str, default=DEFAULT_LLM_MODE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default=DEFAULT_VLLM_API_BASE,
        help="OpenAI-compatible base URL for `vllm serve` (used when --backend vllm and not --vllm-inline)",
    )
    parser.add_argument(
        "--vllm-served-model-name",
        type=str,
        default="",
        help="Model id passed to vLLM chat/completions; defaults to --model_name",
    )
    parser.add_argument(
        "--vllm-request-timeout-sec",
        type=float,
        default=300.0,
        help="HTTP timeout for each vLLM chat completion call",
    )
    parser.add_argument(
        "--vllm-inline",
        action="store_true",
        help="Load vLLM in-process instead of calling an external vllm serve",
    )
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=DEFAULT_VLLM_MAX_NUM_SEQS)
    parser.add_argument("--vllm-max-model-len", type=int, default=DEFAULT_VLLM_MAX_MODEL_LEN)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=DEFAULT_VLLM_MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = create_generator(args)
    server = AnswerServer(generator)
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
