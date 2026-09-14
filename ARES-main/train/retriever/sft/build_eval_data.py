from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import torch

DATASET_CHOICES = ("webqsp", "cwq")
SPLIT_CHOICES = ("train", "val", "test")
DEFAULT_TOP_K = 100

_SCRIPT_DIR = Path(__file__).resolve().parent
_ARES_MAIN_ROOT = _SCRIPT_DIR.parents[2]
_INFERENCE_ROOT = _ARES_MAIN_ROOT / "retrieve" / "stage2_results" / "inference"
_OUTPUT_ROOT = _SCRIPT_DIR / "eval_data"

RESULT_FILE_BY_SPLIT = {
    "train": "retrieval_result_train.pth",
    "val": "retrieval_result_val.pth",
    "test": "retrieval_result.pth",
}

OUTPUT_FILE_BY_SPLIT = {
    "train": "sub100train.json",
    "val": "sub100val.json",
    "test": "sub100.json",
}


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


def normalize_entity_list(value: Any) -> List[str]:
    return [safe_text(item) for item in ensure_list(value) if safe_text(item)]


def normalize_triple(raw_triple: Sequence[Any]) -> Tuple[str, str, str]:
    if len(raw_triple) < 3:
        raise ValueError(f"Expected triple with at least three elements, got: {raw_triple}")
    return safe_text(raw_triple[0]), safe_text(raw_triple[1]), safe_text(raw_triple[2])


def dedupe_top_k_triples(scored_triples: Sequence[Sequence[Any]], top_k: int) -> List[List[str]]:
    seen: set[Tuple[str, str, str]] = set()
    triplets: List[List[str]] = []
    for raw in scored_triples or []:
        head, relation, tail = normalize_triple(raw)
        key = (head, relation, tail)
        if key in seen:
            continue
        seen.add(key)
        triplets.append([head, relation, tail])
        if len(triplets) >= top_k:
            break
    return triplets


def resolve_inference_file(dataset: str, split: str) -> Path:
    path = _INFERENCE_ROOT / dataset / RESULT_FILE_BY_SPLIT[split]
    if path.is_file():
        return path
    if split == "test":
        alt = _INFERENCE_ROOT / dataset / "retrieval_result_test.pth"
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Missing inference file for {dataset}/{split}: {path}")


def has_answer(sample: Dict[str, Any]) -> bool:
    return bool(normalize_entity_list(sample.get("a_entity_in_graph") or sample.get("a_entity")))


def convert_sample(sample_id: str, sample: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    question = safe_text(sample.get("question"))
    topic_entities = normalize_entity_list(sample.get("q_entity_in_graph") or sample.get("q_entity"))
    triplets = dedupe_top_k_triples(sample.get("scored_triples") or [], top_k=top_k)
    return {
        "id": safe_text(sample.get("id")) or sample_id,
        "question": question,
        "topic_entities": topic_entities,
        "triplets": triplets,
    }


def convert_split(dataset: str, split: str, top_k: int, require_answer: bool = False) -> Dict[str, Any]:
    inference_path = resolve_inference_file(dataset, split)
    payload = torch.load(inference_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {inference_path}, got {type(payload)}")

    output: Dict[str, Any] = {}
    skipped_without_answer = 0
    for sample_id, sample in payload.items():
        if not isinstance(sample, dict):
            continue
        if require_answer and not has_answer(sample):
            skipped_without_answer += 1
            continue
        output[str(sample_id)] = convert_sample(str(sample_id), sample, top_k=top_k)
    if require_answer and skipped_without_answer:
        print(f"  skipped {skipped_without_answer} test samples without answers")
    return output


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_dataset(dataset: str, splits: Sequence[str], top_k: int, output_root: Path) -> None:
    out_dir = output_root / dataset
    for split in splits:
        require_answer = split == "test"
        payload = convert_split(dataset, split, top_k=top_k, require_answer=require_answer)
        out_path = out_dir / OUTPUT_FILE_BY_SPLIT[split]
        write_json(out_path, payload)
        print(
            f"[{dataset}/{split}] wrote {out_path} "
            f"(samples={len(payload)}, top_k={top_k}"
            f"{', require_answer' if require_answer else ''})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sub100-style JSON from stage2 inference retrieval_result*.pth."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_CHOICES,
        default=list(DATASET_CHOICES),
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_CHOICES,
        default=list(SPLIT_CHOICES),
        help="Splits to convert.",
    )
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Max triplets per sample.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(_OUTPUT_ROOT),
        help="Root directory for output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("error: --top_k must be positive")

    output_root = Path(args.output_root)
    for dataset in args.datasets:
        build_dataset(dataset, args.splits, top_k=args.top_k, output_root=output_root)


if __name__ == "__main__":
    main()
