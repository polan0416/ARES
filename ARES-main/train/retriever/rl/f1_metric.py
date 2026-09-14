from __future__ import annotations

import re
import string
from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple


def normalize(text: str) -> str:
    value = (text or "").lower()
    exclude = set(string.punctuation)
    value = "".join(char for char in value if char not in exclude)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"\b(<pad>)\b", " ", value)
    return " ".join(value.split())


def match(prediction: str, answer: str) -> bool:
    return normalize(answer) in normalize(prediction)


def remove_duplicates(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def get_pred(prediction: str, split: str | None = None) -> List[str]:
    if split is not None:
        return prediction.split(split)
    values = [line for line in (prediction or "").split("\n") if "ans:" in line and "none" not in line.lower()]
    if values:
        values = [
            line
            for line in values
            if "ans: not available" not in line.lower() and "ans: no information available" not in line.lower()
        ]
    return remove_duplicates(values)


def eval_recall(prediction: Sequence[str], answer: Sequence[str], double_check: bool = True) -> Tuple[float, float, int]:
    prediction_copy = sorted(deepcopy(list(prediction)), key=len, reverse=True)
    matched = 0.0
    gold_answers = list(answer)
    if not gold_answers:
        return 0.0, 0.0, 0
    for gold in gold_answers:
        for pred in list(prediction_copy):
            if match(pred, gold):
                matched += 1
                prediction_copy.remove(pred)
                break
            if double_check and (match(gold, pred.split("ans:")[-1].strip()) or match(gold, pred)):
                matched += 1
                prediction_copy.remove(pred)
                break
    return matched / len(gold_answers), matched, len(gold_answers)


def eval_precision(prediction: Sequence[str], answer: Sequence[str], double_check: bool = True) -> Tuple[float, float, int]:
    prediction_copy = sorted(deepcopy(list(prediction)), key=len, reverse=True)
    if not prediction_copy:
        return 0.0, 0.0, 0
    matched = 0.0
    for gold in list(answer):
        for pred in list(prediction_copy):
            if match(pred, gold):
                matched += 1
                prediction_copy.remove(pred)
                break
            if double_check and (match(gold, pred.split("ans:")[-1].strip()) or match(gold, pred)):
                matched += 1
                prediction_copy.remove(pred)
                break
    return matched / len(prediction), matched, len(prediction)


def eval_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def answer_f1(prediction_text: str, gold_answers: Sequence[str], double_check: bool = True) -> dict:
    parsed_answers = get_pred(prediction_text or "")
    precision, matched_pred, num_pred = eval_precision(parsed_answers, gold_answers, double_check=double_check)
    recall, matched_gold, num_gold = eval_recall(parsed_answers, gold_answers, double_check=double_check)
    f1 = eval_f1(precision, recall)
    return {
        "prediction_lines": parsed_answers,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_predictions": matched_pred,
        "num_predictions": num_pred,
        "matched_gold": matched_gold,
        "num_gold": num_gold,
    }
