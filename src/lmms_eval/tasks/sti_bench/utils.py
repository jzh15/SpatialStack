import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml
from loguru import logger as eval_logger

ANSWER_PATTERNS: List[re.Pattern] = [
    re.compile(r"\(([A-E])\)", flags=re.IGNORECASE),
    re.compile(r"Answer\s*[:=]\s*['\"]?([A-E])['\"]?", flags=re.IGNORECASE),
    re.compile(r"Ans\s*=\s*['\"]?([A-E])['\"]?", flags=re.IGNORECASE),
    re.compile(r"Option\s+([A-E])", flags=re.IGNORECASE),
    re.compile(r"\b([A-E])\s*(?:is|was)\s*correct\b", flags=re.IGNORECASE),
    re.compile(r"\b([A-E])[\.\)]\s*$", flags=re.IGNORECASE | re.MULTILINE),
    re.compile(r"\b([A-E])\b"),
]

_YAML_PATH = Path(__file__).parent / "sti_bench.yaml"
with _YAML_PATH.open("r") as yaml_file:
    raw_lines = [line for line in yaml_file.readlines() if "!function" not in line]
CONFIG = yaml.safe_load("".join(raw_lines))

_HF_HOME = Path(os.getenv("HF_HOME", "~/.cache/huggingface/")).expanduser()
_DEFAULT_CACHE_DIR = _HF_HOME / CONFIG["dataset_kwargs"]["cache_dir"]


def _candidate_video_roots() -> Iterable[Path]:
    env_root = os.getenv("STI_BENCH_VIDEO_DIR")
    if env_root:
        yield Path(env_root)

    dataset_path = CONFIG.get("dataset_path")
    if dataset_path and os.path.isdir(dataset_path):
        candidate = Path(dataset_path)
        yield candidate
        yield candidate / "video"
        yield candidate / "videos"

    data_dir_override = CONFIG["dataset_kwargs"].get("data_dir")
    if data_dir_override and os.path.isdir(data_dir_override):
        candidate = Path(data_dir_override)
        yield candidate
        yield candidate / "video"
        yield candidate / "videos"

    yield _DEFAULT_CACHE_DIR / "video"
    yield _DEFAULT_CACHE_DIR / "videos"


def _resolve_video_path(filename: str) -> str:
    for root in _candidate_video_roots():
        video_path = (root / filename).expanduser()
        if video_path.exists():
            return str(video_path)
    raise FileNotFoundError(f"Could not locate video file '{filename}'. "
                            f"Checked roots: {', '.join(str(p) for p in _candidate_video_roots())}")


def sti_bench_doc_to_visual(doc: Dict, *_args, **_kwargs) -> List[str]:
    video_name = doc.get("Video")
    if not video_name:
        raise ValueError("Missing 'Video' field in STI-Bench sample.")
    return [_resolve_video_path(video_name)]


def sti_bench_doc_to_text(doc: Dict, lmms_eval_specific_kwargs: Optional[Dict] = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "Answer with the option letter (A-E) only.")

    time_window = f"Time span: {doc.get('time_start', '')}s → {doc.get('time_end', '')}s."
    extra_prompt = doc.get("Prompt", "")
    question = doc.get("Question", "")

    candidates = doc.get("Candidates") or {}
    candidate_lines = [
        f"({key}) {value}"
        for key, value in candidates.items()
        if value
    ]
    options_block = "\n".join(candidate_lines)

    segments = [segment for segment in [pre_prompt, time_window, extra_prompt, question, options_block, post_prompt] if segment]
    return "\n".join(segments)


def _extract_option_letter(text: str) -> Optional[str]:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    text = text.strip().upper()
    if text and text[-1] in "ABCDE":
        return text[-1]
    return None


def sti_bench_process_results(doc: Dict, results: List[str]) -> Dict[str, Dict]:
    if not results:
        raise ValueError("Results list is empty for STI-Bench evaluation.")
    raw_prediction = results[0]
    pred_letter = _extract_option_letter(raw_prediction or "")
    target_letter = (doc.get("Answer") or "").strip().upper()

    record = {
        "id": doc.get("ID"),
        "video": doc.get("Video"),
        "task": doc.get("Task"),
        "scene": doc.get("scene") or doc.get("Scene"),
        "source": doc.get("Source"),
        "prediction": pred_letter,
        "ground_truth": target_letter,
        "raw_prediction": raw_prediction,
        "correct": int(pred_letter == target_letter) if pred_letter and target_letter else 0,
    }

    if pred_letter is None:
        eval_logger.debug(f"Unable to parse option letter from prediction: '{raw_prediction}'")

    return {"sti_bench_accuracy": record}


def sti_bench_aggregate_results(results: List[Dict]) -> float:
    if not results:
        eval_logger.warning("No STI-Bench results to aggregate; returning 0.")
        return 0.0

    df = pd.DataFrame(results)

    if "correct" not in df:
        raise ValueError("Aggregated STI-Bench results lack 'correct' field.")

    for column in ("task", "scene", "source"):
        if column in df.columns and df[column].notna().any():
            breakdown = df.groupby(column)["correct"].mean().sort_values(ascending=False)
            eval_logger.info(f"STI-Bench accuracy by {column}: {breakdown.to_dict()}")

    overall = df["correct"].mean()
    eval_logger.info(f"STI-Bench overall accuracy: {overall:.4f}")
    return overall * 100.0
