import io
import logging
import re
from collections import defaultdict

from PIL import Image

eval_logger = logging.getLogger("lmms-eval")


def msr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Compose the prompt text for MMSI-Bench."""
    question = doc["question"].strip()
    if not lmms_eval_specific_kwargs:
        return question
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    if pre_prompt:
        question = f"{pre_prompt}{question}"
    if post_prompt:
        question = f"{question}{post_prompt}"
    return question


def msr_doc_to_visual(doc):
    """Load image payloads and convert them to RGB PIL images."""
    image_list = []
    for payload in doc["images"]:
        if isinstance(payload, Image.Image):
            image = payload
        elif isinstance(payload, dict) and "bytes" in payload:
            image = Image.open(io.BytesIO(payload["bytes"]))
        elif isinstance(payload, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(payload))
        else:
            raise TypeError(f"Unsupported image payload type: {type(payload)}")
        image_list.append(image.convert("RGB"))
    return image_list


def extract_single_choice_with_word_boundary(pred, gt):
    """Extract a single-choice answer wrapped in backticks and compare."""
    pattern_1 = r"``([^`]*)``"
    match = re.search(pattern_1, pred)
    if match:
        pred = match.group(1)

    pattern_2 = r"`([^`]*)`"
    match = re.search(pattern_2, pred)
    if match:
        pred = match.group(1)

    pattern_add = r"\{([^}]*)\}"
    match = re.search(pattern_add, pred)
    if match:
        pred = match.group(1)

    pattern_3 = r"\b[A-D]\b(?!\s[a-zA-Z])"
    match = re.search(pattern_3, pred)
    if match:
        pred = match.group()
    else:
        return None

    answer = gt.lower().replace("\n", " ").strip()
    predict = pred.lower().replace("\n", " ").strip()
    try:
        # direct match on single letter
        if answer == predict[0]:
            return 1.0
        if predict[0] == "(" and len(predict) > 1 and answer == predict[1]:
            return 1.0
        if predict.startswith("option ") and len(predict) > 7 and answer == predict[7]:
            return 1.0
        if predict.startswith("the answer is ") and len(predict) > 14 and answer == predict[14]:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def msr_process_results(doc, results):
    """Score a model response for a single MMSI-Bench example."""
    pred = results[0]
    gt = doc["answer"]

    score = extract_single_choice_with_word_boundary(pred, gt)
    category = doc["question_type"]
    l2_category = doc["question_type"]
    if score is None:
        return {
            category: {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"},
            "average": {"question_id": doc["id"], "l2_category": l2_category, "score": 0, "note": "cannot find answer"},
        }
    return {
        category: {"question_id": doc["id"], "l2_category": l2_category, "score": score},
        "average": {"question_id": doc["id"], "l2_category": l2_category, "score": score},
    }


def msr_aggregate_results(results):
    """Aggregate scores across MMSI-Bench sub-categories."""
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    all_scores = [score for scores in l2_category_scores.values() for score in scores]
    return sum(all_scores) / len(all_scores) if all_scores else 0.0
