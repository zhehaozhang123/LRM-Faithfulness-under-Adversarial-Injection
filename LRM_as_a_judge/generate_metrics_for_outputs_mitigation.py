import os
import json
import glob
from typing import List
from utils import label_from_scores, compute_classification_metrics, count_tokens_openai

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_mitigation")


def load_records(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    return records


def compute_metrics(records: List[dict]) -> dict:
    completed = [r for r in records if "pred_scores" in r]
    if not completed:
        return {
            "overall": {
                "agreement": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "total": 0,
                "total_examples": len(records),
                "total_tokens": 0,
                "avg_tokens_per_example": 0.0,
            }
        }

    total = len(completed)
    gt_labels = [label_from_scores(r["gt_scores"][0], r["gt_scores"][1]) for r in completed]
    pred_labels = [label_from_scores(r["pred_scores"][0], r["pred_scores"][1]) for r in completed]
    metrics = compute_classification_metrics(gt_labels, pred_labels)

    total_tokens = 0
    for r in completed:
        reasoning = r.get("llm_reasoning", "") or ""
        output = r.get("llm_output", "") or ""
        combined = f"{reasoning} {output}".strip()
        if combined:
            total_tokens += count_tokens_openai(combined)
    avg_tokens = total_tokens / total if total > 0 else 0.0

    return {
        "overall": {
            "agreement": metrics["agreement"] * 100,
            "precision": metrics["precision"] * 100,
            "recall": metrics["recall"] * 100,
            "f1": metrics["f1"] * 100,
            "total": total,
            "total_examples": len(records),
            "total_tokens": total_tokens,
            "avg_tokens_per_example": avg_tokens,
        }
    }


essential_fields = ("question_body", "answer1_body", "answer2_body", "gt_scores")


def has_required_fields(records: List[dict]) -> bool:
    if not records:
        return False
    r0 = records[0]
    return all(field in r0 for field in essential_fields)


def main():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")))
    if not files:
        print(f"No JSONL files found in {OUTPUT_DIR}")
        return

    for path in files:
        metrics_path = path.replace(".jsonl", "_metrics.json")
        # Skip if already exists
        if os.path.exists(metrics_path):
            print(f"Metrics already exist, skipping: {os.path.basename(metrics_path)}")
            continue

        print(f"Processing: {os.path.basename(path)}")
        records = load_records(path)
        if not has_required_fields(records):
            print(f"  Warning: Missing essential fields, skipping metrics for this file.")
            # Still write a placeholder to mark as processed
            placeholder = {
                "overall": {
                    "agreement": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "total": 0,
                    "total_examples": len(records),
                    "total_tokens": 0,
                    "avg_tokens_per_example": 0.0,
                },
                "note": "Essential fields missing or file malformed."
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(placeholder, f, indent=2)
            continue

        metrics = compute_metrics(records)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Wrote: {os.path.basename(metrics_path)}")


if __name__ == "__main__":
    main()
