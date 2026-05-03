#!/usr/bin/env python3
"""
Evaluate a fine-tuned Privacy Filter BR model on the holdout dataset.

Usage:
    python scripts/benchmark.py \
        --model privacy-filter-br/ \
        --holdout data/dataset_br_holdout.jsonl \
        --output data/benchmark_results.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn
    }


def spans_to_set(entities: list[dict]) -> set[tuple]:
    return {(e["start"], e["end"], e["label"]) for e in entities}


def evaluate(model_path: str, holdout_path: str) -> dict:
    from transformers import pipeline

    print(f"Loading model from {model_path}...")
    pipe = pipeline("token-classification", model=model_path,
                    aggregation_strategy="first", device=-1)

    per_label_tp = defaultdict(int)
    per_label_fp = defaultdict(int)
    per_label_fn = defaultdict(int)

    with open(holdout_path) as f:
        examples = [json.loads(line) for line in f]

    print(f"Evaluating on {len(examples)} examples...")
    for i, ex in enumerate(examples):
        if i % 100 == 0:
            print(f"  {i}/{len(examples)}...")

        pred_raw = pipe(ex["text"])
        pred_spans = {
            (p["start"], p["end"], p["entity_group"])
            for p in pred_raw
        }
        gold_spans = spans_to_set(ex["entities"])

        for span in gold_spans:
            label = span[2]
            if span in pred_spans:
                per_label_tp[label] += 1
            else:
                per_label_fn[label] += 1

        for span in pred_spans:
            label = span[2]
            if span not in gold_spans:
                per_label_fp[label] += 1

    all_labels = set(per_label_tp) | set(per_label_fp) | set(per_label_fn)
    results = {}
    for label in sorted(all_labels):
        results[label] = compute_f1(
            per_label_tp[label], per_label_fp[label], per_label_fn[label]
        )

    total_tp = sum(per_label_tp.values())
    total_fp = sum(per_label_fp.values())
    total_fn = sum(per_label_fn.values())
    results["__overall__"] = compute_f1(total_tp, total_fp, total_fn)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", default="data/benchmark_results.json")
    args = parser.parse_args()

    results = evaluate(args.model, args.holdout)

    print("\n=== BENCHMARK RESULTS ===")
    for label, metrics in sorted(results.items()):
        status = "OK" if metrics["f1"] >= 0.90 else "FAIL"
        print(f"{status} {label:35s} F1={metrics['f1']:.3f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")

    failing = [label for label, m in results.items()
               if label != "__overall__" and m["f1"] < 0.90]
    if failing:
        print(f"\nWARNING: Categories below F1 0.90: {failing}")
        print("Action: double examples for these categories and retrain.")
    else:
        print("\nSUCCESS: All categories above F1 0.90 — ready for integration!")


if __name__ == "__main__":
    main()
