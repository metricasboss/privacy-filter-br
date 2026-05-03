#!/usr/bin/env python3
"""
Generate the Privacy Filter BR training dataset.

Usage:
    python scripts/generate_dataset.py --n 11000 --output data/dataset_br.jsonl
    python scripts/generate_dataset.py --n 11000 --output data/dataset_br.jsonl  # resumes if file exists

Writes each example to disk immediately — safe to kill and resume at any time.
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import generate_example, GeneratorStats
from src.haiku import HaikuGenerator


def count_existing(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="Generate BR PII dataset")
    parser.add_argument("--n", type=int, default=11000, help="Target valid examples")
    parser.add_argument("--output", type=str, default="data/dataset_br.jsonl")
    parser.add_argument("--holdout-ratio", type=float, default=0.09)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    holdout_path = args.output.replace(".jsonl", "_holdout.jsonl")

    # Resume: count how many we already have
    already_done = count_existing(args.output) + count_existing(holdout_path)
    target = args.n
    remaining = target - already_done

    if already_done > 0:
        print(f"Resuming: {already_done} examples already on disk, need {remaining} more")
    else:
        print(f"Generating {target} valid examples → {args.output}")

    if remaining <= 0:
        print("Already complete!")
        return

    haiku = HaikuGenerator()
    stats = GeneratorStats()

    # Write directly to disk, appending
    with open(args.output, "a", encoding="utf-8") as train_f, \
         open(holdout_path, "a", encoding="utf-8") as holdout_f:

        while stats.valid < remaining:
            example = generate_example(haiku)
            stats.record(example, None if example else "validation_failed")

            if example:
                # Route ~9% to holdout, rest to train
                if random.random() < args.holdout_ratio:
                    holdout_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    holdout_f.flush()
                else:
                    train_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    train_f.flush()

            status = "✓" if example else "✗"
            rate = stats.valid / stats.total * 100 if stats.total > 0 else 0
            total_so_far = already_done + stats.valid
            print(f"  [{status}] attempt {stats.total:4d} | valid {total_so_far:5d}/{target} "
                  f"| {rate:.0f}% acceptance | discarded {stats.discarded}",
                  flush=True)

    final_train = count_existing(args.output)
    final_holdout = count_existing(holdout_path)
    print(f"\nDone!")
    print(f"  Train: {final_train} examples → {args.output}")
    print(f"  Holdout: {final_holdout} examples → {holdout_path}")
    print(f"  This run: {stats.total} attempts ({stats.discarded} discarded)")


if __name__ == "__main__":
    main()
