#!/usr/bin/env python3
"""
Create training subsets from a generated synthetic corpus.

For each subset, samples are drawn at random from the corpus until the target
token budget is reached. One source sample = one entry. Samples may repeat
across subsets and (if the corpus is smaller than the budget) within a subset.

Usage:
    python create_subsets.py \\
        --data_dir synthetic_data/qwen2.5-0.5b/temp1_25 \\
        --num_subsets 3 \\
        --tokens_per_subset 5000000

Input file (must exist in --data_dir):
    texts.json

Outputs (written to --data_dir/subsets/):
    subset_v{i}.json            training file ([{"text": ...}, ...])
    summary.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_synthetic_data(data_dir: Path) -> Tuple[List[str], dict]:
    """Load texts from data_dir/texts.json (with train.json fallback)."""

    candidates = ["texts.json", "train.json"]
    texts_file = None
    for candidate in candidates:
        p = data_dir / candidate
        if p.exists():
            texts_file = p
            break

    if texts_file is None:
        raise FileNotFoundError(
            f"No data file found in {data_dir}. Tried: {candidates}"
        )

    print(f"Loading texts from {texts_file}...")
    with open(texts_file) as f:
        texts_data = json.load(f)

    if isinstance(texts_data, list):
        if texts_data and isinstance(texts_data[0], dict):
            texts = [t.get("text", t.get("content", "")) for t in texts_data]
        else:
            texts = texts_data
    else:
        raise ValueError(f"Unknown texts format: {type(texts_data)}")

    print(f"  Loaded {len(texts):,} texts")

    metadata_file = data_dir / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    return texts, metadata


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    return int(len(text) / chars_per_token)


def create_pt_subsets(texts, num_subsets, tokens_per_subset, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_indices = list(range(len(texts)))

    print(f"\n{'='*60}")
    print(f"CREATING SUBSETS")
    print(f"{'='*60}")
    print(f"Total source samples: {len(texts):,}")
    print(f"Subsets to create:    {num_subsets}")
    print(f"Tokens per subset:    {tokens_per_subset:,}")

    subsets_info = []
    for subset_idx in range(num_subsets):
        samples = []
        total_tokens = 0

        while total_tokens < tokens_per_subset:
            idx = random.choice(all_indices)
            text = texts[idx]
            samples.append({"text": text})
            total_tokens += estimate_tokens(text)

        subset_file = output_dir / f"subset_v{subset_idx + 1}.json"
        with open(subset_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False)

        info = {
            'subset_idx': subset_idx + 1,
            'num_samples': len(samples),
            'total_tokens': total_tokens,
            'file': subset_file.name
        }
        subsets_info.append(info)
        print(f"  Subset {subset_idx+1}: {len(samples):,} samples, ~{total_tokens:,} tokens")

    summary = {
        'num_subsets': num_subsets,
        'tokens_per_subset': tokens_per_subset,
        'source_samples': len(texts),
        'subsets': subsets_info,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir}/summary.json")
    return subsets_info


def _parse_token_count(s: str) -> int:
    """Accept '5M', '500K', '1500000', etc."""
    s = s.strip()
    if s[-1] in "Mm":
        return int(float(s[:-1]) * 1_000_000)
    if s[-1] in "Kk":
        return int(float(s[:-1]) * 1_000)
    return int(s)


def main():
    parser = argparse.ArgumentParser(
        description="Create training subsets from a generated synthetic corpus."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing texts.json")
    parser.add_argument("--num_subsets", type=int, default=3,
                        help="Number of subsets to create")
    parser.add_argument("--tokens_per_subset", type=_parse_token_count, default="5M",
                        help="Target tokens per subset. Accepts '5M', '500K', "
                             "or a raw integer.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    texts, metadata = load_synthetic_data(data_dir)

    total_tokens_est = int(sum(len(t) for t in texts) / 4.0)
    print(f"\nData statistics:")
    print(f"  Samples: {len(texts):,}")
    print(f"  Estimated total tokens: {total_tokens_est:,}")

    output_dir = data_dir / "subsets"

    create_pt_subsets(
        texts=texts,
        num_subsets=args.num_subsets,
        tokens_per_subset=args.tokens_per_subset,
        output_dir=output_dir,
    )

    print(f"\n{'='*60}")
    print("COMPLETE!")
    print(f"{'='*60}")
    print(f"Created {args.num_subsets} subsets in {output_dir}")


if __name__ == "__main__":
    main()
