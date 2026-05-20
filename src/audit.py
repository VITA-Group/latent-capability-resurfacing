#!/usr/bin/env python3
"""
Contamination Audit v2
Scans synthetic corpus for n-gram overlap with benchmark datasets.
Called by contamination_audit_v2.sh with all parameters as arguments.

Usage:
    python contamination_audit_v2.py \
        --corpus_dir synthetic_data/qwen2.5-0.5b/temp1_0 \
        --benchmarks arc_challenge gsm8k hellaswag mmlu truthfulqa_mc2 minerva_math humaneval \
        --n 13 \
        --min_cutoff 0 \
        --save_histogram \
        --num_bins 50 \
        --output_dir synthetic_data_clean/qwen2.5-0.5b/temp1_0
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ============================================================
# NORMALIZATION AND N-GRAM UTILS
# ============================================================

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_ngrams(text: str, n: int) -> set:
    tokens = text.split()
    if len(tokens) < n:
        return set()
    return set(zip(*[tokens[i:] for i in range(n)]))


# ============================================================
# BENCHMARK LOADING
# ============================================================

BENCHMARK_LOADERS = {
    "arc_challenge": lambda: load_arc("ARC-Challenge"),
    "arc_easy":      lambda: load_arc("ARC-Easy"),
    "gsm8k":         lambda: load_gsm8k(),
    "hellaswag":     lambda: load_hellaswag(),
    "mmlu":          lambda: load_mmlu(),
    "truthfulqa_mc2":lambda: load_truthfulqa(),
    "minerva_math":  lambda: load_math(),
    "humaneval":     lambda: load_humaneval(),
}

BENCHMARK_DISPLAY = {
    "arc_challenge":  "ARC-Challenge",
    "arc_easy":       "ARC-Easy",
    "gsm8k":          "GSM8K",
    "hellaswag":      "HellaSwag",
    "mmlu":           "MMLU",
    "truthfulqa_mc2": "TruthfulQA",
    "minerva_math":   "MATH (Minerva)",
    "humaneval":      "HumanEval",
}


def load_arc(config):
    from datasets import load_dataset
    ds = load_dataset("ai2_arc", config, split="test+validation+train", trust_remote_code=True)
    return [ex["question"] + " " + " ".join(ex["choices"]["text"]) for ex in ds]


def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test+train", trust_remote_code=True)
    return [ex["question"] + " " + ex["answer"] for ex in ds]


def load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation+train", trust_remote_code=True)
    return [ex["activity_label"] + " " + ex["ctx"] + " " + " ".join(ex["endings"]) for ex in ds]


def load_mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test+validation+dev", trust_remote_code=True)
    return [ex["question"] + " " + " ".join(ex["choices"]) for ex in ds]


def load_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    return [ex["question"] + " " + " ".join(ex["mc1_targets"]["choices"]) for ex in ds]


def load_math():
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test+train", trust_remote_code=True)
    return [ex["problem"] + " " + ex["solution"] for ex in ds]


def load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    return [ex["prompt"] + " " + ex["canonical_solution"] for ex in ds]


# ============================================================
# BUILD FINGERPRINT SETS
# ============================================================

def build_fingerprint_sets(benchmark_names: list, n: int) -> tuple:
    """
    Returns:
        global_set: union of all benchmark n-grams
        per_bench:  {name: set_of_ngrams}
    """
    print(f"\nBuilding {n}-gram fingerprint sets...")
    global_set = set()
    per_bench = {}

    for name in benchmark_names:
        if name not in BENCHMARK_LOADERS:
            print(f"  WARNING: Unknown benchmark '{name}', skipping")
            continue
        try:
            texts = BENCHMARK_LOADERS[name]()
            bench_set = set()
            for text in tqdm(texts, desc=f"  {BENCHMARK_DISPLAY.get(name, name)}", leave=False):
                bench_set.update(get_ngrams(normalize(text), n))
            global_set.update(bench_set)
            per_bench[name] = bench_set
            print(f"  {BENCHMARK_DISPLAY.get(name, name)}: {len(texts):,} examples → {len(bench_set):,} unique {n}-grams")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    print(f"  Total unique {n}-grams: {len(global_set):,}")
    return global_set, per_bench


# ============================================================
# SCAN CORPUS
# ============================================================

def scan_corpus(texts: list, global_set: set, per_bench: dict, n: int) -> tuple:
    """
    Returns:
        flagged_indices: list of int
        per_bench_flags: {bench_name: set of int}
    """
    print(f"\nScanning {len(texts):,} samples for {n}-gram overlap...")
    flagged_indices = []
    per_bench_flags = defaultdict(set)

    for i, text in enumerate(tqdm(texts, desc="Scanning")):
        sample_ngrams = get_ngrams(normalize(text), n)
        if not sample_ngrams:
            continue
        if sample_ngrams & global_set:
            flagged_indices.append(i)
            for bench_name, bench_set in per_bench.items():
                if sample_ngrams & bench_set:
                    per_bench_flags[bench_name].add(i)

    return flagged_indices, dict(per_bench_flags)


# ============================================================
# HISTOGRAM (online, no raw data stored)
# ============================================================



def compute_histogram(avg_logps: np.ndarray, n_bins: int) -> dict:
    """
    Compute histogram data from avg NLL array.
    Range is auto-set to [1st percentile, 99th percentile].
    Returns a dict with all info needed to replot later.
    """
    lo = float(np.percentile(avg_logps, 1))
    hi = float(np.percentile(avg_logps, 99))
    counts, edges = np.histogram(avg_logps, bins=n_bins, range=(lo, hi))
    return {
        "bin_edges":  edges.tolist(),
        "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "counts":     counts.tolist(),
        "mean":       float(np.mean(avg_logps)),
        "std":        float(np.std(avg_logps)),
        "median":     float(np.median(avg_logps)),
        "p1":         lo,
        "p99":        hi,
        "n_bins":     n_bins,
        "n_samples":  int(len(avg_logps)),
    }


def save_histogram(
    orig_logps: np.ndarray,
    clean_logps: np.ndarray,
    n_bins: int,
    output_dir: Path,
    label: str = "",
):
    """
    Save two histogram plots and their bin data as JSON.

    Plot 1 (nll_histogram_clean.png):
        NLL distribution of the clean corpus only.

    Plot 2 (nll_histogram_comparison.png):
        Original corpus vs clean corpus overlaid,
        with flagged samples shown separately.

    JSON (nll_histogram.json):
        Bin edges, counts, and summary stats for both
        original and clean — so you can replot yourself.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not installed, skipping histogram. pip install matplotlib")
        return

    # Compute histograms for both original and clean
    # Use a shared range across both so bars are comparable
    all_vals = np.concatenate([orig_logps, clean_logps])
    shared_lo = float(np.percentile(all_vals, 1))
    shared_hi = float(np.percentile(all_vals, 99))

    orig_counts,  orig_edges  = np.histogram(orig_logps,  bins=n_bins, range=(shared_lo, shared_hi))
    clean_counts, clean_edges = np.histogram(clean_logps, bins=n_bins, range=(shared_lo, shared_hi))

    # Flagged-only logps (orig minus clean)
    # We do not have flagged indices here, so approximate as orig - clean counts
    flagged_counts = np.maximum(orig_counts - clean_counts, 0)

    orig_centers  = ((orig_edges[:-1]  + orig_edges[1:])  / 2).tolist()
    clean_centers = ((clean_edges[:-1] + clean_edges[1:]) / 2).tolist()
    bar_width = (orig_edges[1] - orig_edges[0]) * 0.85

    # ── Plot 1: clean corpus only ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(clean_centers, clean_counts, width=bar_width, color='steelblue', alpha=0.85, label='Clean corpus')
    ax1.axvline(float(np.mean(clean_logps)), color='red',    linestyle='--', linewidth=1.2,
                label=f"mean={np.mean(clean_logps):.3f}")
    ax1.axvline(float(np.median(clean_logps)), color='orange', linestyle=':', linewidth=1.2,
                label=f"median={np.median(clean_logps):.3f}")
    ax1.set_xlabel("Average NLL per sample")
    ax1.set_ylabel("Count")
    ax1.set_title(f"NLL Distribution — Clean Corpus{' (' + label + ')' if label else ''}")
    ax1.set_xlim(shared_lo, shared_hi)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = output_dir / "nll_histogram_clean.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  Plot 1 saved: {path1}")

    # ── Plot 2: original vs clean vs flagged ────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: original vs clean overlaid
    ax2 = axes[0]
    ax2.bar(orig_centers,  orig_counts,  width=bar_width,       color='gray',       alpha=0.5, label=f'Original (n={len(orig_logps):,})')
    ax2.bar(clean_centers, clean_counts, width=bar_width * 0.6, color='steelblue',  alpha=0.85, label=f'Clean (n={len(clean_logps):,})')
    ax2.axvline(float(np.mean(orig_logps)),  color='gray',      linestyle='--', linewidth=1.1, label=f"orig mean={np.mean(orig_logps):.3f}")
    ax2.axvline(float(np.mean(clean_logps)), color='steelblue', linestyle='--', linewidth=1.1, label=f"clean mean={np.mean(clean_logps):.3f}")
    ax2.set_xlabel("Average NLL per sample")
    ax2.set_ylabel("Count")
    ax2.set_title("Original vs Clean")
    ax2.set_xlim(shared_lo, shared_hi)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Right: flagged samples (approximate as difference)
    ax3 = axes[1]
    ax3.bar(orig_centers, flagged_counts, width=bar_width, color='crimson', alpha=0.75, label=f'Flagged (approx n={int(np.sum(flagged_counts)):,})')
    ax3.set_xlabel("Average NLL per sample")
    ax3.set_ylabel("Count")
    ax3.set_title("Flagged Samples by NLL Bin")
    ax3.set_xlim(shared_lo, shared_hi)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    suptitle = f"Contamination Audit{' — ' + label if label else ''}"
    fig2.suptitle(suptitle, fontsize=13, y=1.02)
    fig2.tight_layout()
    path2 = output_dir / "nll_histogram_comparison.png"
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Plot 2 saved: {path2}")

    # ── Save JSON with all bin data for replotting ───────────────────────────
    hist_data = {
        "description": "Histogram bin data for original and clean corpus NLL distributions. "
                       "Use bin_edges and counts to replot with any tool.",
        "shared_range": {"lo": shared_lo, "hi": shared_hi},
        "n_bins": n_bins,
        "original": {
            "bin_edges":   orig_edges.tolist(),
            "bin_centers": orig_centers,
            "counts":      orig_counts.tolist(),
            "mean":        float(np.mean(orig_logps)),
            "std":         float(np.std(orig_logps)),
            "median":      float(np.median(orig_logps)),
            "n_samples":   int(len(orig_logps)),
        },
        "clean": {
            "bin_edges":   clean_edges.tolist(),
            "bin_centers": clean_centers,
            "counts":      clean_counts.tolist(),
            "mean":        float(np.mean(clean_logps)),
            "std":         float(np.std(clean_logps)),
            "median":      float(np.median(clean_logps)),
            "n_samples":   int(len(clean_logps)),
        },
        "flagged_approx": {
            "bin_edges":   orig_edges.tolist(),
            "bin_centers": orig_centers,
            "counts":      flagged_counts.tolist(),
            "n_samples_approx": int(np.sum(flagged_counts)),
        },
    }
    json_path = output_dir / "nll_histogram.json"
    with open(json_path, 'w') as f:
        json.dump(hist_data, f, indent=2)
    print(f"  Bin data saved: {json_path}")


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(total: int, flagged_indices: list, per_bench_flags: dict, n: int, min_cutoff: int):
    flagged_set = set(flagged_indices)
    print("\n" + "=" * 62)
    print(f"CONTAMINATION AUDIT RESULTS  (n={n}, min_overlap_cutoff={min_cutoff})")
    print("=" * 62)
    print(f"{'Benchmark':<22} {'Flagged':>10} {'Flagged%':>10}")
    print("-" * 44)
    for name, flags in sorted(per_bench_flags.items()):
        count = len(flags)
        pct = 100.0 * count / total
        display = BENCHMARK_DISPLAY.get(name, name)
        print(f"{display:<22} {count:>10,} {pct:>9.3f}%")
    print("-" * 44)
    total_flagged = len(flagged_set)
    total_pct = 100.0 * total_flagged / total
    print(f"{'TOTAL (any bench)':<22} {total_flagged:>10,} {total_pct:>9.3f}%")
    print(f"{'CLEAN':<22} {total - total_flagged:>10,} {100 - total_pct:>9.3f}%")
    print("=" * 62)


# ============================================================
# SAVE CLEAN CORPUS
# ============================================================

def save_clean_corpus(
    corpus_dir: Path,
    output_dir: Path,
    texts: list,
    flagged_indices: list,
    per_bench_flags: dict,
    n: int,
    do_histogram: bool,
    n_bins: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_set = set(flagged_indices)
    clean_indices = [i for i in range(len(texts)) if i not in flagged_set]
    clean_texts = [texts[i] for i in clean_indices]

    print(f"\nSaving clean corpus to: {output_dir}")

    # texts.json
    with open(output_dir / "texts.json", 'w', encoding='utf-8') as f:
        json.dump(clean_texts, f, ensure_ascii=False)
    print(f"  texts.json: {len(clean_texts):,} samples")

    # train.json  (same format as generate_synthetic.py)
    train_data = [{"text": t} for t in clean_texts]
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    print(f"  train.json: {len(clean_texts):,} samples")

    # Copy and filter npy arrays
    clean_avg_logps = None
    for fname in ["sample_avg_logps.npy", "tokens_per_sample.npy"]:
        src = corpus_dir / fname
        if src.exists():
            arr = np.load(src)
            clean_arr = arr[clean_indices]
            np.save(output_dir / fname, clean_arr)
            print(f"  {fname}: {len(clean_arr):,} entries")
            if fname == "sample_avg_logps.npy":
                clean_avg_logps = clean_arr

    # Recompute sample_offsets
    tokens_path = output_dir / "tokens_per_sample.npy"
    if tokens_path.exists():
        tokens = np.load(tokens_path)
        offsets = np.zeros(len(tokens), dtype=np.int64)
        if len(tokens) > 1:
            offsets[1:] = np.cumsum(tokens[:-1])
        np.save(output_dir / "sample_offsets.npy", offsets)
        print(f"  sample_offsets.npy: recomputed")

    # Also copy token_logps if present (can be large — only copy if exists)
    token_logps_src = corpus_dir / "token_logps.npy"
    if token_logps_src.exists():
        # Rebuild from per-sample logps is expensive; just note it's not copied
        print(f"  token_logps.npy: NOT copied (rebuild from re-generation if needed)")

    # Update metadata
    meta_src = corpus_dir / "metadata.json"
    if meta_src.exists():
        with open(meta_src) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["num_samples_original"] = len(texts)
    meta["num_samples_flagged"] = len(flagged_set)
    meta["num_samples"] = len(clean_texts)
    meta["contamination_n"] = n
    meta["contamination_rate_pct"] = round(100.0 * len(flagged_set) / len(texts), 4)
    meta["benchmarks_checked"] = list(per_bench_flags.keys())
    meta["per_benchmark_flagged"] = {k: len(v) for k, v in per_bench_flags.items()}

    if clean_avg_logps is not None and len(clean_avg_logps) > 0:
        meta["mean_avg_nll"] = float(np.mean(clean_avg_logps))
        meta["std_avg_nll"] = float(np.std(clean_avg_logps))

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata.json: updated")

    # Flagged indices for inspection
    flagged_info = {
        "n": n,
        "total_samples": len(texts),
        "num_flagged": len(flagged_set),
        "flagged_indices": flagged_indices,
        "per_benchmark": {k: list(v) for k, v in per_bench_flags.items()},
        "flagged_examples": [
            {"index": i, "text_preview": texts[i][:300]}
            for i in flagged_indices[:20]
        ],
    }
    with open(output_dir / "flagged_indices.json", 'w') as f:
        json.dump(flagged_info, f, indent=2)
    print(f"  flagged_indices.json: {len(flagged_indices):,} flagged (with 20 examples)")

    # Histogram
    if do_histogram:
        print("\nGenerating NLL histograms...")
        # Load original logps for comparison plot
        orig_logps_path = corpus_dir / "sample_avg_logps.npy"
        if orig_logps_path.exists() and clean_avg_logps is not None and len(clean_avg_logps) > 0:
            orig_avg_logps = np.load(orig_logps_path)
            model_name = meta.get("model", corpus_dir.parent.name)
            temp = meta.get("temperature", corpus_dir.name)
            save_histogram(
                orig_logps=orig_avg_logps,
                clean_logps=clean_avg_logps,
                n_bins=n_bins,
                output_dir=output_dir,
                label=f"{model_name} T={temp}",
            )
        else:
            print("  WARNING: sample_avg_logps.npy not found, cannot generate NLL histogram")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir",      type=str, required=True)
    parser.add_argument("--benchmarks",      nargs="+", required=True)
    parser.add_argument("--n",               type=int, default=13)
    parser.add_argument("--min_cutoff",      type=int, default=1,
                        help="Min number of matching n-grams to flag a sample (default=1)")
    parser.add_argument("--save_histogram",  action="store_true")
    parser.add_argument("--num_bins",        type=int, default=50)
    parser.add_argument("--output_dir",      type=str, required=True)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)

    # Load corpus
    texts_path = corpus_dir / "texts.json"
    if not texts_path.exists():
        print(f"ERROR: texts.json not found in {corpus_dir}")
        return 1

    print(f"Loading corpus from {corpus_dir}...")
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    print(f"  Loaded {len(texts):,} samples")

    # Load metadata if available
    meta_path = corpus_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Model: {meta.get('model', 'unknown')}")
        print(f"  Temperature: {meta.get('temperature', 'unknown')}")
        print(f"  Total tokens: {meta.get('total_tokens', 'unknown'):,}" if isinstance(meta.get('total_tokens'), int) else "")

    # Build fingerprints
    global_set, per_bench = build_fingerprint_sets(args.benchmarks, args.n)

    if not per_bench:
        print("ERROR: No benchmarks loaded successfully.")
        return 1

    # Scan
    flagged_indices, per_bench_flags = scan_corpus(texts, global_set, per_bench, args.n)

    # Apply min_cutoff filter if > 1
    # (scan_corpus flags on >= 1 match; for higher cutoffs we re-filter)
    if args.min_cutoff > 1:
        print(f"\nApplying min_cutoff={args.min_cutoff} (re-scanning for exact overlap counts)...")
        strict_flagged = []
        strict_per_bench = defaultdict(set)
        for i, text in enumerate(tqdm(texts, desc="Re-scanning")):
            sample_ngrams = get_ngrams(normalize(text), args.n)
            for bench_name, bench_set in per_bench.items():
                overlap = len(sample_ngrams & bench_set)
                if overlap >= args.min_cutoff:
                    strict_flagged.append(i)
                    strict_per_bench[bench_name].add(i)
        # Deduplicate
        strict_flagged = list(dict.fromkeys(strict_flagged))
        flagged_indices = strict_flagged
        per_bench_flags = dict(strict_per_bench)

    # Report
    print_report(len(texts), flagged_indices, per_bench_flags, args.n, args.min_cutoff)

    # Save
    save_clean_corpus(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        texts=texts,
        flagged_indices=flagged_indices,
        per_bench_flags=per_bench_flags,
        n=args.n,
        do_histogram=args.save_histogram,
        n_bins=args.num_bins,
    )

    print("\nDone.")
    print(f"Clean corpus: {output_dir}/train.json")
    return 0


if __name__ == "__main__":
    exit(main())
