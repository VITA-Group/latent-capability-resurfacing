# Latent Capability Resurfacing — Code

Reference implementation of the prompt-free BOS-only self-training pipeline
described in the paper, restricted to Qwen and LLaMA families. 
## Repository layout

```
.
├── setup.sh                          # one-time environment setup
├── pipeline.sh                       # interactive pipeline: gen + audit + subset + train
├── evaluate.sh                       # interactive eval over base model + checkpoints
├── requirements.txt
├── configs/
│   └── ds_z3.json                    # deepspeed ZeRO-3 config (from llamafactory)
└── src/
    ├── generate_synthetic.py         # vLLM unconditional sampler + shard merge
    ├── audit.py                      # n-gram contamination scanner
    └── create_subsets.py             # random training subset builder
```

## Quick start

```bash
bash setup.sh                # asks for env name, creates conda env, installs deps
conda activate <env_name>
bash pipeline.sh             # generate → audit → subset → train
bash evaluate.sh             # evaluate base model + every checkpoint
```

Both interactive scripts ask every question up front, print a summary, ask
one confirmation, then run.

## What setup.sh does

1. Asks for a conda env name (defaults to `lcr`) and creates it with Python 3.11.
2. Installs **torch 2.9.0** from the **cu128** wheel index. Matches the
   versions verified to work on our reference machine (NVIDIA driver 550.x
   supporting CUDA 12.4, A100 GPUs).
3. Installs **deepspeed 0.16.9**.
4. Installs the remaining pinned requirements (`requirements.txt`).
5. Clones **LLaMA-Factory** to `third_party/LLaMA-Factory`.
6. Runs a smoke test: imports, `torch.cuda.is_available()`, GPU count, a
   small GPU matmul, and CLI availability for `llamafactory-cli` and `lm_eval`.

The cu128 wheel needs an NVIDIA driver supporting CUDA ≥ 12.4. If your
driver is older the script will warn and torch will likely fail to import.

## pipeline.sh

Four stages, each independently toggleable. For any stage whose output
already exists, the script asks whether to reuse, re-run (overwriting), or
abort.

1. **Generate.** vLLM sampling from `<BOS>` at temperature τ, fanned out
   across all visible GPUs with per-GPU seed `seed + gpu_id * 1000`.
   Per-GPU shards are stitched into a single corpus.
2. **Audit.** N-gram overlap scan against the test/dev/train splits of the
   benchmarks listed in the paper. Writes a clean corpus to a parallel
   directory tree.
3. **Subset.** Builds N independent training subsets at a fixed token budget.
   Each subset is built by drawing samples uniformly at random from the
   corpus (one sample per entry) until the token budget is reached.
4. **Train.** Continued pretraining with `llamafactory-cli` and ZeRO-3.
   The training subset is registered dynamically in `data/dataset_info.json`.
   The pipeline targets an effective batch size of 64 by default (paper
   Appendix C), auto-computing gradient accumulation steps based on
   per-device batch size and GPU count.

## evaluate.sh

Standalone evaluation script. Scans `output/` for training runs and their
checkpoints, lets you pick which to evaluate, runs lm-evaluation-harness
through the vLLM backend, writes results to `evals/`.

Benchmark options include the seven paper-core tasks (ARC-Challenge, MMLU,
TruthfulQA, HellaSwag, GSM8K, MATH, HumanEval) plus ARC-Easy and
WinoGrande, with two named suites:

- **Suite A** — zero-shot on all paper-core benchmarks (default).
- **Suite B** — standard few-shot: ARC-Challenge 25-shot, GSM8K 8-shot,
  HellaSwag 5-shot, MATH 4-shot.

Both suites can be selected together (`AB`), in which case `arc_challenge`
runs at both 0-shot AND 25-shot, etc.

## Output layout

```
synthetic_data/{model_short}/temp{T}/                        # stage 1
synthetic_data_clean/{model_short}/temp{T}_clean_n{N}/       # stage 2
└── subsets/                                                  # stage 3
output/{run_name}/checkpoint-*/                              # stage 4
evals/{model_short}/{base|run_name/checkpoint-*}/            # evaluate.sh
    {benchmark}_fs{N}/results.json
```
