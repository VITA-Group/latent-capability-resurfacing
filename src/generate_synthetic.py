#!/usr/bin/env python3
"""
Generate synthetic samples from any model using vLLM with configurable temperature.

Features:
- Configurable model and temperature
- Multi-GPU parallel generation
- Automatic merge and cleanup of per-GPU files
- Minimum token length enforcement
- Organized output: synthetic_data/{model_name}/temp{temperature}/

Usage:
    # Run all 8 GPUs in parallel:
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python generate_synthetic.py \
            --model Qwen/Qwen2.5-0.5B \
            --temperature 1.0 \
            --gpu_id $i \
            --num_gpus 8 &
    done
    wait
    
    # Then merge results and cleanup:
    python generate_synthetic.py \
        --model Qwen/Qwen2.5-0.5B \
        --temperature 1.0 \
        --merge_only

Examples:
    # Neutral sampling (temperature = 1.0)
    python generate_synthetic.py --model Qwen/Qwen2.5-0.5B --temperature 1.0 ...
    
    # Mode-seeking (temperature = 0.7)
    python generate_synthetic.py --model Qwen/Qwen2.5-0.5B --temperature 0.7 ...
    
    # Strongly mode-seeking (temperature = 0.3)
    python generate_synthetic.py --model Qwen/Qwen2.5-0.5B --temperature 0.3 ...
    
    # Different model
    python generate_synthetic.py --model Qwen/Qwen2.5-1.5B --temperature 1.0 ...
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_model_short_name(model_name: str) -> str:
    """Convert model path to short name for folder structure."""
    # e.g., "Qwen/Qwen2.5-0.5B" -> "qwen2.5-0.5b"
    short_name = model_name.split("/")[-1].lower()
    return short_name


def get_output_dir(base_dir: str, model_name: str, temperature: float) -> Path:
    """Get output directory path: synthetic_data/{model_name}/temp{temperature}/"""
    model_short = get_model_short_name(model_name)
    # Use temperature value directly, replace decimal point with underscore
    # 1.0 -> temp1_0, 0.75 -> temp0_75, 0.3 -> temp0_3
    temp_str = f"temp{temperature}".replace(".", "_")
    return Path(base_dir) / model_short / temp_str


def get_sampling_mode(temperature: float) -> str:
    """Get human-readable sampling mode description."""
    if temperature > 1.0:
        return f"diversity-seeking (temperature = {temperature})"
    elif temperature == 1.0:
        return f"neutral (temperature = {temperature})"
    elif temperature >= 0.5:
        return f"mode-seeking (temperature = {temperature})"
    else:
        return f"strongly mode-seeking (temperature = {temperature})"


# Predefined prompt options
PROMPT_OPTIONS = {
    "bos": None,  # Will use tokenizer's BOS token
    "random": "Generate a passage of text on a random topic. The topic, style, and format are entirely up to you.\n\n",
    "diverse": "Write a random piece of text. It can be about any subject, in any style (narrative, technical, conversational, formal, etc.), and any format (prose, list, dialogue, code, etc.).\n\n",
    "creative": "Write about anything:\n\n",
    "document": "The following is a random text:\n\n",
    "continue": "The ",
}


def get_prompt(prompt_arg: str, tokenizer) -> tuple[str, str]:
    """
    Resolve prompt argument to actual prompt string.
    
    Returns:
        tuple: (prompt_string, prompt_name) where prompt_name is for display/logging
    """
    prompt_arg_lower = prompt_arg.lower().strip()
    
    if prompt_arg_lower in PROMPT_OPTIONS:
        if prompt_arg_lower == "bos":
            # Use BOS token or fallback to newline
            if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
                return tokenizer.bos_token, "bos"
            else:
                return "\n", "bos (fallback to newline)"
        else:
            return PROMPT_OPTIONS[prompt_arg_lower], prompt_arg_lower
    else:
        # Custom prompt provided directly
        return prompt_arg, "custom"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data with configurable model and temperature")
    
    # Model and sampling parameters
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to use (e.g., Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (1.0=neutral, <1=mode-seeking, >1=diversity-seeking)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p sampling (1.0 = no modification)")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k sampling (-1 = disabled)")
    
    # Generation parameters
    parser.add_argument("--total_samples", type=int, default=100000,
                        help="Total number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum tokens per sample")
    parser.add_argument("--min_tokens", type=int, default=50,
                        help="Minimum tokens per sample (forces generation to continue until this length)")
    parser.add_argument("--prompt", type=str, default="bos",
                        help="Prompt to use for generation (bos, random, diverse, creative, or custom text)")
    
    # Multi-GPU parameters
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for data parallelism")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Total GPUs for data parallelism")
    
    # Output parameters
    parser.add_argument("--base_dir", type=str, default="synthetic_data",
                        help="Base directory for output")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing GPU results (and cleanup)")
    parser.add_argument("--keep_gpu_files", action="store_true",
                        help="Keep per-GPU files after merging (default: delete them)")
    
    args = parser.parse_args()
    
    # Get output directory
    output_dir = get_output_dir(args.base_dir, args.model, args.temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.merge_only:
        merge_results(output_dir, args.num_gpus, args.model, args.temperature, 
                     keep_gpu_files=args.keep_gpu_files)
        return
    
    # Import vLLM here to avoid loading it during merge_only
    from vllm import LLM, SamplingParams
    
    # Calculate samples for this GPU
    samples_per_gpu = args.total_samples // args.num_gpus
    
    # Last GPU handles remainder
    if args.gpu_id == args.num_gpus - 1:
        samples_per_gpu += args.total_samples % args.num_gpus
    
    # Set random seed based on GPU ID to ensure different outputs
    random_seed = 42 + args.gpu_id * 1000
    
    sampling_mode = get_sampling_mode(args.temperature)
    
    print("=" * 70)
    print(f"SYNTHETIC DATA GENERATION")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Temperature:     {args.temperature} ({sampling_mode})")
    print(f"Top-p:           {args.top_p}")
    print(f"Top-k:           {args.top_k}")
    print(f"Min tokens:      {args.min_tokens}")
    print(f"Max tokens:      {args.max_new_tokens}")
    print(f"GPU:             {args.gpu_id}/{args.num_gpus}")
    print(f"Samples (GPU):   {samples_per_gpu:,}")
    print(f"Random seed:     {random_seed}")
    print(f"Output dir:      {output_dir}")
    print("=" * 70)
    
    # Initialize vLLM with GPU-specific seed
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_new_tokens + 10,  # Small buffer
        gpu_memory_utilization=0.90,
        seed=random_seed,
    )
    
    # Get tokenizer for BOS token
    tokenizer = llm.get_tokenizer()
    
    # Get prompt based on argument
    prompt, prompt_name = get_prompt(args.prompt, tokenizer)
    prompt_len = len(prompt)  # For stripping from output
    
    print(f"Prompt type:     {prompt_name}")
    print(f"Prompt text:     {repr(prompt[:100])}{'...' if len(prompt) > 100 else ''}")
    
    # Sampling params with min_tokens to enforce minimum length
    # NOTE: We explicitly set stop_token_ids=[] because:
    # 1. Many models (Qwen, etc.) have additional stop tokens in their generation_config
    # 2. vLLM's min_tokens only blocks EOS, not these extra stop tokens
    # 3. This is a known vLLM bug: https://github.com/vllm-project/vllm/issues/21987
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens=args.min_tokens,  # Force at least this many tokens before EOS
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],  # Clear any model-default stop tokens that bypass min_tokens
        n=1,
        logprobs=1,  # Get log probs (unscaled by temperature)
        skip_special_tokens=False,
    )
    
    # Create prompts
    prompts = [prompt] * samples_per_gpu
    
    print(f"\nGenerating {samples_per_gpu:,} samples...")
    print(f"  (minimum {args.min_tokens} tokens, maximum {args.max_new_tokens} tokens per sample)")
    
    # Generate
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    # Process outputs
    all_texts = []
    all_token_logps = []
    all_sample_avg_logps = []
    all_tokens_per_sample = []
    
    # Track filtered samples
    filtered_count = 0
    
    print(f"\nProcessing outputs...")
    for output in tqdm(outputs, desc=f"GPU {args.gpu_id}"):
        completion = output.outputs[0]
        
        # Get text - vLLM's completion.text already excludes the input prompt
        # so we just need to strip whitespace
        text = completion.text.strip()
        
        # Skip empty texts
        if not text:
            filtered_count += 1
            continue
        
        # Get token log probs (UNSCALED by temperature)
        token_logps = []
        if completion.logprobs:
            for logprob_dict in completion.logprobs:
                if logprob_dict:
                    for token_id, logprob_obj in logprob_dict.items():
                        token_logps.append(logprob_obj.logprob)
                        break
        
        token_logps = np.array(token_logps, dtype=np.float32)
        num_tokens = len(token_logps)
        
        # Filter out samples that are too short (backup check)
        if num_tokens < args.min_tokens:
            filtered_count += 1
            continue
        
        # Average NLL (true model NLL, not affected by temperature)
        avg_logp = -np.mean(token_logps) if num_tokens > 0 else 0.0
        
        all_texts.append(text)
        all_token_logps.append(token_logps)
        all_sample_avg_logps.append(avg_logp)
        all_tokens_per_sample.append(num_tokens)
    
    if filtered_count > 0:
        print(f"\n⚠️ Filtered {filtered_count} samples below {args.min_tokens} tokens")
        print(f"  Kept {len(all_texts):,} samples")
    
    print(f"\nGPU {args.gpu_id}: Saving...")
    
    # Save results for this GPU
    gpu_prefix = f"gpu_{args.gpu_id}"
    
    # Save texts
    texts_path = output_dir / f"{gpu_prefix}_texts.json"
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)
    print(f"  ✓ {texts_path.name}")
    
    # Save token logps
    flat_token_logps = np.concatenate(all_token_logps) if all_token_logps else np.array([], dtype=np.float32)
    np.save(output_dir / f"{gpu_prefix}_token_logps.npy", flat_token_logps)
    print(f"  ✓ {gpu_prefix}_token_logps.npy")
    
    # Save sample avg logps
    sample_avg_logps = np.array(all_sample_avg_logps, dtype=np.float32)
    np.save(output_dir / f"{gpu_prefix}_sample_avg_logps.npy", sample_avg_logps)
    print(f"  ✓ {gpu_prefix}_sample_avg_logps.npy")
    
    # Save tokens per sample
    tokens_per_sample = np.array(all_tokens_per_sample, dtype=np.int32)
    np.save(output_dir / f"{gpu_prefix}_tokens_per_sample.npy", tokens_per_sample)
    print(f"  ✓ {gpu_prefix}_tokens_per_sample.npy")
    
    # Save offsets
    offsets = np.zeros(len(all_tokens_per_sample), dtype=np.int64)
    offsets[1:] = np.cumsum(tokens_per_sample[:-1])
    np.save(output_dir / f"{gpu_prefix}_sample_offsets.npy", offsets)
    print(f"  ✓ {gpu_prefix}_sample_offsets.npy")
    
    # Save GPU metadata
    metadata = {
        "model": args.model,
        "gpu_id": args.gpu_id,
        "num_samples": len(all_texts),
        "min_tokens": args.min_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "prompt_type": prompt_name,
        "prompt_text": prompt if len(prompt) < 500 else prompt[:500] + "...",
        "total_tokens": int(np.sum(tokens_per_sample)),
        "mean_tokens_per_sample": float(np.mean(tokens_per_sample)),
        "std_tokens_per_sample": float(np.std(tokens_per_sample)),
        "min_tokens_actual": int(np.min(tokens_per_sample)),
        "max_tokens_actual": int(np.max(tokens_per_sample)),
        "mean_avg_nll": float(np.mean(sample_avg_logps)),
        "sampling_mode": sampling_mode,
    }
    
    with open(output_dir / f"{gpu_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ {gpu_prefix}_metadata.json")
    
    print(f"\n✓ GPU {args.gpu_id} complete!")
    print(f"  Samples: {len(all_texts):,}")
    print(f"  Tokens:  {np.sum(tokens_per_sample):,}")
    print(f"  Tokens/sample: {np.mean(tokens_per_sample):.1f} ± {np.std(tokens_per_sample):.1f}")
    print(f"  Token range: [{np.min(tokens_per_sample)}, {np.max(tokens_per_sample)}]")
    print(f"  Mean NLL: {np.mean(sample_avg_logps):.4f}")


def merge_results(output_dir: Path, num_gpus: int, model_name: str, temperature: float, 
                  keep_gpu_files: bool = False):
    """Merge results from all GPUs into single files and optionally cleanup."""
    
    sampling_mode = get_sampling_mode(temperature)
    model_short = get_model_short_name(model_name)
    
    print("=" * 70)
    print(f"MERGING RESULTS")
    print("=" * 70)
    print(f"Model:       {model_name}")
    print(f"Temperature: {temperature} ({sampling_mode})")
    print(f"Output dir:  {output_dir}")
    print(f"Cleanup:     {'No (keeping GPU files)' if keep_gpu_files else 'Yes (removing GPU files)'}")
    print("=" * 70)
    
    all_texts = []
    all_token_logps = []
    all_sample_avg_logps = []
    all_tokens_per_sample = []
    gpu_files_to_delete = []
    
    for gpu_id in range(num_gpus):
        gpu_prefix = f"gpu_{gpu_id}"
        
        texts_path = output_dir / f"{gpu_prefix}_texts.json"
        if not texts_path.exists():
            print(f"  ⚠️ Missing {texts_path.name}, skipping GPU {gpu_id}")
            continue
        
        print(f"  Loading GPU {gpu_id}...")
        
        # Load texts
        with open(texts_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        all_texts.extend(texts)
        
        # Load arrays
        token_logps_path = output_dir / f"{gpu_prefix}_token_logps.npy"
        avg_logps_path = output_dir / f"{gpu_prefix}_sample_avg_logps.npy"
        tokens_path = output_dir / f"{gpu_prefix}_tokens_per_sample.npy"
        offsets_path = output_dir / f"{gpu_prefix}_sample_offsets.npy"
        metadata_path = output_dir / f"{gpu_prefix}_metadata.json"
        
        all_token_logps.append(np.load(token_logps_path))
        all_sample_avg_logps.append(np.load(avg_logps_path))
        all_tokens_per_sample.append(np.load(tokens_path))
        
        print(f"    {len(texts):,} samples")
        
        # Track files to delete
        gpu_files_to_delete.extend([
            texts_path, token_logps_path, avg_logps_path, 
            tokens_path, offsets_path, metadata_path
        ])
    
    if not all_texts:
        print("\n✗ No data found to merge!")
        return
    
    print(f"\nMerging {len(all_texts):,} total samples...")
    
    # Concatenate arrays
    merged_token_logps = np.concatenate(all_token_logps)
    merged_sample_avg_logps = np.concatenate(all_sample_avg_logps)
    merged_tokens_per_sample = np.concatenate(all_tokens_per_sample)
    
    # Compute offsets
    merged_offsets = np.zeros(len(merged_tokens_per_sample), dtype=np.int64)
    merged_offsets[1:] = np.cumsum(merged_tokens_per_sample[:-1])
    
    # Save merged files
    print("\nSaving merged files...")
    
    # Texts
    texts_path = output_dir / "texts.json"
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)
    print(f"  ✓ texts.json ({len(all_texts):,} samples)")
    
    # Token logps
    np.save(output_dir / "token_logps.npy", merged_token_logps)
    print(f"  ✓ token_logps.npy ({len(merged_token_logps):,} tokens)")
    
    # Sample avg logps
    np.save(output_dir / "sample_avg_logps.npy", merged_sample_avg_logps)
    print(f"  ✓ sample_avg_logps.npy")
    
    # Tokens per sample
    np.save(output_dir / "tokens_per_sample.npy", merged_tokens_per_sample)
    print(f"  ✓ tokens_per_sample.npy")
    
    # Offsets
    np.save(output_dir / "sample_offsets.npy", merged_offsets)
    print(f"  ✓ sample_offsets.npy")
    
    # Training format
    training_data = [{"text": t} for t in all_texts]
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False)
    print(f"  ✓ train.json")
    
    # Metadata
    metadata = {
        "model": model_name,
        "model_short": model_short,
        "temperature": temperature,
        "sampling_mode": sampling_mode,
        "num_samples": len(all_texts),
        "total_tokens": int(np.sum(merged_tokens_per_sample)),
        "mean_tokens_per_sample": float(np.mean(merged_tokens_per_sample)),
        "std_tokens_per_sample": float(np.std(merged_tokens_per_sample)),
        "min_tokens_actual": int(np.min(merged_tokens_per_sample)),
        "max_tokens_actual": int(np.max(merged_tokens_per_sample)),
        "mean_avg_nll": float(np.mean(merged_sample_avg_logps)),
        "std_avg_nll": float(np.std(merged_sample_avg_logps)),
        "min_avg_nll": float(np.min(merged_sample_avg_logps)),
        "max_avg_nll": float(np.max(merged_sample_avg_logps)),
        "median_avg_nll": float(np.median(merged_sample_avg_logps)),
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")
    
    # Cleanup GPU files
    if not keep_gpu_files:
        print("\nCleaning up per-GPU files...")
        for fpath in gpu_files_to_delete:
            if fpath.exists():
                fpath.unlink()
                print(f"  🗑️ Deleted {fpath.name}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model:              {model_name}")
    print(f"Temperature:        {temperature} ({sampling_mode})")
    print(f"Total samples:      {len(all_texts):,}")
    print(f"Total tokens:       {np.sum(merged_tokens_per_sample):,}")
    print(f"Tokens/sample:      {np.mean(merged_tokens_per_sample):.1f} ± {np.std(merged_tokens_per_sample):.1f}")
    print(f"Token range:        [{np.min(merged_tokens_per_sample)}, {np.max(merged_tokens_per_sample)}]")
    print(f"Mean NLL:           {np.mean(merged_sample_avg_logps):.4f}")
    print(f"Std NLL:            {np.std(merged_sample_avg_logps):.4f}")
    print(f"NLL range:          [{np.min(merged_sample_avg_logps):.4f}, {np.max(merged_sample_avg_logps):.4f}]")
    print(f"\nOutput directory:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()