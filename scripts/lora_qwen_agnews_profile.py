from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", default="fancyzhx/ag_news")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--output_root", default="results/lora_qwen_agnews_profile")
    parser.add_argument("--max_train_samples", type=int, default=200)
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--checkpoint_fractions", default="0.25,0.5,1.0")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu_power_w", type=float, default=450.0)
    parser.add_argument("--gpu_utilization", type=float, default=0.75)
    return parser.parse_args()


LABEL_TEXT = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Technology",
}


def prompt_text(text: str) -> str:
    return (
        "Classify the following news article into one category.\n"
        "Choices: World, Sports, Business, Technology.\n"
        f"Article: {text}\n"
        "Answer with exactly one word:"
    )

def tokenize_train(example, tokenizer, max_length: int, label_token_ids: dict[int, int]):
    prompt = prompt_text(example["text"])
    answer = " " + LABEL_TEXT[int(example["label"])]

    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length - 1, add_special_tokens=False)["input_ids"]
    answer_id = label_token_ids[int(example["label"])]

    input_ids = prompt_ids + [answer_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + [answer_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def tokenize_eval(example, tokenizer, max_length: int):
    encoded = tokenizer(prompt_text(example["text"]), truncation=True, max_length=max_length)
    encoded["label"] = int(example["label"])
    return encoded


def count_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, label_token_ids, threshold: float, device: torch.device):
    model.eval()
    correct = 0
    accepted = 0
    accepted_correct = 0
    n = 0
    total_conf = 0.0
    total_latency = 0.0

    label_order = sorted(label_token_ids)
    label_ids = torch.tensor([label_token_ids[i] for i in label_order], device=device)

    for batch in dataloader:
        labels = batch.pop("label").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_latency += time.perf_counter() - start

        logits = outputs.logits[:, -1, :]
        choice_logits = logits.index_select(dim=-1, index=label_ids)
        probs = torch.softmax(choice_logits, dim=-1)

        conf, pred = probs.max(dim=-1)
        accepted_mask = conf >= threshold

        correct += int((pred == labels).sum().item())
        accepted += int(accepted_mask.sum().item())
        accepted_correct += int(((pred == labels) & accepted_mask).sum().item())
        total_conf += float(conf.sum().item())
        n += labels.numel()

    return {
        "accuracy": correct / n if n else 0.0,
        "accept_prob": accepted / n if n else 0.0,
        "accepted_accuracy": accepted_correct / accepted if accepted else "",
        "avg_confidence": total_conf / n if n else 0.0,
        "latency_s_per_sample": total_latency / n if n else 0.0,
        "eval_samples": n,
    }

def save_rows(run_dir: Path, metadata: dict, rows: list[dict]):
    csv_path = run_dir / "metrics.csv"
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"qwen25_7b_rank{args.rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_token_ids = {i: tokenizer.encode(" " + name, add_special_tokens=False)[0] for i, name in LABEL_TEXT.items()}

    raw = load_dataset(args.dataset_name) if args.dataset_config in (None, "None", "") else load_dataset(args.dataset_name, args.dataset_config)
    train_raw = raw["train"].shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(raw["train"]))))
    eval_split = "validation" if "validation" in raw else "test"
    eval_source = raw[eval_split].shuffle(seed=args.seed)
    eval_raw = eval_source.select(range(min(args.max_eval_samples, len(eval_source))))

    train_ds = train_raw.map(
        lambda ex: tokenize_train(ex, tokenizer, args.max_length, label_token_ids),
        remove_columns=train_raw.column_names,
    )
    eval_ds = eval_raw.map(
        lambda ex: tokenize_eval(ex, tokenizer, args.max_length),
        remove_columns=[c for c in eval_raw.column_names if c not in {"label"}],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: tokenizer.pad(features, padding=True, return_tensors="pt"),
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = count_trainable_params(model)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    checkpoint_fractions = sorted(float(x) for x in args.checkpoint_fractions.split(","))
    total_steps = max(1, len(train_loader) * args.epochs)
    checkpoint_steps = {max(1, int(total_steps * f)): f for f in checkpoint_fractions}

    metadata = {
        "model_name": args.model_name,
        "slm_type": "qwen2.5_7b_4bit_qlora",
        "dataset": "ag_news",
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "confidence_threshold": args.confidence_threshold,
        "label_token_ids": label_token_ids,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total if total else 0.0,
        "device": str(device),
        "run_dir": str(run_dir),
    }

    rows = []
    rows.append({
        "phase": "baseline",
        "checkpoint_fraction": 0.0,
        "global_step": 0,
        "cumulative_train_time_s": 0.0,
        "estimated_train_energy_j": 0.0,
        "adapter_size_mb": 0.0,
        **evaluate(model, eval_loader, tokenizer, label_token_ids, args.confidence_threshold, device),
    })
    save_rows(run_dir, metadata, rows)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    global_step = 0
    train_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum_steps
            loss.backward()
            if (global_step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step in checkpoint_steps:
                fraction = checkpoint_steps[global_step]
                elapsed = time.perf_counter() - train_start
                ckpt_dir = run_dir / f"checkpoint_step_{global_step}"
                model.save_pretrained(ckpt_dir)
                adapter_size = dir_size_mb(ckpt_dir)
                metrics = evaluate(model, eval_loader, tokenizer, label_token_ids, args.confidence_threshold, device)
                rows.append({
                    "phase": "lora_checkpoint",
                    "checkpoint_fraction": fraction,
                    "global_step": global_step,
                    "cumulative_train_time_s": elapsed,
                    "estimated_train_energy_j": args.gpu_power_w * args.gpu_utilization * elapsed,
                    "adapter_size_mb": adapter_size,
                    "checkpoint_dir": str(ckpt_dir),
                    **metrics,
                })
                save_rows(run_dir, metadata, rows)
                model.train()

    if total_steps not in checkpoint_steps:
        elapsed = time.perf_counter() - train_start
        ckpt_dir = run_dir / f"checkpoint_step_{global_step}"
        model.save_pretrained(ckpt_dir)
        metrics = evaluate(model, eval_loader, tokenizer, label_token_ids, args.confidence_threshold, device)
        rows.append({
            "phase": "lora_checkpoint",
            "checkpoint_fraction": 1.0,
            "global_step": global_step,
            "cumulative_train_time_s": elapsed,
            "estimated_train_energy_j": args.gpu_power_w * args.gpu_utilization * elapsed,
            "adapter_size_mb": dir_size_mb(ckpt_dir),
            "checkpoint_dir": str(ckpt_dir),
            **metrics,
        })

    if torch.cuda.is_available():
        metadata["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    save_rows(run_dir, metadata, rows)
    print(json.dumps({"run_dir": str(run_dir), "metadata": metadata, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
