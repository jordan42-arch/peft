from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf_cache/datasets")

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", default="nyu-mll/glue")
    parser.add_argument("--dataset_config", default="sst2")
    parser.add_argument("--text_field", default="sentence")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--output_root", default="results/lora_sst2_profile")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", default="q_lin,v_lin")
    parser.add_argument("--modules_to_save", default="pre_classifier,classifier")
    parser.add_argument("--checkpoint_fractions", default="0.25,0.5,1.0")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gpu_power_w", type=float, default=250.0)
    parser.add_argument("--gpu_utilization", type=float, default=0.75)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def adapter_size_bytes(path: Path) -> int:
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = path / name
        if p.exists():
            return p.stat().st_size
    return dir_size_bytes(path)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total else 0.0
    return trainable, total, ratio


@torch.inference_mode()
def evaluate(model, loader, device, confidence_threshold):
    model.eval()
    total = 0
    correct = 0
    accepted = 0
    accepted_correct = 0
    conf_sum = 0.0
    latency_s = 0.0
    latency_samples = 0

    for batch in loader:
        labels_cpu = batch["labels"].detach().cpu()
        batch = move_to_device(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_s += time.perf_counter() - start
        latency_samples += labels_cpu.numel()

        probs = torch.softmax(outputs.logits.detach().cpu(), dim=-1)
        conf, preds = probs.max(dim=-1)
        match = preds.eq(labels_cpu)
        accept_mask = conf.ge(confidence_threshold)

        total += labels_cpu.numel()
        correct += int(match.sum().item())
        accepted += int(accept_mask.sum().item())
        accepted_correct += int((match & accept_mask).sum().item())
        conf_sum += float(conf.sum().item())

    return {
        "accuracy": correct / total if total else 0.0,
        "accept_prob": accepted / total if total else 0.0,
        "accepted_accuracy": accepted_correct / accepted if accepted else "",
        "avg_confidence": conf_sum / total if total else 0.0,
        "latency_s_per_sample": latency_s / latency_samples if latency_samples else 0.0,
        "eval_samples": total,
    }


def write_outputs(run_dir: Path, rows, metadata):
    ensure_dir(run_dir)
    (run_dir / "metrics.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"rank{args.rank}_{run_id}"
    ensure_dir(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset(args.dataset_name, args.dataset_config)

    train_ds = raw["train"].shuffle(seed=args.seed)
    eval_ds = raw["validation"]
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    def tokenize(batch):
        return tokenizer(
            batch[args.text_field],
            truncation=True,
            max_length=args.max_length,
        )

    remove_train_cols = [c for c in train_ds.column_names if c != args.label_field]
    remove_eval_cols = [c for c in eval_ds.column_names if c != args.label_field]
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=remove_train_cols)
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=remove_eval_cols)

    if args.label_field != "labels":
        train_ds = train_ds.rename_column(args.label_field, "labels")
        eval_ds = eval_ds.rename_column(args.label_field, "labels")

    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    if getattr(base_model.config, "pad_token_id", None) is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)

    rows = []
    baseline_metrics = evaluate(
        base_model,
        eval_loader,
        device,
        args.confidence_threshold,
    )
    rows.append({
        "phase": "baseline",
        "checkpoint_fraction": 0.0,
        "global_step": 0,
        "cumulative_train_time_s": 0.0,
        "estimated_train_energy_j": 0.0,
        "adapter_size_mb": 0.0,
        **baseline_metrics,
    })

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    modules_to_save = [x.strip() for x in args.modules_to_save.split(",") if x.strip()]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(base_model, lora_config)
    trainable_params, total_params, trainable_ratio = parameter_counts(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    fractions = [float(x.strip()) for x in args.checkpoint_fractions.split(",") if x.strip()]
    step_to_fraction = {
        max(1, min(total_steps, int(round(total_steps * f)))): f
        for f in fractions
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = move_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

            if global_step in step_to_fraction:
                elapsed = time.perf_counter() - train_start
                ckpt_dir = run_dir / f"checkpoint_step_{global_step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

                metrics = evaluate(
                    model,
                    eval_loader,
                    device,
                    args.confidence_threshold,
                )
                rows.append({
                    "phase": "lora_checkpoint",
                    "checkpoint_fraction": step_to_fraction[global_step],
                    "global_step": global_step,
                    "cumulative_train_time_s": elapsed,
                    "estimated_train_energy_j": elapsed * args.gpu_power_w * args.gpu_utilization,
                    "adapter_size_mb": adapter_size_bytes(ckpt_dir) / (1024 * 1024),
                    "checkpoint_dir": str(ckpt_dir),
                    **metrics,
                })
                write_outputs(run_dir, rows, {})

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    metadata = {
        "model_name": args.model_name,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train_samples": len(train_ds),
        "max_eval_samples": len(eval_ds),
        "confidence_threshold": args.confidence_threshold,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
        "device": str(device),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "run_dir": str(run_dir),
    }
    write_outputs(run_dir, rows, metadata)
    print(json.dumps({"run_dir": str(run_dir), "metadata": metadata, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
