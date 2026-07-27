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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_name", default="nyu-mll/glue")
    parser.add_argument("--dataset_config", default="sst2")
    parser.add_argument("--output_root", default="results/lora_gpt2_sst2_profile")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=872)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="c_attn,c_proj,c_fc")
    parser.add_argument("--checkpoint_fractions", default="0.25,0.5,1.0")
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_power_w", type=float, default=250.0)
    parser.add_argument("--gpu_utilization", type=float, default=0.75)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def adapter_size_bytes(path: Path) -> int:
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = path / name
        if p.exists():
            return p.stat().st_size
    return dir_size_bytes(path)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class SST2CausalDataset(Dataset):
    def __init__(self, hf_ds, tokenizer, max_length: int):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_text = {0: " negative", 1: " positive"}

        for row in hf_ds:
            prompt = f"Review: {row['sentence']}\nSentiment:"
            label = self.label_text[int(row["label"])]

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            label_ids = tokenizer(label, add_special_tokens=False).input_ids
            if not label_ids:
                continue

            max_prompt_len = max(1, max_length - len(label_ids))
            prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + label_ids
            labels = [-100] * len(prompt_ids) + label_ids

            self.rows.append(
                {
                    "prompt_ids": prompt_ids,
                    "input_ids": input_ids,
                    "labels": labels,
                    "gold": int(row["label"]),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def train(self, batch):
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, masks = [], [], []
        for row in batch:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)
            masks.append([1] * len(row["input_ids"]) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def eval(self, batch):
        max_len = max(len(row["prompt_ids"]) for row in batch)
        input_ids, masks, golds = [], [], []
        for row in batch:
            pad_len = max_len - len(row["prompt_ids"])
            input_ids.append([self.pad_token_id] * pad_len + row["prompt_ids"])
            masks.append([0] * pad_len + [1] * len(row["prompt_ids"]))
            golds.append(row["gold"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "gold": torch.tensor(golds, dtype=torch.long),
        }


@torch.inference_mode()
def evaluate(model, loader, device, candidate_token_ids, threshold):
    model.eval()
    total = 0
    correct = 0
    accepted = 0
    accepted_correct = 0
    conf_sum = 0.0
    latency_s = 0.0

    for batch in loader:
        gold = batch["gold"]
        model_batch = to_device(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = model(**model_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_s += time.perf_counter() - start

        seq_lens = batch["attention_mask"].sum(dim=1) - 1
        logits_cpu = outputs.logits.detach().cpu()
        next_logits = logits_cpu[torch.arange(logits_cpu.shape[0]), seq_lens, :]
        pair_logits = next_logits[:, candidate_token_ids]
        probs = torch.softmax(pair_logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        match = pred.eq(gold)
        accept = conf.ge(threshold)

        total += gold.numel()
        correct += int(match.sum().item())
        accepted += int(accept.sum().item())
        accepted_correct += int((match & accept).sum().item())
        conf_sum += float(conf.sum().item())

    return {
        "accuracy": correct / total if total else 0.0,
        "accept_prob": accepted / total if total else 0.0,
        "accepted_accuracy": accepted_correct / accepted if accepted else "",
        "avg_confidence": conf_sum / total if total else 0.0,
        "latency_s_per_sample": latency_s / total if total else 0.0,
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
    run_dir = (
        Path(args.output_root)
        / f"gpt2_rank{args.rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    ensure_dir(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    neg_ids = tokenizer(" negative", add_special_tokens=False).input_ids
    pos_ids = tokenizer(" positive", add_special_tokens=False).input_ids
    if len(neg_ids) != 1 or len(pos_ids) != 1:
        raise ValueError(
            f"Expected single-token labels, got negative={neg_ids}, positive={pos_ids}"
        )
    candidate_token_ids = [neg_ids[0], pos_ids[0]]

    raw = load_dataset(args.dataset_name, args.dataset_config)
    train_raw = raw["train"].shuffle(seed=args.seed)
    eval_raw = raw["validation"]

    train_raw = train_raw.select(range(min(args.max_train_samples, len(train_raw))))
    eval_raw = eval_raw.select(range(min(args.max_eval_samples, len(eval_raw))))

    train_ds = SST2CausalDataset(train_raw, tokenizer, args.max_length)
    eval_ds = SST2CausalDataset(eval_raw, tokenizer, args.max_length)
    collator = Collator(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.train,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator.eval,
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)

    rows = []
    baseline = evaluate(
        base_model,
        eval_loader,
        device,
        candidate_token_ids,
        args.confidence_threshold,
    )
    rows.append(
        {
            "phase": "baseline",
            "checkpoint_fraction": 0.0,
            "global_step": 0,
            "cumulative_train_time_s": 0.0,
            "estimated_train_energy_j": 0.0,
            "adapter_size_mb": 0.0,
            **baseline,
        }
    )

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    total_steps = len(train_loader) * args.epochs
    fractions = [float(x.strip()) for x in args.checkpoint_fractions.split(",") if x.strip()]
    step_to_fraction = {
        max(1, min(total_steps, int(round(total_steps * f)))): f for f in fractions
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    train_start = time.perf_counter()
    for _epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = to_device(batch, device)

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
                    candidate_token_ids,
                    args.confidence_threshold,
                )
                rows.append(
                    {
                        "phase": "lora_checkpoint",
                        "checkpoint_fraction": step_to_fraction[global_step],
                        "global_step": global_step,
                        "cumulative_train_time_s": elapsed,
                        "estimated_train_energy_j": elapsed
                        * args.gpu_power_w
                        * args.gpu_utilization,
                        "adapter_size_mb": adapter_size_bytes(ckpt_dir)
                        / (1024 * 1024),
                        "checkpoint_dir": str(ckpt_dir),
                        **metrics,
                    }
                )
                write_outputs(run_dir, rows, {})

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    metadata = {
        "model_name": args.model_name,
        "slm_type": "causal_language_model",
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "task_format": "prompt_next_token_sentiment_classification",
        "label_tokens": {
            "negative": candidate_token_ids[0],
            "positive": candidate_token_ids[1],
        },
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train_samples": len(train_ds),
        "max_eval_samples": len(eval_ds),
        "confidence_threshold": args.confidence_threshold,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
        "device": str(device),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "run_dir": str(run_dir),
    }
    write_outputs(run_dir, rows, metadata)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "metadata": metadata,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
