from __future__ import annotations

import argparse, csv, json, os, random, time
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
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="gpt2")
    p.add_argument("--output_root", default="results/lora_gpt2_sst2_profile")
    p.add_argument("--max_train_samples", type=int, default=2000)
    p.add_argument("--max_eval_samples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="c_attn,c_proj,c_fc")
    p.add_argument("--checkpoint_fractions", default="0.25,0.5,1.0")
    p.add_argument("--confidence_threshold", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_power_w", type=float, default=250.0)
    p.add_argument("--gpu_utilization", type=float, default=0.75)
    return p.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def adapter_size_bytes(path: Path) -> int:
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = path / name
        if p.exists():
            return p.stat().st_size
    return dir_size_bytes(path)


class SST2GenerativeDataset(Dataset):
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
            max_prompt = max(1, max_length - len(label_ids))
            prompt_ids = prompt_ids[-max_prompt:]
            input_ids = prompt_ids + label_ids
            labels = [-100] * len(prompt_ids) + label_ids
            self.rows.append({
                "prompt_ids": prompt_ids,
                "input_ids": input_ids,
                "labels": labels,
                "gold": int(row["label"]),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad = pad_token_id

    def train(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, masks = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
            masks.append([1] * len(x["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels),
        }

    def eval(self, batch):
        max_len = max(len(x["prompt_ids"]) for x in batch)
        input_ids, masks, gold = [], [], []
        for x in batch:
            pad_len = max_len - len(x["prompt_ids"])
            input_ids.append([self.pad] * pad_len + x["prompt_ids"])
            masks.append([0] * pad_len + [1] * len(x["prompt_ids"]))
            gold.append(x["gold"])
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(masks),
            "gold": torch.tensor(gold),
        }


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.inference_mode()
def evaluate(model, loader, device, candidate_ids, threshold):
    model.eval()
    total = correct = accepted = accepted_correct = 0
    conf_sum = 0.0
    latency_s = 0.0

    for batch in loader:
        gold = batch["gold"]
        model_batch = to_device(
            {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(**model_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_s += time.perf_counter() - start

        logits = out.logits[:, -1, :].detach().cpu()
        pair_logits = logits[:, candidate_ids]
        probs = torch.softmax(pair_logits, dim=-1)
        conf, pred = probs.max(dim=-1)

        match = pred.eq(gold)
        accept = conf.ge(threshold)

        total += gold.numel()
        correct += int(match.sum())
        accepted += int(accept.sum())
        accepted_correct += int((match & accept).sum())
        conf_sum += float(conf.sum())

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

    run_dir = Path(args.output_root) / f"gpt2_rank{args.rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    neg_id = tokenizer(" negative", add_special_tokens=False).input_ids[0]
    pos_id = tokenizer(" positive", add_special_tokens=False).input_ids[0]
    candidate_ids = [neg_id, pos_id]

    raw = load_dataset("glue", "sst2")
    train_raw = raw["train"].shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(raw["train"]))))
    eval_raw = raw["validation"].select(range(min(args.max_eval_samples, len(raw["validation"]))))

    train_ds = SST2GenerativeDataset(train_raw, tokenizer, args.max_length)
    eval_ds = SST2GenerativeDataset(eval_raw, tokenizer, args.max_length)
    collator = Collator(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator.train)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator.eval)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.to(device)

    rows = []
    base_metrics = evaluate(base_model, eval_loader, device, candidate_ids, args.confidence_threshold)
    rows.append({
        "phase": "baseline",
        "checkpoint_fraction": 0.0,
        "global_step": 0,
        "cumulative_train_time_s": 0.0,
        "estimated_train_energy_j": 0.0,
        "adapter_size_mb": 0.0,
        **base_metrics,
    })

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    fractions = [float(x) for x in args.checkpoint_fractions.split(",")]
    step_to_fraction = {max(1, min(total_steps, round(total_steps * f))): f for f in fractions}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_train = time.perf_counter()
    step = 0
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            step += 1
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            opt.step()

            if step in step_to_fraction:
                elapsed = time.perf_counter() - start_train
                ckpt = run_dir / f"checkpoint_step_{step}"
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                metrics = evaluate(model, eval_loader, device, candidate_ids, args.confidence_threshold)
                rows.append({
                    "phase": "lora_checkpoint",
                    "checkpoint_fraction": step_to_fraction[step],
                    "global_step": step,
                    "cumulative_train_time_s": elapsed,
                    "estimated_train_energy_j": elapsed * args.gpu_power_w * args.gpu_utilization,
                    "adapter_size_mb": adapter_size_bytes(ckpt) / (1024 * 1024),
                    "checkpoint_dir": str(ckpt),
                    **metrics,
                })
                write_outputs(run_dir, rows, {})

    metadata = {
        "model_name": args.model_name,
        "slm_type": "causal_language_model",
        "dataset": "glue/sst2",
        "task_format": "prompt_next_token_sentiment_classification",
        "label_tokens": {"negative": neg_id, "positive": pos_id},
        "rank": args.rank,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train_samples": len(train_ds),
        "max_eval_samples": len(eval_ds),
        "confidence_threshold": args.confidence_threshold,
        "trainable_params": trainable,
        "total_params": total_params,
        "trainable_ratio": trainable / total_params,
        "device": str(device),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0,
        "run_dir": str(run_dir),
    }
    write_outputs(run_dir, rows, metadata)
    print(json.dumps({"run_dir": str(run_dir), "metadata": metadata, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
