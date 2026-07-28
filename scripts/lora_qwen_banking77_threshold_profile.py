
import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


def parse_list(text, kind=float):
    return [kind(x.strip()) for x in text.split(",") if x.strip()]


def make_label_lines(label_names, label_codes):
    lines = []
    for i, name in enumerate(label_names):
        lines.append(f"{label_codes[i]}: {name.replace('_', ' ')}")
    return "\\n".join(lines)


def prompt_text(text, label_lines):
    return (
        "Classify the user banking request into exactly one intent.\\n"
        "Choose only one label code from the list.\\n"
        f"Request: {text}\\n"
        "Label list:\\n"
        f"{label_lines}\\n"
        "Answer code: "
    )


def build_single_token_codes(tokenizer, n):
    selected = []
    used = set()

    candidates = []
    candidates += [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    candidates += [chr(i) for i in range(ord("a"), ord("z") + 1)]
    candidates += [f"{a}{b}" for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    candidates += [f"{a}{b}" for a in "abcdefghijklmnopqrstuvwxyz" for b in "abcdefghijklmnopqrstuvwxyz"]

    for code in candidates:
        ids = tokenizer.encode(code, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in used:
            selected.append((code, ids[0]))
            used.add(ids[0])
        if len(selected) >= n:
            break

    if len(selected) < n:
        raise ValueError(f"Only found {len(selected)} single-token label codes, need {n}")

    label_codes = {i: selected[i][0] for i in range(n)}
    label_token_ids = {i: selected[i][1] for i in range(n)}
    return label_codes, label_token_ids


def tokenize_train(example, tokenizer, max_length, label_token_ids, label_codes, label_lines):
    y = int(example["label"])
    prompt_ids = tokenizer(
        prompt_text(example["text"], label_lines),
        truncation=True,
        max_length=max_length - 1,
        add_special_tokens=False,
    )["input_ids"]
    answer_id = label_token_ids[y]
    input_ids = prompt_ids + [answer_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + [answer_id],
    }


def tokenize_eval(example, tokenizer, max_length, label_lines):
    enc = tokenizer(
        prompt_text(example["text"], label_lines),
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    enc["label"] = int(example["label"])
    return enc


def make_train_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def make_eval_collate(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["label"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 / 1024


@torch.no_grad()
def evaluate(model, dataloader, label_token_ids, thresholds, device):
    model.eval()
    label_order = sorted(label_token_ids)
    label_ids = torch.tensor([label_token_ids[i] for i in label_order], device=device)
    stats = {
        th: {"correct": 0, "accepted": 0, "accepted_correct": 0, "n": 0, "conf_sum": 0.0, "lat": 0.0}
        for th in thresholds
    }

    for batch in dataloader:
        labels = batch.pop("label").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        last_pos = batch["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(out.logits.size(0), device=device), last_pos, :]
        probs = torch.softmax(logits.index_select(-1, label_ids), dim=-1)
        conf, pred = probs.max(dim=-1)

        for th in thresholds:
            mask = conf >= th
            st = stats[th]
            st["correct"] += int((pred == labels).sum().item())
            st["accepted"] += int(mask.sum().item())
            st["accepted_correct"] += int(((pred == labels) & mask).sum().item())
            st["n"] += int(labels.numel())
            st["conf_sum"] += float(conf.sum().item())
            st["lat"] += dt

    rows = []
    for th, st in stats.items():
        n = st["n"]
        ap = st["accepted"] / n if n else 0.0
        rows.append({
            "confidence_threshold": th,
            "accuracy": st["correct"] / n if n else 0.0,
            "accept_prob": ap,
            "offload_prob": 1 - ap,
            "accepted_accuracy": st["accepted_correct"] / st["accepted"] if st["accepted"] else "",
            "avg_confidence": st["conf_sum"] / n if n else 0.0,
            "latency_s_per_sample": st["lat"] / n if n else 0.0,
            "eval_samples": n,
        })
    return rows


def write_outputs(run_dir, metadata, rows):
    preferred = [
        "phase", "checkpoint_fraction", "global_step", "confidence_threshold",
        "accuracy", "accept_prob", "offload_prob", "accepted_accuracy",
        "avg_confidence", "latency_s_per_sample", "eval_samples",
        "cumulative_train_time_s", "estimated_train_energy_j", "adapter_size_mb",
        "checkpoint_dir",
    ]
    all_fields = []
    for r in rows:
        for k in r:
            if k not in all_fields:
                all_fields.append(k)
    fields = [k for k in preferred if k in all_fields] + [k for k in all_fields if k not in preferred]

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    (run_dir / "metrics.json").write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", default="mteb/banking77")
    parser.add_argument("--output_root", default="results/lora_qwen_banking77_threshold_profile")
    parser.add_argument("--max_train_samples", type=int, default=2000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.8,0.9")
    parser.add_argument("--checkpoint_fractions", default="1.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_power_w", type=float, default=575.0)
    parser.add_argument("--gpu_utilization", type=float, default=0.587)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    thresholds = parse_list(args.thresholds, float)
    fractions = parse_list(args.checkpoint_fractions, float)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(args.dataset_name)
    label_feature = raw["train"].features["label"]
    if hasattr(label_feature, "names") and label_feature.names:
        label_names = list(label_feature.names)
    elif "label_text" in raw["train"].column_names:
        pairs = {}
        for ex in raw["train"]:
            pairs[int(ex["label"])] = str(ex["label_text"])
        label_names = [pairs[i] for i in range(max(pairs) + 1)]
    else:
        label_names = [str(i) for i in sorted(set(raw["train"]["label"]))]
    label_codes, label_token_ids = build_single_token_codes(tokenizer, len(label_names))
    label_lines = make_label_lines(label_names, label_codes)

    train_src = raw["train"].shuffle(seed=args.seed)
    eval_split = "validation" if "validation" in raw else "test"
    eval_src = raw[eval_split].shuffle(seed=args.seed)
    train_raw = train_src.select(range(min(args.max_train_samples, len(train_src))))
    eval_raw = eval_src.select(range(min(args.max_eval_samples, len(eval_src))))

    train_ds = train_raw.map(
        lambda x: tokenize_train(x, tokenizer, args.max_length, label_token_ids, label_codes, label_lines),
        remove_columns=train_raw.column_names,
    )
    eval_ds = eval_raw.map(
        lambda x: tokenize_eval(x, tokenizer, args.max_length, label_lines),
        remove_columns=[c for c in eval_raw.column_names if c != "label"],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_train_collate(tokenizer))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=make_eval_collate(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=qconf,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"qwen25_7b_banking77_M{args.max_train_samples}_rank{args.rank}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    metadata = vars(args) | {
        "dataset": "banking77",
        "eval_split": eval_split,
        "label_names": label_names,
        "label_codes": label_codes,
        "label_token_ids": label_token_ids,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total,
        "device": str(device),
        "run_dir": str(run_dir),
    }

    rows = []
    for m in evaluate(model, eval_loader, label_token_ids, thresholds, device):
        rows.append({
            "phase": "baseline",
            "checkpoint_fraction": 0.0,
            "global_step": 0,
            "cumulative_train_time_s": 0.0,
            "estimated_train_energy_j": 0.0,
            "adapter_size_mb": 0.0,
            **m,
        })

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total_batches = len(train_loader) * args.epochs
    checkpoint_steps = {max(1, int(total_batches * f)): f for f in fractions}
    train_time = 0.0
    step = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(args.epochs):
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            loss = model(**batch).loss / args.grad_accum_steps
            loss.backward()
            if step % args.grad_accum_steps == 0 or step == total_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            train_time += time.perf_counter() - t0

            if step in checkpoint_steps:
                ckpt_dir = run_dir / f"checkpoint_step_{step}"
                model.save_pretrained(ckpt_dir)
                adapter_mb = dir_size_mb(ckpt_dir)
                energy = args.gpu_power_w * args.gpu_utilization * train_time
                for m in evaluate(model, eval_loader, label_token_ids, thresholds, device):
                    rows.append({
                        "phase": "lora_checkpoint",
                        "checkpoint_fraction": checkpoint_steps[step],
                        "global_step": step,
                        "cumulative_train_time_s": train_time,
                        "estimated_train_energy_j": energy,
                        "adapter_size_mb": adapter_mb,
                        "checkpoint_dir": str(ckpt_dir),
                        **m,
                    })
                model.train()
                write_outputs(run_dir, metadata, rows)

    metadata["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    write_outputs(run_dir, metadata, rows)
    print(json.dumps({"run_dir": str(run_dir), "metadata": metadata, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
