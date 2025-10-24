"""
Reward Model Training Module

Supports:
- Scalar regression mode (for "IDK" / abstention training)
- Pairwise preference mode (for standard RLHF preference learning)

Usage:
    python train_reward_model.py
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from x_utils import load_jsonl, dataset_results_paths
from z_configs import REWARD_MODEL, DEVICE


# ---------------------------
# Dataset definitions
# ---------------------------

class ScalarRewardDataset(Dataset):
    """For scalar (regression-style) reward model training."""
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expect entries like {"prompt": "...", "response": "...", "score": float}
        self.data = [
            d for d in self.data if "prompt" in d and "response" in d and "score" in d
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"Question: {item['prompt']}\nAnswer: {item['response']}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        score = torch.tensor([float(item["score"])], dtype=torch.float)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": score,
        }


class PairwiseRewardDataset(Dataset):
    """For preference-style reward model training."""
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = load_jsonl(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expect entries like {"prompt": "...", "chosen": "...", "rejected": "..."}
        self.data = [
            d for d in self.data if "prompt" in d and "chosen" in d and "rejected" in d
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        chosen_enc = self.tokenizer(
            f"Question: {item['prompt']}\nAnswer: {item['chosen']}",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            f"Question: {item['prompt']}\nAnswer: {item['rejected']}",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


# ---------------------------
# Training logic
# ---------------------------

def train_reward_model(
    model_name_or_path,
    data_path,
    save_dir,
    mode="scalar",
    lr=1e-5,
    batch_size=4,
    epochs=3,
    max_length=512,
    warmup_ratio=0.05,
):
    """
    Trains a reward model in either scalar or pairwise mode.

    Args:
        model_name_or_path: base model to fine-tune (e.g., "roberta-base")
        data_path: JSONL file containing training data
        save_dir: directory to save trained model
        mode: "scalar" or "pairwise"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1
    ).to(DEVICE)

    if mode == "scalar":
        dataset = ScalarRewardDataset(data_path, tokenizer, max_length)
    elif mode == "pairwise":
        dataset = PairwiseRewardDataset(data_path, tokenizer, max_length)
    else:
        raise ValueError("mode must be 'scalar' or 'pairwise'")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * warmup_ratio), num_training_steps=total_steps
    )

    loss_fn = nn.MSELoss() if mode == "scalar" else nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()

            if mode == "scalar":
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                )
                preds = outputs.logits.squeeze(-1)
                loss = loss_fn(preds, batch["labels"].to(DEVICE))

            else:  # pairwise
                chosen_out = model(
                    input_ids=batch["chosen_input_ids"].to(DEVICE),
                    attention_mask=batch["chosen_attention_mask"].to(DEVICE),
                )
                rejected_out = model(
                    input_ids=batch["rejected_input_ids"].to(DEVICE),
                    attention_mask=batch["rejected_attention_mask"].to(DEVICE),
                )
                diff = chosen_out.logits - rejected_out.logits
                # standard preference loss: log(sigmoid(diff))
                loss = -torch.nn.functional.logsigmoid(diff).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[✓] Reward model saved to {save_dir}")


# ---------------------------
# Main entrypoint
# ---------------------------

if __name__ == "__main__":
    DATA_PATH = "data/reward_data.jsonl"   # change as needed
    SAVE_DIR = "trained_reward_model"
    MODE = "scalar"  # or "pairwise"

    train_reward_model(
        model_name_or_path=REWARD_MODEL,  # base model
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        mode=MODE,
        lr=1e-5,
        batch_size=4,
        epochs=3,
    )
