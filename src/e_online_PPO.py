"""
Online PPO module for RLHF.

Uses a reward model to compute scalar rewards for candidate responses,
then updates a causal LM using a PPO-style clipped policy gradient.

"""

import os
from typing import List, Dict
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from x_utils import load_jsonl, dataset_results_paths
from z_configs import DEVICE, RLHF_MODEL

# -----------------------------
# Utilities
# -----------------------------
def tokenize_prompt_answer(tokenizer, prompt, answer, device, max_length=None):
    """Tokenize prompt + answer, return input_ids, attention_mask, prompt_len"""
    max_len = max_length or getattr(tokenizer, "model_max_length", None)
    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)
    a = tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)
    input_ids = torch.cat([p["input_ids"], a["input_ids"]], dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    prompt_len = p["input_ids"].size(1)
    return input_ids, attention_mask, prompt_len

def compute_reward(reward_model, tokenizer, prompt, answer, device, reward_mode="regression"):
    """Compute scalar reward using trained reward model."""
    input_ids, attention_mask, _ = tokenize_prompt_answer(tokenizer, prompt, answer, device)
    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)
        if reward_mode == "classification":
            probs = torch.softmax(logits, dim=-1)
            return float(probs[1].item())
        elif reward_mode == "sigmoid":
            return float(torch.sigmoid(logits).item())
        else:
            return float(logits.item() if logits.numel() == 1 else logits.mean().item())

def compute_token_log_probs(model, input_ids, attention_mask=None):
    """Compute log-probs per token."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        label_mask = attention_mask[:, 1:].float().to(token_log_probs.device)
    else:
        label_mask = torch.ones_like(token_log_probs)
    token_log_probs = token_log_probs * label_mask
    valid_counts = label_mask.sum(dim=1)
    return token_log_probs, valid_counts

# -----------------------------
# PPO Update
# -----------------------------
def ppo_update(
    model,
    tokenizer,
    prompt,
    answer,
    reward,
    old_log_probs,
    optimizer,
    device,
    clip_epsilon=0.2,
    max_length=None
):
    """Perform one PPO policy gradient update."""
    model.train()
    input_ids, attention_mask, prompt_len = tokenize_prompt_answer(tokenizer, prompt, answer, device, max_length)
    token_log_probs, valid_counts = compute_token_log_probs(model, input_ids, attention_mask)
    answer_log_probs = token_log_probs[0, prompt_len:]
    answer_valid = valid_counts[0] - prompt_len
    if answer_valid <= 0:
        return 0.0

    # Average log-prob over answer tokens
    mean_logprob = answer_log_probs.sum() / answer_valid

    # PPO clipped objective
    ratio = torch.exp(mean_logprob - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.min(ratio * reward, clipped_ratio * reward)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), float(mean_logprob.item())

# -----------------------------
# Main training loop
# -----------------------------
def train_online_ppo(
    model,
    tokenizer,
    reward_model,
    jsonl_path,
    optimizer,
    device=DEVICE,
    reward_mode="regression",
    clip_epsilon=0.2,
    max_length=None,
    log_every=20
):
    """Online PPO updates from dataset of (prompt, answer) entries."""
    entries = load_jsonl(jsonl_path)
    baseline = 0.0
    alpha = 0.9
    eps = 1e-8

    for i, entry in enumerate(entries, start=1):
        prompt = entry["query"]
        answer = entry["answer"]

        # --- Compute reward ---
        reward = compute_reward(reward_model, tokenizer, prompt, answer, device, reward_mode)

        # --- Baseline-adjusted reward ---
        baseline = alpha * baseline + (1 - alpha) * reward
        adjusted_reward = reward - baseline
        adjusted_reward_tensor = torch.tensor(adjusted_reward / (abs(baseline) + eps), dtype=torch.float32, device=device)
        adjusted_reward_tensor = torch.clamp(adjusted_reward_tensor, -5.0, 5.0)

        # --- Compute old log-probs (for PPO ratio) ---
        input_ids, attention_mask, prompt_len = tokenize_prompt_answer(tokenizer, prompt, answer, device, max_length)
        token_log_probs, valid_counts = compute_token_log_probs(model, input_ids, attention_mask)
        old_log_probs = (token_log_probs[0, prompt_len:].sum() / (valid_counts[0] - prompt_len)).detach()

        # --- PPO update ---
        loss, new_logprob = ppo_update(model, tokenizer, prompt, answer, adjusted_reward_tensor, old_log_probs, optimizer, device, clip_epsilon, max_length)

        if i % log_every == 0:
            print(f"[PPO] Step {i}/{len(entries)} | Loss: {loss:.6f} | Reward: {reward:.6f} | Adj: {adjusted_reward_tensor.item():.6f}")

# -----------------------------
# Script entry
# -----------------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(RLHF_MODEL)
    ppo_model = AutoModelForCausalLM.from_pretrained(RLHF_MODEL).to(DEVICE)
    optimizer = AdamW(ppo_model.parameters(), lr=5e-6)

    REWARD_MODEL_NAME = "your-reward-model-name"  # e.g., "OpenAssistant/reward-model-deberta-v3-base"
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME).to(DEVICE)

    DATASETS = ["hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train"]

    for dataset in DATASETS:
        for split in SPLITS:
            paths = dataset_results_paths(dataset, split, RLHF_MODEL, "ppo")
            if not os.path.exists(paths["jsonl_path"]):
                print(f"[skip] JSONL not found: {paths['jsonl_path']}")
                continue

            print(f"[PPO] Training on {dataset}/{split}...")
            train_online_ppo(
                model=ppo_model,
                tokenizer=tokenizer,
                reward_model=reward_model,
                jsonl_path=paths["jsonl_path"],
                optimizer=optimizer,
                device=DEVICE,
                reward_mode="regression"
            )
