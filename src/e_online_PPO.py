"""
Online PPO module for RLHF.

Uses a reward model to compute scalar rewards for candidate responses,
then updates a causal LM using a PPO-style clipped policy gradient.

"""

import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from x_utils import load_jsonl, dataset_results_paths
from z_configs import DEVICE, POLICY_MODEL, REWARD_MODEL
from .rlhf_utils import tokenize_prompt_answer, compute_token_log_probs, compute_reward, update_baseline, save_model_and_tokenizer










def ppo_update(
    model,
    input_ids,
    attention_mask,
    prompt_len,
    reward_tensor,
    old_log_probs,
    optimizer,
    clip_epsilon=0.2,
    max_grad_norm=1.0
):
    """
    Perform one PPO policy gradient update at the token level.

    Args:
        model: causal LM.
        input_ids, attention_mask: tokenized prompt+answer.
        prompt_len: number of prompt tokens.
        reward_tensor: scalar advantage/reward for the answer.
        old_log_probs: tensor of shape (answer_len,) with detached old log-probs.
        optimizer: optimizer for the model.
    """
    model.train()

    # Compute new log-probs for all tokens
    token_log_probs = compute_token_log_probs(model, input_ids, attention_mask)
    # Only keep log-probs of answer tokens
    answer_log_probs = token_log_probs[0, prompt_len:]  # shape: (answer_len,)

    if answer_log_probs.numel() == 0:
        return 0.0, 0.0

    # PPO ratio: new / old
    ratio = torch.exp(answer_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    # Policy loss (per-token)
    loss = -torch.min(ratio * reward_tensor, clipped_ratio * reward_tensor).mean()

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.item()), float(answer_log_probs.mean().item())







def train_ppo(
    model,
    tokenizer,
    reward_model,
    jsonl_path,
    optimizer,
    paths,
    device="cuda",
    reward_mode="regression",
    clip_epsilon=0.2,
    log_every=20,
    save_every=200,
    max_length=None
):
    entries = load_jsonl(jsonl_path)
    baseline = 0.0
    alpha = 0.9
    eps = 1e-8

    for i, entry in enumerate(entries, start=1):
        prompt = entry["query"]
        answer = entry["answer"]

        # Tokenize once
        encoded = tokenize_prompt_answer(tokenizer, prompt, answer, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_len = encoded["prompt_length"]

        # Compute reward
        reward = compute_reward(reward_model, input_ids, attention_mask, reward_mode).item()

        # Update baseline (running average)
        baseline, adjusted_reward_tensor = update_baseline(
            baseline,
            torch.tensor([reward], device=device),
            alpha=alpha
        )

        # Standardize and clamp advantage
        advantage = adjusted_reward_tensor / (adjusted_reward_tensor.std() + eps)
        advantage = torch.clamp(advantage, -5.0, 5.0)

        # Compute old log-probs per token
        token_log_probs = compute_token_log_probs(model, input_ids, attention_mask)
        old_log_probs = token_log_probs[0, prompt_len:].detach()  # shape: (answer_len,)

        # PPO update
        loss, mean_logprob = ppo_update(
            model,
            input_ids,
            attention_mask,
            prompt_len,
            advantage,
            old_log_probs,
            optimizer,
            clip_epsilon
        )

        # Logging
        if i % log_every == 0:
            print(f"[PPO] Step {i}/{len(entries)} | Reward: {reward:.4f} | Adv: {advantage.item():.4f} | Loss: {loss:.4f}")

        # Checkpoint
        if i % save_every == 0 or i == len(entries):
            save_model_and_tokenizer(model, tokenizer, paths, step=i)







if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
    ppo_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL).to(DEVICE)
    optimizer = AdamW(ppo_model.parameters(), lr=5e-6)

    # Load reward model from config
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(DEVICE)

    DATASETS = ["hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train"]

    for dataset in DATASETS:
        for split in SPLITS:
            paths = dataset_results_paths(dataset, split, POLICY_MODEL, "PPO")
            if not os.path.exists(paths["jsonl_path"]):
                print(f"[skip] JSONL not found: {paths['jsonl_path']}")
                continue

            print(f"[PPO] Training on {dataset}/{split}/{POLICY_MODEL}/PPO...")
            train_ppo(
                model=ppo_model,
                tokenizer=tokenizer,
                reward_model=reward_model,
                jsonl_path=paths["jsonl_path"],
                optimizer=optimizer,
                paths=paths,
                device=DEVICE,
                reward_mode="regression"
            )
