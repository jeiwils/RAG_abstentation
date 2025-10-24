

import torch
from torch.nn import functional as F
import os 






def save_model_and_tokenizer(model, tokenizer, paths, step=None):
    """Save model/tokenizer periodically or finally."""
    folder = "final" if step is None else f"step_{step}"
    save_path = os.path.join(paths["updated_model_dir"], folder)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[INFO] Model saved at {save_path}")






import torch
import os

def tokenize_prompt_answer(tokenizer, prompt, answer=None, max_length=None, return_tensors="pt"):
    """
    Tokenize a prompt (and optionally an answer) for RLHF, PPO, DPO pipelines.
    - Keeps the full prompt for proper conditioning.
    - Computes prompt_length on untruncated prompt for masking.
    Returns: dict with input_ids, attention_mask, prompt_length.
    """
    # Ensure space between prompt and answer
    if answer is not None:
        text = prompt + " " + answer
    else:
        text = prompt

    # Tokenize full sequence
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=return_tensors,
    )

    # Prompt length computed on untruncated prompt
    prompt_ids = tokenizer(prompt, truncation=False, return_tensors=return_tensors).input_ids[0]
    encoded["prompt_length"] = len(prompt_ids)

    return encoded


def batch_tokenize(tokenizer, prompts, answers=None, max_length=None, return_tensors="pt"):
    """
    Tokenize batches of (prompt, answer) pairs.
    - Keeps full prompt for each pair.
    - Computes prompt_length individually on untruncated prompts.
    """
    if answers is not None:
        texts = [p + " " + a for p, a in zip(prompts, answers)]
    else:
        texts = prompts

    # Tokenize full batch
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors=return_tensors,
    )

    # Compute prompt lengths individually (on untruncated prompts)
    encoded["prompt_length"] = [len(tokenizer(p, truncation=False).input_ids) for p in prompts]

    return encoded





def compute_token_log_probs(model, input_ids, attention_mask):
    """
    Compute log-probabilities per token under the model.
    Returns tensor of shape (batch_size, seq_len).
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
        token_log_probs = token_log_probs * mask
    return token_log_probs







def compute_reward(reward_model, input_ids, attention_mask, mode="sigmoid"):
    """
    Compute scalar rewards given a reward model output.
    Supports:
      - regression: raw scalar output
      - classification: logit[1]
      - sigmoid: sigmoid(logit)


    potential problem: If reward model outputs more classes, or the shape differs, you can get incorrect rewards or runtime errors. Could silently produce wrong reward values.
    """
    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        if mode == "regression":
            reward = logits.squeeze(-1)
        elif mode == "classification":
            reward = logits[:, 1]
        else:  # sigmoid (default)
            reward = torch.sigmoid(logits.squeeze(-1))

    return reward






def update_baseline(old_baseline, rewards, alpha=0.9):
    """
    Exponential moving average baseline to reduce variance in PPO/RLHF.
    """
    mean_reward = rewards.mean().item()
    new_baseline = alpha * old_baseline + (1 - alpha) * mean_reward
    adjusted_rewards = rewards - new_baseline
    return new_baseline, adjusted_rewards
