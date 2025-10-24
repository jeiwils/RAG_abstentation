
"""

TO DO: 
- train reward model in separate module 





"""




"""

regression-based RLHF module - trains policy model 



"""



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from z_configs import BGE_MODEL, SPACY_MODEL, DEVICE, POLICY_MODEL, REWARD_MODEL
from x_utils import dataset_results_paths, load_jsonl
import os
from transformers import AutoModelForSequenceClassification

from .rlhf_utils import tokenize_prompt_answer, compute_token_log_probs, compute_reward, update_baseline, save_model_and_tokenizer







def load_rlhf_data(
        jsonl_path, 
        mode="offline"
        ):
    """
    Load RLHF dataset entries from a JSONL file.
    Each line must contain 'query' and 'answer'.
    If mode='offline', also expect a numeric 'reward'.
    """
    entries = []

    for data in load_jsonl(jsonl_path, log_skipped=True):
        if not {"query", "answer"}.issubset(data.keys()):
            continue

        entry = {
            "prompt": data["query"],
            "answer": data["answer"]
        }

        if mode == "offline" and "reward" in data:
            entry["reward"] = float(data["reward"])

        entries.append(entry)

    return entries



def update_policy_with_reward(model, tokenizer, prompt, answer_text, reward, optimizer, device, max_length=None):
    """
    Perform a REINFORCE-style policy update for one (prompt, answer, reward) tuple.
    """
    model.train()

    # ✅ updated to match new utils format
    encoded = tokenize_prompt_answer(tokenizer, prompt, answer_text, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = encoded["prompt_length"]

    # compute token-level log-probs with gradient
    token_log_probs = compute_token_log_probs(model, input_ids, attention_mask)  # [1, seq_len-1]
    answer_log_probs = token_log_probs[0, prompt_len:]
    answer_valid = answer_log_probs.numel()

    if answer_valid <= 0:
        return 0.0  # nothing to update

    # ensure reward is a tensor on correct device/dtype
    reward_t = torch.tensor(float(reward), dtype=answer_log_probs.dtype, device=device)

    # average over valid answer tokens
    mean_logprob = answer_log_probs.mean()
    loss = - reward_t * mean_logprob

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())





def train_rlhf(
    model,
    tokenizer,
    jsonl_path,
    optimizer,
    paths,
    device="cuda",
    mode="offline",
    reward_model=None,
    reward_mode="regression",
    log_every=50,
    save_every=200,
    max_length=None
):
    """RLHF training (REINFORCE-style) with baseline-adjusted reward and standardized saving."""
    baseline = 0.0
    alpha = 0.9
    eps = 1e-8
    entries = load_jsonl(jsonl_path)

    for i, entry in enumerate(entries, start=1):
        prompt = entry["query"] if mode=="online" else entry.get("prompt", entry.get("query"))
        answer = entry["answer"]

        # compute reward
        if mode == "offline":
            reward = float(entry.get("reward", 0.0))
        else:
            assert reward_model is not None
            encoded = tokenize_prompt_answer(tokenizer, prompt, answer, max_length=max_length)
            reward = compute_reward(reward_model, encoded["input_ids"].to(device), encoded["attention_mask"].to(device), reward_mode).item()

        # baseline update
        baseline, adjusted_reward_tensor = update_baseline(baseline, torch.tensor([reward], device=device), alpha=alpha)
        adjusted_reward = adjusted_reward_tensor.item()
        adjusted_reward_tensor = torch.clamp(torch.tensor(adjusted_reward / (max(abs(baseline), eps)+eps), device=device), -5.0, 5.0)

        # policy update
        update_policy_with_reward(model, tokenizer, prompt, answer, adjusted_reward_tensor, optimizer, device, max_length=max_length)

        # logging
        if i % log_every == 0:
            print(f"[RLHF] Step {i}/{len(entries)} | Reward: {reward:.4f} | Adj: {adjusted_reward_tensor.item():.4f}")

        # checkpointing
        if i % save_every == 0 or i == len(entries):
            save_model_and_tokenizer(model, tokenizer, paths, step=i)




if __name__ == "__main__":


    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train", "dev"]
    MODELS = ["qwen-7b", "qwen-14b"]
    REWARD_CONFIGS = [
        "reward_config_1", "reward_config_2", "reward_config_3",
        "reward_config_4", "reward_config_5"
    ]
    RLHF_MODE = "offline"  
    LR = 5e-6


    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
    rlhf_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(rlhf_model.parameters(), lr=LR)

    reward_model = None
    if RLHF_MODE == "online":
        print(f"[RLHF] Loading reward model from {REWARD_MODEL}...")
        reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL).to(DEVICE)


    for dataset in DATASETS:
        for split in SPLITS:
            for model_name in MODELS:
                for reward_config in REWARD_CONFIGS:

                    paths = dataset_results_paths(dataset, split, model_name, reward_config)

                    if not os.path.exists(paths["jsonl_path"]):
                        print(f"[skip] JSONL not found: {paths['jsonl_path']}")
                        continue

                    print(f"[RLHF] Training on {dataset}/{split}/{model_name}/{reward_config} ({RLHF_MODE})...")

                    # Call the unified RLHF update function
                    train_rlhf(
                        model=rlhf_model,
                        tokenizer=tokenizer,
                        jsonl_path=paths["jsonl_path"],
                        optimizer=optimizer,
                        device=DEVICE,
                        paths=paths,  
                        mode=RLHF_MODE,
                        reward_model=reward_model
                    )
