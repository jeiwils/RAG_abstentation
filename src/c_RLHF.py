
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
from z_configs import BGE_MODEL, SPACY_MODEL, DEVICE, RLHF_MODEL
from x_utils import dataset_results_paths, load_jsonl
import os
from transformers import AutoModelForSequenceClassification







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






def tokenize_prompt_answer(tokenizer, prompt, answer, device, max_length=None):
    """
    Tokenizes prompt and answer separately then concatenates.
    Returns:
      - input_ids: LongTensor shape [1, seq_len] on device
      - attention_mask: LongTensor shape [1, seq_len] on device
      - prompt_len: int number of tokens in the prompt
    """
    # Allow optional truncation to model max length
    max_len = max_length or getattr(tokenizer, "model_max_length", None)

    p = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)
    a = tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_len)

    prompt_ids = p["input_ids"]
    answer_ids = a["input_ids"]

    input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)
    # build attention mask: ones for the concatenated tokens
    attention_mask = torch.ones_like(input_ids)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    prompt_len = prompt_ids.size(1)
    return input_ids, attention_mask, prompt_len



def compute_reward(reward_model, tokenizer, prompt, answer, device, reward_mode="regression"):
    """
    Compute scalar reward using a trained reward model.

    reward_mode:
      - "regression": expects a single scalar output
      - "classification": expects logits with dim >= 2; uses softmax probability for class 1
      - "sigmoid": expects single logit, applies sigmoid to map to (0,1)
    Returns a python float.
    """
    input_ids, attention_mask, _ = tokenize_prompt_answer(tokenizer, prompt, answer, device)
    reward_model.eval()
    with torch.no_grad():
        outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)  # assume batch=1

        if reward_mode == "classification":
            probs = torch.softmax(logits, dim=-1)
            reward_val = probs[1].item()  # probability of preferred class
        elif reward_mode == "sigmoid":
            reward_val = torch.sigmoid(logits).item()
        else:
            reward_val = logits.item() if logits.numel() == 1 else logits.mean().item()

    return float(reward_val)






def compute_token_log_probs(
        model, 
        input_ids, 
        attention_mask=None # 'masks' padding tokens away from transformer attention 
        ):
    """
    Computes log-probs for next-token prediction with masking.
    Returns:
        - token_log_probs: tensor [batch, seq_len-1] (log-probs per token)
        - valid_counts: tensor [batch] (number of non-padding tokens per example)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # shift logits and labels to align prediction
    shift_logits = logits[:, :-1, :].contiguous()   # predict token 1..T-1 from 0..T-2
    shift_labels = input_ids[:, 1:].contiguous()    # tokens 1..T-1

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len-1]

    # handle masking
    if attention_mask is not None:
        label_mask = attention_mask[:, 1:].contiguous().to(token_log_probs.device).float()
    else:
        label_mask = torch.ones_like(token_log_probs, dtype=torch.float, device=token_log_probs.device)

    token_log_probs = token_log_probs * label_mask
    valid_counts = label_mask.sum(dim=1)  # number of valid positions per example

    return token_log_probs, valid_counts




def update_policy_with_reward(model, tokenizer, prompt, answer_text, reward, optimizer, device, max_length=None):
    """
    Perform a REINFORCE-style policy update for one (prompt, answer, reward) tuple.
    """
    model.train()
    input_ids, attention_mask, prompt_len = tokenize_prompt_answer(tokenizer, prompt, answer_text, device, max_length=max_length)

    # compute token-level log-probs with gradient
    token_log_probs, valid_counts = compute_token_log_probs(model, input_ids, attention_mask=attention_mask)  # [1, seq_len-1]
    answer_log_probs = token_log_probs[0, prompt_len:]
    answer_valid = valid_counts[0] - prompt_len

    if answer_valid <= 0:
        return 0.0  # nothing to update

    # ensure reward is a tensor on correct device/dtype
    if not torch.is_tensor(reward):
        reward_t = torch.tensor(float(reward), dtype=answer_log_probs.dtype, device=device)
    else:
        reward_t = reward.to(device).type(answer_log_probs.dtype)

    # average over valid answer tokens
    mean_logprob = answer_log_probs.sum() / answer_valid
    loss = - reward_t * mean_logprob

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())




def rlhf_update_from_jsonl(
    model,
    tokenizer,
    jsonl_path,
    optimizer,
    device,
    paths,
    log_every=50,
    mode="offline",
    reward_model=None,
    reward_mode="regression",
    max_length=None
):
    SAVE_EVERY = 200
    baseline = 0.0
    alpha = 0.9  # smoothing factor for running average
    eps = 1e-8

    entries = load_rlhf_data(jsonl_path, mode=mode)
    for i, entry in enumerate(entries, start=1):
        prompt = entry["prompt"]
        answer = entry["answer"]

        # --- Reward computation ---
        if mode == "offline":
            reward = float(entry.get("reward", 0.0))
        elif mode == "online":
            assert reward_model is not None, "Reward model required for online RLHF"
            reward = compute_reward(reward_model, tokenizer, prompt, answer, device, reward_mode)

        # --- Baseline update & adjusted reward ---
        baseline = alpha * baseline + (1 - alpha) * reward
        adjusted_reward = reward - baseline

        # --- Normalize / clip for stability (industry-standard lightweight) ---
        adjusted_reward_tensor = torch.tensor(adjusted_reward, dtype=torch.float32, device=device)
        # normalize by baseline magnitude to keep scale stable
        denom = max(abs(baseline), eps)
        adjusted_reward_tensor = adjusted_reward_tensor / (denom + eps)
        adjusted_reward_tensor = torch.clamp(adjusted_reward_tensor, -5.0, 5.0)

        # --- Policy update ---
        loss = update_policy_with_reward(model, tokenizer, prompt, answer, adjusted_reward_tensor, optimizer, device, max_length=max_length)

        # --- Logging ---
        if i % log_every == 0:
            print(f"[RLHF] Processed {i}/{len(entries)} entries | Loss: {loss:.6f} | Reward: {reward:.6f} | Adjusted: {adjusted_reward_tensor.item():.6f}")

        # --- Periodic checkpoint saving ---
        if i % SAVE_EVERY == 0 or i == len(entries):
            save_path = os.path.join(paths["updated_model_dir"], f"step_{i}")
            os.makedirs(save_path, exist_ok=True)
            # For speed you might prefer torch.save(model.state_dict(), ...)
            model.save_pretrained(save_path)
            print(f"[RLHF] Checkpoint saved at step {i}")





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


    tokenizer = AutoTokenizer.from_pretrained(RLHF_MODEL)
    rlhf_model = AutoModelForCausalLM.from_pretrained(RLHF_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(rlhf_model.parameters(), lr=LR)

    # reward_model = None
    # if RLHF_MODE == "online":
    #     reward_model = AutoModelForSequenceClassification.from_pretrained(RLHF_MODEL).to(DEVICE)

    # Placeholder for your actual reward model
    REWARD_MODEL_NAME = "your-reward-model-name"  # e.g., "OpenAssistant/reward-model-deberta-v3-base"

    reward_model = None
    if RLHF_MODE == "online":
        print(f"[RLHF] Loading reward model from {REWARD_MODEL_NAME}...")
        reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME).to(DEVICE)



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
                    rlhf_update_from_jsonl(
                        model=rlhf_model,
                        tokenizer=tokenizer,
                        jsonl_path=paths["jsonl_path"],
                        optimizer=optimizer,
                        device=DEVICE,
                        paths=paths,   # ✅ added
                        mode=RLHF_MODE,
                        reward_model=reward_model
                    )
