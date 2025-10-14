

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from z_configs import BGE_MODEL, SPACY_MODEL, DEVICE, RLHF_MODEL
from x_utils import dataset_results_paths
import os





def load_rlhf_data(jsonl_path): 
    """
    Load a JSONL file where each line contains an answer with its precomputed reward and the prompt/query.
    Keep the fields needed for RLHF: 'prompt', 'answer', 'reward'.
    """
    entries = []
    with open(jsonl_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "answer" not in data or "reward" not in data or "query" not in data:
                continue
            entries.append({
                "prompt": data["query"],
                "answer": data["answer"],
                "reward": float(data["reward"])
            })
    return entries



def compute_log_probs(
        model, 
        tokenizer, 
        prompt, 
        answer, 
        device
    ):
    """
    Compute log probabilities for a generated answer conditioned on the prompt.
    """
    full_text = prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    
    # Only keep log-probs corresponding to the answer tokens
    prompt_len = len(tokenizer(prompt)["input_ids"])
    answer_log_probs = token_log_probs[0, prompt_len:]
    return answer_log_probs





def apply_reinforce_update(
    model,
    tokenizer,
    prompt,
    answer_text,
    reward,
    optimizer,
    device
):
    """
    Perform a REINFORCE update using the provided reward for the generated answer, conditioned on the prompt.
    """
    model.train()
    full_text = prompt + answer_text
    input_ids = tokenizer(full_text, return_tensors="pt").to(device)["input_ids"]
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    # Only keep log-probs corresponding to the answer tokens
    prompt_len = len(tokenizer(prompt)["input_ids"])
    answer_log_probs = token_log_probs[0, prompt_len:]

    loss = -reward * answer_log_probs.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





def rlhf_update_from_jsonl(
    model,
    tokenizer,
    jsonl_path,
    optimizer,
    device,
    log_every
):
    """
    Apply REINFORCE updates for all answers in a JSONL file with precomputed rewards.
    Each entry must contain 'query' (prompt), 'answer', and 'reward'.
    """
    entries = load_rlhf_data(jsonl_path)
    for i, entry in enumerate(entries, start=1):
        reward = entry.get("reward", 0.0)
        prompt = entry["prompt"]
        answer_text = entry["answer"]
        apply_reinforce_update(model, tokenizer, prompt, answer_text, reward, optimizer, device)
        if i % log_every == 0:
            print(f"[RLHF] Processed {i}/{len(entries)} entries")













if __name__ == "__main__":
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train", "dev"]
    MODELS = ["qwen-7b", "qwen-14b"]  
    REWARD_CONFIGS = ["reward_config_1", "reward_config_2", "reward_config_3", "reward_config_4", "reward_config_5"]

    LR = 5e-6
    LOG_EVERY = 50

    tokenizer = AutoTokenizer.from_pretrained(RLHF_MODEL)
    rlhf_model = AutoModelForCausalLM.from_pretrained(RLHF_MODEL).to(DEVICE)
    optimizer = torch.optim.AdamW(rlhf_model.parameters(), lr=LR) ################### ???????????????

    for dataset in DATASETS:
        for split in SPLITS:
            for model_name in MODELS:  
                for reward_config in REWARD_CONFIGS:

                    # Get paths
                    paths = dataset_results_paths(dataset, split, model_name, reward_config)

                    # Skip if JSONL doesn't exist
                    if not os.path.exists(paths["jsonl_path"]):
                        print(f"[skip] JSONL not found: {paths['jsonl_path']}")
                        continue

                    # Run RLHF updates
                    print(f"[RLHF] Training on {dataset}/{split}/{model_name}/{reward_config}...")
                    rlhf_update_from_jsonl(
                        model=rlhf_model,
                        tokenizer=tokenizer,
                        jsonl_path=paths["jsonl_path"],
                        optimizer=optimizer,
                        device=DEVICE,
                        log_every=LOG_EVERY
                    )

                    # Save updated RLHF model after each reward config
                    rlhf_model.save_pretrained(paths["updated_model_dir"]) 
                    print(f"[RLHF] Model saved to {paths['updated_model_dir']}")
