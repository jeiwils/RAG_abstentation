"""
Offline DPO training module.

Loads RAG-generated candidate answers, constructs preferred/rejected pairs,
and fine-tunes the model using offline DPO.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from .x_utils import load_jsonl, dataset_results_paths
import torch.nn.functional as F

# -----------------------------
# Dataset for offline DPO
# -----------------------------
class DPODataset(Dataset):
    """
    Stores (query, chosen, rejected) triplets for DPO training
    """
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return e["query"], e["chosen"], e["rejected"]


# -----------------------------
# Load candidates and construct DPO pairs
# -----------------------------
def load_dpo_pairs(dataset, split, model_name, config_name, seed):
    paths = dataset_results_paths(
        dataset=dataset,
        split=split,
        model_name=model_name,
        paradigm=config_name,
        reward_scheme=f"reward_config_{seed + 1}",
        seed=seed
    )
    answers_path = paths["answers"]
    entries = load_jsonl(answers_path)
    pairs = []

    for entry in entries:
        query_text = entry["query"]
        final_answer = {"answer": entry["final_answer"]}
        candidates = entry.get("answers", [])
        # Preferred = final answer, Rejected = others
        for cand in candidates:
            if cand["answer"] != final_answer["answer"]:
                pairs.append({
                    "query": query_text,
                    "chosen": final_answer["answer"],
                    "rejected": cand["answer"]
                })
    return pairs


# -----------------------------
# Tokenization utility
# -----------------------------
def tokenize_triplet(tokenizer, query, chosen, rejected, max_length=512):
    """
    Returns input_ids and attention_mask for chosen and rejected sequences.
    Concatenate query + answer as input for causal LM.
    """
    chosen_enc = tokenizer(
        query + "\n" + chosen,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    rejected_enc = tokenizer(
        query + "\n" + rejected,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return chosen_enc, rejected_enc


# -----------------------------
# DPO loss computation
# -----------------------------
def dpo_loss(chosen_log_probs, rejected_log_probs, beta=0.1):
    """
    Offline DPO loss: log softmax over chosen vs rejected
    L = -log( sigmoid( beta*(R(chosen)-R(rejected)) ) )
    Here we approximate R by log-prob difference
    """
    # log p_chosen - log p_rejected
    diff = chosen_log_probs - rejected_log_probs
    loss = -torch.log(torch.sigmoid(beta * diff) + 1e-8).mean()
    return loss


# -----------------------------
# Training function
# -----------------------------
def train_dpo(
    dataset,
    split,
    model_name,
    model_path,
    config_name,
    seed,
    tokenizer_name=None,
    lr=5e-6,
    batch_size=2,
    epochs=1,
    device="cuda"
):
    # Load pretrained model + tokenizer
    tokenizer_name = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.train()

    # Load DPO pairs
    print(f"[INFO] Loading DPO pairs for {dataset}/{split}/{model_name}/{config_name}/{seed}")
    pairs = load_dpo_pairs(dataset, split, model_name, config_name, seed)
    dataset_obj = DPODataset(pairs)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}]")
        for batch in tqdm(dataloader, desc="DPO training"):
            queries, chosens, rejecteds = batch
            batch_loss = 0.0
            optimizer.zero_grad()
            for q, c, r in zip(queries, chosens, rejecteds):
                c_enc, r_enc = tokenize_triplet(tokenizer, q, c, r)
                c_enc = {k: v.to(device) for k,v in c_enc.items()}
                r_enc = {k: v.to(device) for k,v in r_enc.items()}

                with torch.cuda.amp.autocast():  # mixed precision if possible
                    c_logits = model(**c_enc).logits
                    r_logits = model(**r_enc).logits

                    # Sum log-probs over sequence
                    c_log_probs = F.log_softmax(c_logits, dim=-1)
                    r_log_probs = F.log_softmax(r_logits, dim=-1)
                    # gather token probabilities for labels
                    c_labels = c_enc["input_ids"]
                    r_labels = r_enc["input_ids"]
                    chosen_lp = torch.gather(c_log_probs, 2, c_labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)
                    rejected_lp = torch.gather(r_log_probs, 2, r_labels.unsqueeze(-1)).squeeze(-1).sum(dim=1)

                    loss = dpo_loss(chosen_lp, rejected_lp)
                    loss.backward()
                    batch_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1} complete. Avg loss: {batch_loss/len(dataloader):.4f}")

    # Save fine-tuned model
    save_dir = f"models/dpo_finetuned/{dataset}_{split}_{model_name}_{config_name}_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] DPO-finetuned model saved to {save_dir}")


# -----------------------------
# __main__ for batch training
# -----------------------------
if __name__ == "__main__":
    datasets = ["hotpotqa", "2wikimultihopqa"]
    splits = ["train"]
    models = {
        "7B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-7B-Instruct",
        "14B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-14B-Instruct"
    }
    reward_configs = ["DPO"]
    seeds = [0]

    for dataset in datasets:
        for split in splits:
            for model_name, model_path in models.items():
                for config_name in reward_configs:
                    for seed in seeds:
                        train_dpo(
                            dataset,
                            split,
                            model_name,
                            model_path,
                            config_name,
                            seed,
                            lr=5e-6,
                            batch_size=1,
                            epochs=1,
                            device="cuda"
                        )
