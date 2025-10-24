"""
Offline DPO training module.

Loads RAG-generated candidate answers, constructs preferred/rejected pairs,
and fine-tunes the model using offline DPO.




"""


import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from .x_utils import load_jsonl, dataset_results_paths, model_paths, processed_dataset_paths
from .b_answer_gen import rerank_answers
from z_configs import BGE_MODEL, DEVICE, reward_scheme
from sentence_transformers import SentenceTransformer


from .rlhf_utils import tokenize_prompt_answer, compute_token_log_probs, batch_tokenize, save_model_and_tokenizer






class DPODataset(Dataset):
    """Stores (query, chosen, rejected) triplets for DPO training."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e = self.pairs[idx]
        return e["query"], e["chosen"], e["rejected"]





def dpo_collate_fn(batch):
    queries, chosens, rejecteds = zip(*batch)
    return list(queries), list(chosens), list(rejecteds)




def construct_dpo_pairs(
        dataset, 
        split, 
        model_name, 
        seed, 
        passages, 
        embed_model, 
        reward_scheme
        ):
    """
    Construct DPO pairs (chosen vs rejected) for offline training.

    Returns:
        list[dict]: Each dict has keys: "query", "chosen", "rejected".
    """
    paths = dataset_results_paths(
        dataset=dataset,
        split=split,
        model_name=model_name,
        paradigm="DPO",
        reward_scheme=f"reward_config_{seed + 1}",
        seed=seed
    )
    entries = load_jsonl(paths["answers"])
    pairs = []

    for idx, entry in enumerate(entries):
        query_text = entry["query"]
        candidates = entry.get("answers", [])
        if not candidates:
            continue

        # Rank candidates using rerank_answers()
        sorted_candidates = rerank_answers(candidates, passages[idx], embed_model, reward_scheme)
        top_candidate = sorted_candidates[0][2]     
        bottom_candidate = sorted_candidates[-1][2] 

        pairs.append({
            "query": query_text.strip(),
            "chosen": top_candidate["answer"].strip(),
            "rejected": bottom_candidate["answer"].strip()
        })

    return pairs





# -----------------------
# DPO loss
# -----------------------
def compute_dpo_loss(model, tokenizer, queries, chosens, rejecteds, device="cuda", max_length=512, beta=0.1):
    """
    Compute Direct Preference Optimization (DPO) loss for a batch.
    Fully differentiable; model.train() should be active.
    """
    # Concatenate queries + responses
    chosen_texts = [f"{q}\n{c}" for q, c in zip(queries, chosens)]
    rejected_texts = [f"{q}\n{r}" for q, r in zip(queries, rejecteds)]

    # Tokenize
    chosen_batch = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    rejected_batch = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # Forward pass to get logits
    with autocast(device_type='cuda', dtype=torch.float16):
        chosen_logits = model(chosen_batch["input_ids"], attention_mask=chosen_batch["attention_mask"]).logits
        rejected_logits = model(rejected_batch["input_ids"], attention_mask=rejected_batch["attention_mask"]).logits

    # Compute per-token log-probs
    chosen_logprobs = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
    rejected_logprobs = torch.nn.functional.log_softmax(rejected_logits, dim=-1)

    # Gather log-probs of actual tokens
    chosen_token_ids = chosen_batch["input_ids"].unsqueeze(-1)
    rejected_token_ids = rejected_batch["input_ids"].unsqueeze(-1)

    chosen_token_logprobs = torch.gather(chosen_logprobs, -1, chosen_token_ids).squeeze(-1)
    rejected_token_logprobs = torch.gather(rejected_logprobs, -1, rejected_token_ids).squeeze(-1)

    # Average over non-padding tokens
    chosen_mask = chosen_batch["attention_mask"]
    rejected_mask = rejected_batch["attention_mask"]

    chosen_mean = (chosen_token_logprobs * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
    rejected_mean = (rejected_token_logprobs * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)

    # DPO pairwise loss
    diff = chosen_mean - rejected_mean
    loss = -torch.log(torch.sigmoid(beta * diff) + 1e-8).mean()

    return loss

# -----------------------
# Training loop
# -----------------------
def train_dpo(
    dataset,
    split,
    model_name,
    model_path,
    seed,
    passages,
    embed_model,
    reward_scheme,
    tokenizer_name=None,
    device="cuda",
    batch_size=2,
    epochs=1,
    lr=5e-6,
    max_length=512,
    beta=0.1,
    max_grad_norm=1.0
):
    """
    Offline DPO training: internally constructs DPO pairs from candidates,
    then fine-tunes the model using batch DPO loss.
    """
    tokenizer_name = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.train()

    # Construct DPO pairs internally
    print(f"[INFO] Constructing DPO pairs for {dataset}/{split}/{model_name}/seed_{seed}...")
    pairs = construct_dpo_pairs(dataset, split, model_name, seed, passages, embed_model, reward_scheme)
    if not pairs:
        print("[WARNING] No DPO pairs were constructed. Skipping training.")
        return

    dataset_obj = DPODataset(pairs)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, collate_fn=dpo_collate_fn)

    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()  

    for epoch in range(epochs):
        total_loss = 0.0
        for queries, chosens, rejecteds in tqdm(dataloader, desc=f"DPO Epoch {epoch+1}"):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                loss = compute_dpo_loss(model, tokenizer, queries, chosens, rejecteds, device, max_length, beta)
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(dataloader):.4f}")

    # Save final model
    save_paths = model_paths(dataset, split, model_name, "DPO", f"reward_config_{seed+1}", seed, stage="finetuned")
    save_model_and_tokenizer(model, tokenizer, save_paths)
    print(f"[INFO] Model saved at {save_paths}")









if __name__ == "__main__":


    # Datasets, splits, and models to run
    datasets = ["hotpotqa", "2wikimultihopqa"]
    splits = ["train"]
    models = {
        "7B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-7B-Instruct",
        "14B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-14B-Instruct"
    }
    seeds = [0]

    # Load embedding model once
    embed_model = SentenceTransformer(BGE_MODEL, device=DEVICE)

    for dataset in datasets:
        for split in splits:
            # Load passages aligned with queries
            passages_path = processed_dataset_paths(dataset, split)["passages"]
            passages = load_jsonl(passages_path)

            for model_name, model_path in models.items():
                for seed in seeds:
                    print(f"[INFO] Starting DPO training for {dataset}/{split}/{model_name}/seed_{seed}...")
                    train_dpo(
                        dataset=dataset,
                        split=split,
                        model_name=model_name,
                        model_path=model_path,
                        seed=seed,
                        passages=passages,
                        embed_model=embed_model,
                        reward_scheme=reward_scheme,
                        batch_size=1,
                        epochs=1,
                        device=DEVICE
                    )
