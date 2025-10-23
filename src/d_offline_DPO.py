"""
Offline DPO training module.

Loads RAG-generated candidate answers, constructs preferred/rejected pairs,
and fine-tunes the model using offline DPO.




"""


import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from .x_utils import load_jsonl, dataset_results_paths, model_paths, processed_dataset_paths
from .b_answer_gen import rerank_answers
from z_configs import BGE_MODEL, DEVICE, reward_scheme
from sentence_transformers import SentenceTransformer



class DPODataset(Dataset): # why do I need to define this as a class??
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







def compute_dpo_loss(
    model,
    tokenizer,
    queries,
    chosens,
    rejecteds,
    device="cuda",
    max_length=512,
    beta=0.1,
):
    """
    Efficient batch DPO loss computation with proper padding handling.

    Each sample = (query, chosen, rejected)
    Returns scalar DPO loss for the batch.
    """

    # Ensure tokenizer has a pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id

    # Tokenize in batch
    chosen_batch = tokenizer(
        [q + "\n" + c for q, c in zip(queries, chosens)],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    rejected_batch = tokenizer(
        [q + "\n" + r for q, r in zip(queries, rejecteds)],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    # CrossEntropyLoss with ignore_index for padding
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_id)

    # Compute model outputs
    with autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
        chosen_out = model(**chosen_batch)
        rejected_out = model(**rejected_batch)

    # Shift logits and labels for causal LM prediction
    shift_logits_c = chosen_out.logits[..., :-1, :].contiguous()
    shift_labels_c = chosen_batch.input_ids[..., 1:].contiguous()
    shift_logits_r = rejected_out.logits[..., :-1, :].contiguous()
    shift_labels_r = rejected_batch.input_ids[..., 1:].contiguous()

    # Per-token loss
    chosen_loss = loss_fct(shift_logits_c.view(-1, shift_logits_c.size(-1)), shift_labels_c.view(-1))
    rejected_loss = loss_fct(shift_logits_r.view(-1, shift_logits_r.size(-1)), shift_labels_r.view(-1))

    # Reshape back to sequence shape
    chosen_loss = chosen_loss.view(shift_labels_c.size())
    rejected_loss = rejected_loss.view(shift_labels_r.size())

    # Mask padding tokens
    chosen_mask = (shift_labels_c != pad_id).float()
    rejected_mask = (shift_labels_r != pad_id).float()

    # Average log-probs per sequence
    chosen_lp = -(chosen_loss * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
    rejected_lp = -(rejected_loss * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)

    # DPO loss
    diff = chosen_lp - rejected_lp
    loss = -torch.log(torch.sigmoid(beta * diff) + 1e-8).mean()

    return loss







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
    lr=5e-6,
    batch_size=2,
    epochs=1,
    device="cuda"
):
    """
    Train a model using offline DPO.
    """
    tokenizer_name = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.train()

    # Load DPO pairs (reranking happens here)
    print(f"[INFO] Loading DPO pairs for {dataset}/{split}/{model_name}/DPO/{seed}")
    pairs = construct_dpo_pairs(dataset, split, model_name, seed, passages, embed_model, reward_scheme)
    dataset_obj = DPODataset(pairs)
    dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}]")
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="DPO training"):
            queries, chosens, rejecteds = batch
            optimizer.zero_grad()

            # Compute DPO loss for the batch
            loss = compute_dpo_loss(model, tokenizer, queries, chosens, rejecteds, device=device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    # Save model
    save_paths = model_paths(
        dataset=dataset,
        split=split,
        model_name=model_name,
        paradigm="DPO",
        reward_scheme=f"reward_config_{seed+1}",
        seed=seed,
        stage="finetuned"
    )

    model.save_pretrained(save_paths["model"])
    tokenizer.save_pretrained(save_paths["tokenizer"])
    print(f"[INFO] DPO model saved to {save_paths['model']}")









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
                        lr=5e-6,
                        batch_size=1,
                        epochs=1,
                        device=DEVICE
                    )
