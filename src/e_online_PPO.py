"""

GENERATE LOGITS WITH RESPONSES IN RELATION TO PROMPT  


"""


from typing import List, Dict
import torch


def generate_ppo_logits(
    model,
    tokenizer,
    query_text: str,
    candidates: List[Dict],
    gen_params: Dict,
    device: str = "cuda"
):
    """
    Generate PPO logits / log-probs for each candidate response.

    Args:
        model: causal LM
        tokenizer: tokenizer for LM
        query_text: str, prompt/query
        candidates: list of candidate dicts (answers)
        gen_params: generation parameters
        device: "cuda" or "cpu"

    Returns:
        List[Dict] with logits or log-probs per candidate
    """
    ppo_outputs = []

    for cand in candidates:
        # Construct full input: query + candidate answer
        prompt_plus_answer = f"{query_text}\n{cand['answer']}"
        inputs = tokenizer(prompt_plus_answer, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            logits = outputs.logits  # shape [1, seq_len, vocab_size]

        # Optionally compute log-probs for PPO
        # You could compute log-probs of candidate tokens w.r.t model
        cand_tokens = tokenizer(cand['answer'], return_tensors="pt").input_ids.to(device)
        log_probs = torch.log_softmax(logits[:, -cand_tokens.size(1):, :], dim=-1)
        token_log_probs = torch.gather(log_probs, 2, cand_tokens.unsqueeze(-1)).squeeze(-1)
        sum_log_prob = token_log_probs.sum().item()

        ppo_outputs.append({
            "query": query_text,
            "answer": cand["answer"],
            "logits": logits.cpu().numpy(),
            "sum_log_prob": sum_log_prob,
            "num_tokens": cand_tokens.size(1)
        })

    return ppo_outputs
