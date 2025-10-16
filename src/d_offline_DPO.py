"""

preferences = [(final_answer, cand) for cand in candidates if cand != final_answer]

            if out_path_dpo:
                for chosen, rejected in preferences:
                    append_jsonl(out_path_dpo, {
                        "query": query_text,
                        "chosen": chosen["answer"],
                        "rejected": rejected["answer"],
                        
                        # Scaled reward for weighting
                        "chosen_reward_scaled": chosen.get("reward_scaled", chosen.get("reward", 0.0)),
                        "rejected_reward_scaled": rejected.get("reward_scaled", rejected.get("reward", 0.0)),

                        # Cited passages
                        "chosen_cited_ids": [
                            retrieved_ids[i] for i in chosen.get("cited_passages", []) 
                            if 0 <= i < len(retrieved_ids)
                        ],
                        "rejected_cited_ids": [
                            retrieved_ids[i] for i in rejected.get("cited_passages", []) 
                            if 0 <= i < len(retrieved_ids)
                        ],

                        # Model self-confidence
                        "chosen_confidence": chosen.get("self_conf", 0.5),
                        "rejected_confidence": rejected.get("self_conf", 0.5),

                        # IDK / abstention flag
                        "chosen_idk": chosen.get("idk", False),
                        "rejected_idk": rejected.get("idk", False),

                        # Retrieved passages (needed if NLI on retrieved)
                        "retrieved_ids": retrieved_ids
                    })



"""


from typing import List, Dict
from .x_utils import append_jsonl



def build_dpo_pairs(
    query_text: str,
    candidates: List[Dict],
    final_answer: Dict,
    retrieved_ids: List[str],
    out_path_dpo: str = None
):
    """
    Convert generation outputs into DPO (preferred vs rejected) pairs.

    Args:
        query_text: str, the question/query
        candidates: list of all candidate answer dicts
        final_answer: dict, the selected answer after reranking
        retrieved_ids: list of passage IDs corresponding to candidates
        out_path_dpo: path to save JSONL for DPO
    """
    preferences = [(final_answer, cand) for cand in candidates if cand != final_answer]

    if not out_path_dpo:
        return preferences

    for chosen, rejected in preferences:
        append_jsonl(out_path_dpo, {
            "query": query_text,
            "chosen": chosen["answer"],
            "rejected": rejected["answer"],

            # Scaled reward for weighting
            "chosen_reward_scaled": chosen.get("reward_scaled", chosen.get("reward", 0.0)),
            "rejected_reward_scaled": rejected.get("reward_scaled", rejected.get("reward", 0.0)),

            # Cited passages
            "chosen_cited_ids": [
                retrieved_ids[i] for i in chosen.get("cited_passages", []) 
                if 0 <= i < len(retrieved_ids)
            ],
            "rejected_cited_ids": [
                retrieved_ids[i] for i in rejected.get("cited_passages", []) 
                if 0 <= i < len(retrieved_ids)
            ],

            # Model self-confidence
            "chosen_confidence": chosen.get("self_conf", 0.5),
            "rejected_confidence": rejected.get("self_conf", 0.5),

            # IDK / abstention flag
            "chosen_idk": chosen.get("idk", False),
            "rejected_idk": rejected.get("idk", False),

            # Retrieved passages (optional, for NLI / PPO later)
            "retrieved_ids": retrieved_ids
        })
    return preferences



