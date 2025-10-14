"""

things that need to be standardised and used more than once across different modules:
- embedding model 



"""

import os
import torch 
from .x2_utils2 import pid_plus_title


BGE_MODEL = os.environ.get("BGE_MODEL", "all-MiniLM-L6-v2")
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RLHF_MODEL = "EleutherAI/gpt-neo-1.3B"




SEEDS = [1, 2, 3] 



"""                 """






FIELD_MAPS = {
    "hotpotqa": {
        "get_qid": lambda ex: ex["_id"],
        "get_question_text": lambda ex: ex["question"],
        "get_answer_text": lambda ex: ex.get("answer", ""),
        "iter_passages": lambda ex: [
            (pid_plus_title(ex["_id"], title, i), title, sent)
            for title, sents in ex["context"]
            for i, sent in enumerate(sents)
        ],
        "gold_passage_ids": lambda ex: [
            pid_plus_title(ex["_id"], title, idx)
            for title, idx in ex.get("supporting_facts", [])
        ],
    },
    "2wikimultihopqa": {  # similar to hotpotqa
        "get_qid": lambda ex: ex["_id"],
        "get_question_text": lambda ex: ex["question"],
        "get_answer_text": lambda ex: ex.get("answer", ""),
        "iter_passages": lambda ex: [
            (pid_plus_title(ex["_id"], title, i), title, sent)
            for title, sents in ex["context"]
            for i, sent in enumerate(sents)
        ],
        "gold_passage_ids": lambda ex: [
            pid_plus_title(ex["_id"], title, idx)
            for title, idx in ex.get("supporting_facts", [])
        ],
    },
    "musique": {
        "get_qid": lambda ex: ex["id"],
        "get_question_text": lambda ex: ex.get("question", ""),
        "get_answer_text": lambda ex: ex.get("answer", ""),
        "iter_passages": lambda ex: [
            (
                f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}",
                p.get("title", ""),
                p.get("paragraph_text", ""),
            )
            for i, p in enumerate(ex.get("paragraphs", []))
        ],
        "gold_passage_ids": lambda ex: [
            f"{ex['id']}_sent{p.get('idx') if p.get('idx') is not None else i}"
            for i, p in enumerate(ex.get("paragraphs", []))
            if p.get("is_supporting")
        ],
    }
}






"""              REWARD STUFF           """



# Practical subset of reward configs for experimentation
reward_configs = {
    "RLHF": [
        # Full shaping: NLI on cited passages, confidence shaping, IDK penalty
        {"scale_idk_penalty": True,  "NLI_score_passages": "cited",   "use_confidence": True},

        # Ablation: ignore confidence shaping
        {"scale_idk_penalty": True,  "NLI_score_passages": "cited",   "use_confidence": False},

        # Ablation: ignore NLI entirely (only correctness & citation)
        {"scale_idk_penalty": True,  "NLI_score_passages": "none",    "use_confidence": True},
    ],
    "PPO": [
        # Minimal shaping: NLI on cited passages, no IDK scaling
        {"scale_idk_penalty": False, "NLI_score_passages": "cited",   "use_confidence": True},

        # Baseline: no shaping at all (for comparison)
        {"scale_idk_penalty": False, "NLI_score_passages": "none",    "use_confidence": False},
    ],
    "DPO": [
        # Full shaping + agreement: NLI on retrieved passages, confidence included
        {"scale_idk_penalty": True,  "NLI_score_passages": "retrieved","use_confidence": True},

        # Variant: ignore IDK scaling but still use NLI on cited passages
        {"scale_idk_penalty": False, "NLI_score_passages": "cited",   "use_confidence": True},
    ]
}





reward_scheme = {
    "idk": {
        "safe_abstention": 0.2,
        "penalty_ignore_gold": -1.0,
    },
    "answer": {
        "correct_answer": 1.0,
        "incorrect_answer": -1.0,
    },
    "nli": {
        "entailment": +0.5,
        "neutral": -0.2,
        "contradiction": -1.0,
    },
    "citation": {
        "reward": 0.3,
    },
    "agreement": {
        "reward": 0.2,   
    }
}



"""             GENERATION STUFF            """


# Generation parameters
gen_params = {
    "max_new_tokens": 254,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "do_sample": True,
    "num_return_sequences": 3,
    "repetition_penalty": 1.1,
    "output_scores": True,            # return token logits
    "return_dict_in_generate": True   # return a dict instead of raw tensor
}
