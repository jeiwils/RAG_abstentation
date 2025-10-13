"""

things that need to be standardised and used more than once across different modules:
- embedding model 



"""

import os
import torch 


BGE_MODEL = os.environ.get("BGE_MODEL", "all-MiniLM-L6-v2")
SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RLHF_MODEL = "EleutherAI/gpt-neo-1.3B"




SEEDS = [1, 2, 3] 



"""                 """









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
    "max_new_tokens": 254, # "max_length": 512, # MAX_NEW_TOKENS just for generation tokens, while max_length includes prompt length

    "temperature": 0.7, ############# ADJUST ACCORDING TO THE MODEL I USE

    "top_p": 0.95, # cumulative probability threshold # I NEED TO LOOK INTO NUCLEUS SAMPLING
    "top_k": 40, # take only 50 highest-prob tokens for sampling

    "do_sample": True, # samples from distribution, instead of taking just the most likely token #### can be used to capture model uncertainty???

    "num_return_sequences": 3, # sample multiple answers 
    "repetition_penalty": 1.1 # penalise repeating same token sequences
}
