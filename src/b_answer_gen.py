"""



TO DO FOR DPO/PPO
- generate multiple answers per query (RLHF uses just the first answer of these)
- 








same quantisation (8-b), batch size, context/sequence length, device, temp, top-k/p, beam, repetition penalty, do sample/greedy,seed... for all models 



"""





"""




##### REWARDS

# TO ADD:
# - model says it's confident about an answer that contradicts NLI result 
# - two categories: 1) models output itself, 2) model's use of retrieved passages
# - CoT: token consistency between CoT and gold passages/answers, adversarial critique of CoT, 




GENERAL PRINCIPLES:
- generator-controlled first, then NLI grounding (single answer, multi-answer, or CoT answer)
- robustness & stability
- hallucination reduction 
- 



TO LOOK INTO:
- expose model to adversarial prompts or hallucination traps; reward resilience / cautious answering (safety-critical deployments)




RETRIEVAL STUFF:
- weight sentences by semantic importance in passsages
- vary retrieval slightly - rewards stable answers - robustness
- 


ANSWER GEN STUFF:


general:
- generate multiple answers/CoT + reward agreement/consitency between them (embedding-based similarity)
- embed generated answer + retrieved passages - reward cosine similarity (semantic closeness) - reduces hallucination
- generate answers -> paraphrase -> regenerate question -> check match ->   reward match (consistency cycle)


answer itself
- right/wrong/idk + confidence 
--- 
answer in relation to the retrieved passages (NLI)
- right/wrong/idk + confidence + answer/passages  entailment 
--- reward IDK when passages are weak or contradictory 
logic behind the answer (CoT)
- right/wrong/idk + confidence + answer/passages entailment + logic
--- consistency of logic (separate critic model)
--- contradiction of logic (separate critic model)
--- each reasoning step cites a passage(s) -> NLI to check consitency between reasoning and passage -> reward consistency 


"""





from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
import time
import random 
import os
from typing import List, Dict, Set, Callable, Counter
import faiss
import re
import string
from tqdm import tqdm
import json
from .x_utils import load_jsonl, save_jsonl, append_jsonl, load_model_8bit, dataset_results_paths, clean_text
from .z_configs import SEEDS, reward_configs, reward_scheme, gen_params
from .a_datasets_representations import extract_keywords
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer



"""

TO DO:
- get model to cite which passages it was most instructed by
- NLI on those passages only - score accordingly
- move results to the data folder (what's common practise with this?) # i think i've already done this 


"""


# -------------------------
# Configuration / Globals
# -------------------------

"""             MODULE RUNNING PARAMS       """

datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
splits = ["train"] # ["train", "dev"]

MAX_QUERIES = 10


models = {
    "7B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-7B-Instruct",
    "14B": "C:/Users/jeiwi/.cache/huggingface/hub/Qwen2.5-14B-Instruct"
}


ANSWER_PROMPT = "data/prompts/answer_prompt2.txt"










"""             MODEL STUFF             """


tokenizer_nli = AutoTokenizer.from_pretrained("roberta-large-mnli")
model_nli = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to("cuda")
model_nli.eval()


embed_model = SentenceTransformer("all-MiniLM-L6-v2")






"""         RETRIEVAL PARAMS          """

DEFAULT_HYBRID_ALPHA = 0.5 

MAX_RETRIEVED_PASSAGES = 5  # max passages to retrieve per query










# -------------------------
# Utility Functions
# -------------------------








def free_gpu_memory():
    """Clear CUDA cache and run garbage collection."""
    torch.cuda.empty_cache()
    gc.collect()






### dense representation


def load_faiss_index(path):
    """Load a FAISS index from ``path``."""
    index = faiss.read_index(path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {path}")
    return index



def faiss_search_topk(
        query_emb: np.ndarray, 
        index, 
        top_k: int = 50
        ):
    """

    Retrieve ``top_k`` most similar items from a FAISS index.
    
    """
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    norm = np.linalg.norm(query_emb) ###############################
    if not np.isfinite(norm) or norm == 0: ############################
        raise ValueError(
            f"Query embedding norm invalid ({norm}); check emb_model.encode output."
        )
    faiss.normalize_L2(query_emb)
    scores, idx = index.search(query_emb, top_k)
    return idx[0], scores[0]



### sparse representation



def jaccard_similarity(set1, set2):
    return len(set1 & set2) / max(1, len(set1 | set2))



### retrieval 


def retrieve_passages(
    query_vec: np.ndarray,
    query_keywords: Set[str],
    metadata: List[Dict],
    index,
    top_k: int,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    *,
    keyword_field: str | None = None,
    filter_fn: Callable[[int], bool] | None = None,
):
    """
    Retrieve top dense candidates and rerank them by Jaccard overlap.


    """

    if keyword_field is None and metadata:
        keyword_field = "keywords"

    query_vec = query_vec.reshape(1, -1)

    # Dense retrieval via FAISS
    faiss_idxs, faiss_scores = faiss_search_topk(query_vec, index, top_k=top_k)
    n_meta = len(metadata)
    valid_pairs = [
        (int(idx), float(score))
        for idx, score in zip(faiss_idxs, faiss_scores)
        if idx != -1 and 0 <= int(idx) < n_meta
    ]

    # Keep only valid dense candidates and optionally filter them
    candidate_idxs = [idx for idx, _ in valid_pairs]
    if filter_fn is not None:
        candidate_idxs = [i for i in candidate_idxs if filter_fn(i)]

    faiss_dict = {idx: score for idx, score in valid_pairs if idx in candidate_idxs}

    results: List[Dict[str, float]] = []
    for idx in candidate_idxs:
        sim_cos = faiss_dict[idx]
        if query_keywords:
            sim_jac = jaccard_similarity(
                query_keywords, set(metadata[idx].get(keyword_field, []))
            )
        else:
            sim_jac = 0.0
        sim_hybrid = alpha * sim_cos + (1.0 - alpha) * sim_jac
        results.append(
            {
                "idx": idx,
                "sim_cos": float(sim_cos,),
                "sim_jac": float(sim_jac),
                "sim_hybrid": float(sim_hybrid)
            }
        )

    results.sort(key=lambda x: x["sim_hybrid"], reverse=True)
    return results[:top_k]














def fill_prompt(template_path, query, passages, max_passages=MAX_RETRIEVED_PASSAGES):
    """
    Load a prompt template and fill in placeholders with passages and query.
    Expects placeholders: {passage_1} ... {passage_N}, {query}.
    """
    # Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Truncate or pad passages to fixed size
    filled_passages = passages[:max_passages] + [""] * (max_passages - len(passages)) 
    passage_map = {f"passage_{i+1}": filled_passages[i] for i in range(max_passages)}

    # Format template
    return template.format(query=query, **passage_map)




def parse_llm_json(output_str: str) -> dict:
    """
    Robust JSON parser for LLM outputs.
    Returns a dict with: answer, confidence, cited_passages.
    Also returns cleaned text and any parsing error for debugging.
    """
    cleaned = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", output_str, flags=re.DOTALL).strip()
    parsed = {"answer": "", "confidence": 0.0, "cited_passages": []}
    error_msg = None
    raw_json = None

    try:
        raw_json = json.loads(cleaned)
    except json.JSONDecodeError:
        # fallback: extract last {...} block
        blocks = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)
        if blocks:
            try:
                raw_json = json.loads(blocks[-1])
            except Exception as e:
                error_msg = f"Fallback JSON parse failed: {e}"
        else:
            error_msg = "No valid JSON object found"

    if raw_json:
        parsed["answer"] = str(raw_json.get("answer", "")).strip()
        parsed["confidence"] = float(raw_json.get("confidence", 0.0))
        cp = raw_json.get("cited_passages", [])
        if isinstance(cp, list):
            parsed["cited_passages"] = [
                int(v) - 1 if str(v).isdigit() and int(v) > 0 else int(v)
                for v in cp
                if str(v).isdigit()
            ]

    return {
        **parsed,
        "cleaned_output": cleaned,
        "raw_output": output_str,
        "error": error_msg
    }




def generate_answer(model, tokenizer, query, passages, gen_params, prompt_path=ANSWER_PROMPT):
    prompt = fill_prompt(prompt_path, query, passages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        **gen_params,
        return_dict_in_generate=True
    )
    sequence = outputs.sequences[0]
    raw_text = tokenizer.decode(sequence, skip_special_tokens=True)

    parsed = parse_llm_json(raw_text)

    answer_text = parsed.get("answer", "").strip()
    self_conf = float(parsed.get("confidence", 0.0))
    cited_idxs = parsed.get("cited_passages", [])
    is_idk = "i don't know" in answer_text.lower() or answer_text.lower().startswith("idk")

    return {
        "answer": answer_text,
        "self_conf": self_conf,
        "idk": is_idk,
        "cited_passages": cited_idxs,
        "raw_output": parsed.get("raw_output", raw_text),
        "cleaned_output": parsed.get("cleaned_output", ""),
        "error": parsed.get("error")
    }






def rerank_answers(
        candidates, 
        passages, 
        embed_model, 
        score_NLI_result
        ):
    """
    for DPO, and multi-answer RLHF

    Rerank multiple candidate answers based on grounding signals.
    
    Args:
        candidates (list of dict): Each must contain
            {
                "answer": str,
                "confidence": float,
                "cited_passages": list[int]
            }
        passages (list of str): Retrieved passages (1-based indices).
        embed_model: Embedding model with encode() -> np.array
        score_NLI_result: function(answer: str, passage: str) -> {"entailment": float, "neutral": float, "contradiction": float}
    
    Returns:
        dict: Best candidate (same schema as input).
    """
    
    scores = []
    for cand in candidates:
        answer = cand["answer"].strip()
        cited = cand.get("cited_passages", [])
        conf = cand.get("confidence", 0.0)

        # If answer is IDK, rank lower unless all fail
        if answer.lower() in ["i don't know", "idk"]:
            scores.append((-1.0, cand))  # always deprioritize IDK
            continue

        # Collect grounding evidence
        entail_scores, sim_scores = [], []
        for idx in cited:
            if 1 <= idx <= len(passages):
                passage = passages[idx - 1]

                # NLI scoring
                nli = score_NLI_result(answer, passage)
                entail_scores.append(nli.get("entailment", 0.0) - nli.get("contradiction", 0.0))

                # Semantic similarity
                ans_emb = embed_model.encode([answer])
                pas_emb = embed_model.encode([passage])
                sim = cosine_similarity(ans_emb, pas_emb)[0][0]
                sim_scores.append(sim)

        # Aggregate scores
        mean_entail = np.mean(entail_scores) if entail_scores else 0.0
        mean_sim = np.mean(sim_scores) if sim_scores else 0.0

        # Weighted rerank score
        rerank_score = (
            0.5 * mean_entail +
            0.3 * mean_sim +
            0.2 * conf
        )
        scores.append((rerank_score, cand))

    # Choose best
    best = max(scores, key=lambda x: x[0])[1]
    return best


















""" EVALUATION METRICS + REWARD FUNCTION"""



def NLI_check_answer_per_passage(answer, passages, reward_scheme):
    """
    Perform NLI/entailment check between the model-generated answer and each retrieved passage.
    Returns a list of numeric scores per passage based on reward_scheme['nli'].

    Args:
        answer (str): generated answer
        passages (list[str]): list of retrieved passages (premises)
        reward_scheme (dict): reward scheme containing "nli" mapping

    Returns:
        List[float]: numeric scores for each passage
    """
    nli_rewards = reward_scheme.get("nli", {"entailment": 0.5, "neutral": -0.2, "contradiction": -1.0})
    scores = []

    for passage in passages:
        inputs = tokenizer_nli(passage, answer, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            logits = model_nli(**inputs).logits
            pred_class = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

        mnli_labels = ["contradiction", "neutral", "entailment"]
        label = mnli_labels[pred_class]
        scores.append(nli_rewards[label])

    return scores




def normalise_for_em(s: str): # what is s in this case? the output???
    """
    Normalize an answer for EM/F1 calculation:
    - Lowercase
    - Remove punctuation
    - Remove articles ('a', 'an', 'the')
    - Collapse multiple whitespaces
    """
    s = s.lower()  # lowercase
    s = re.sub(r'\b(a|an|the)\b', ' ', s)  # remove articles
    s = s.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    s = ' '.join(s.split())  # remove extra whitespace
    return s



def compute_em(
        prediction, 
        gold_answer
        ):
    """
    EM is 1 if normalized prediction == normalized gold, else 0.
    """
    return int(normalise_for_em(prediction) == normalise_for_em(gold_answer))



def compute_f1(
        prediction, 
        gold_answer
        ):
    """
    Compute token-level F1 between prediction and gold after normalization.
    """
    pred_tokens = normalise_for_em(prediction).split() #### what's the pred + gold tokens????
    gold_tokens = normalise_for_em(gold_answer).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)





def compute_agreement_reward(
    model, 
    tokenizer, 
    query, 
    passages, 
    gen_params, 
    N=3):
    """
    
    
    
    """
    answers = []
    for _ in range(N):
        answer_info = generate_answer(model, tokenizer, query, passages, gen_params)
        answers.append(answer_info["answer"].strip().lower())

    counts = Counter(answers)
    maj_answer, maj_count = counts.most_common(1)[0]
    agreement_score = maj_count / N
    return agreement_score, maj_answer, answers




def compute_total_reward(
    answer_info,
    gold_answer,
    retrieved_ids,
    retrieved_texts,
    gold_passage_ids,
    reward_scheme,
    correctness="hybrid",
    scale_idk_penalty=True,
    use_confidence=True,
    multi_answers=None,       # list of other answers for agreement
    NLI_score_passages="retrieved"  # "retrieved" | "cited" | "None"
):
    reward = 0
    is_idk = answer_info.get("idk", False)

    # --- IDK reward / penalty ---
    retrieved_gold = sum(rid in gold_passage_ids for rid in retrieved_ids)
    retrieved_total = len(retrieved_ids) or 1
    gold_coverage = retrieved_gold / retrieved_total

    if is_idk:
        if retrieved_gold == 0:
            reward += reward_scheme["idk"]["safe_abstention"]
        else:
            penalty = reward_scheme["idk"]["penalty_ignore_gold"]
            reward += penalty * (gold_coverage if scale_idk_penalty else 1)
        return reward

    # --- Citation reward ---
    cited_idxs = answer_info.get("cited_passages", list(range(len(retrieved_texts))))
    cited_texts = [retrieved_texts[i] for i in cited_idxs if 0 <= i < len(retrieved_texts)]
    cited_ids = [retrieved_ids[i] for i in cited_idxs if 0 <= i < len(retrieved_ids)]

    cited_gold = sum(cid in gold_passage_ids for cid in cited_ids)
    cited_total = len(cited_ids) or 1
    cited_coverage = cited_gold / cited_total
    reward += cited_coverage * reward_scheme.get("citation", {}).get("reward", 0.3)

    # --- Answer correctness ---
    em_score = compute_em(answer_info["answer"], gold_answer)
    f1_score = compute_f1(answer_info["answer"], gold_answer)

    if correctness == "hybrid":
        correctness_score = 0.5 * em_score + 0.5 * f1_score
    elif correctness == "F1":
        correctness_score = f1_score
    else:
        correctness_score = em_score

    if correctness_score > 0:
        reward += reward_scheme["answer"]["correct_answer"] * correctness_score
    else:
        reward += reward_scheme["answer"]["incorrect_answer"]

    # --- NLI reward ---
    if NLI_score_passages == "retrieved":
        nli_targets = retrieved_texts
    elif NLI_score_passages == "cited":
        nli_targets = cited_texts
    else:  # "None"
        nli_targets = []

    if nli_targets:
        nli_scores = NLI_check_answer_per_passage(answer_info["answer"], nli_targets, reward_scheme)
        reward += float(np.mean(nli_scores))
        if use_confidence:
            conf = float(answer_info.get("confidence", 0.5))
            reward += conf * np.mean(nli_scores)

    # --- Multi-answer agreement ---
    if multi_answers is not None and len(multi_answers) > 1:
        counts = Counter([ans.lower() for ans in multi_answers])
        maj_count = counts.most_common(1)[0][1]
        agreement_score = maj_count / len(multi_answers)
        reward += reward_scheme.get("agreement", {}).get("reward", 0.2) * agreement_score

    return reward













""" RUN PIPELINE """


def run_rag_pipeline(
    dataset,
    split,
    model_name,
    model,
    tokenizer,
    queries: List[Dict],
    passage_metadata: List[Dict],
    passage_index,
    embed_model,
    gen_params,
    reward_scheme,
    SEEDS=[1, 2, 3],
    top_k: int = MAX_RETRIEVED_PASSAGES,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    scale_idk_penalty: bool = True,
    score_NLI_result: bool = True,
    use_confidence: bool = True,
    out_path: str = None,
    debug_path: str = None,
):
    """
    Full RAG workflow with multi-answer generation and agreement rewards.
    Supports RLHF, PPO, DPO seamlessly.
    """

    for qdata in tqdm(queries, desc="Processing queries", unit="query"):
        query_text = qdata["question"]
        query_id = qdata.get("question_id", None)
        gold_passage_ids = qdata.get("gold_passages", [])
        gold_answer = qdata.get("gold_answer", "")

        query_keywords = set(extract_keywords(
        query_text,
        include_ngrams=True,
        n_max=3,
        remove_numbers=True
    ))
        
        query_for_embedding = clean_text(query_text)
        # Embed query
        query_emb = embed_model.encode(
            [query_for_embedding],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        #query_keywords = set(extract_keywords(query_text))

        if passage_metadata and "keywords" not in passage_metadata[0]:
            print("[warn] passage_metadata has no 'keywords' field — Jaccard will be 0 unless you add them.")


        # Retrieve passages
        retrieved_info = retrieve_passages(
            query_vec=query_emb,
            query_keywords=query_keywords,
            metadata=passage_metadata,
            index=passage_index,
            top_k=top_k,
            alpha=alpha
        )
        retrieved_ids = [passage_metadata[r["idx"]]["passage_id"] for r in retrieved_info]
        retrieved_texts = [passage_metadata[r["idx"]]["text"] for r in retrieved_info]

        for seed in SEEDS:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            # --- Multi-answer generation ---
            candidates = []
            for _ in range(gen_params["num_return_sequences"]):
                cand = generate_answer(model, tokenizer, query_text, retrieved_texts, gen_params)
                cand["seed"] = seed
                candidates.append(cand)

            # Collect all answers for agreement reward
            multi_answers = [c["answer"].strip() for c in candidates]




            # Save multi-answer debug info
            if debug_path:
                debug_entry = {
                    "question_id": query_id,
                    "query": query_text,
                    "answers": multi_answers,
                    "cited_passages": [c["cited_passages"] for c in candidates],
                    "seeds": [c["seed"] for c in candidates],
                }
                append_jsonl(debug_path, debug_entry)






            # --- Rerank candidates ---
            final_answer = rerank_answers(candidates, retrieved_texts, embed_model, score_NLI_result)

            # Map cited indices to texts & IDs
            cited_idxs = final_answer.get("cited_passages", list(range(len(retrieved_texts))))
            cited_texts = [retrieved_texts[i] for i in cited_idxs if 0 <= i < len(retrieved_texts)]
            cited_ids = [retrieved_ids[i] for i in cited_idxs if 0 <= i < len(retrieved_ids)]

            # --- Reward computation (with per-passage NLI and agreement) ---
            reward = compute_total_reward(
                final_answer,
                gold_answer,
                retrieved_ids,
                retrieved_texts,
                gold_passage_ids,
                reward_scheme,
                correctness="hybrid",
                scale_idk_penalty=rc["scale_idk_penalty"],
                use_confidence=rc["use_confidence"],
                multi_answers=multi_answers,
                NLI_score_passages=rc["NLI_score_passages"]
)

            # --- Save final result per seed ---
            if out_path:
                result_entry = {
                    "question_id": query_id,
                    "query": query_text,
                    "gold_answer": gold_answer,
                    "answer": final_answer["answer"],
                    "idk": final_answer.get("idk", False),
                    "confidence": final_answer.get("self_conf", 0.0),
                    "retrieved_texts": retrieved_texts,
                    "retrieved_ids": retrieved_ids,
                    "cited_texts": cited_texts,
                    "cited_ids": cited_ids,
                    "reward": reward,
                    "seed": seed,
                    "multi_answers": multi_answers
                }
                append_jsonl(out_path, result_entry)













if __name__ == "__main__":

    for dataset in datasets:
        for split in splits:
            print(f"\n=== Processing {dataset} / {split} ===")

            # File paths
            queries_path = f"data/processed_datasets/{dataset}/{split}/questions.jsonl"
            metadata_path = f"data/processed_datasets/{dataset}/{split}/passages.jsonl"
            index_path = f"data/representations/datasets/{dataset}/{split}/{dataset}_faiss_passages.faiss"

            # Load queries and passages
            print(f"Loading queries from {queries_path}...")
            queries = list(load_jsonl(queries_path))

            queries = queries[:MAX_QUERIES] 

            print(f"Loading passage metadata from {metadata_path}...")
            passage_metadata = list(load_jsonl(metadata_path))

            print(f"Loading FAISS index from {index_path}...")
            passage_index = load_faiss_index(index_path)

            for model_name, model_path in models.items():
                print(f"\n--- Running {model_name} model ---")

                enable_offload = True if model_name == "14B" else False
                tokenizer, model = load_model_8bit(model_path, enable_fp32_cpu_offload=enable_offload)  

                for rc_idx, rc in enumerate(reward_configs):
                    print(f"\nReward config {rc_idx + 1}: {rc}")

                    out_path = f"data/results/{dataset}/{split}/{model_name}/reward_config_{rc_idx+1}/results.jsonl"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    debug_path = f"data/results/{dataset}/{split}/{model_name}/reward_config_{rc_idx+1}/debug.jsonl"
                    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                    
                    # Run pipeline (streaming per query)
                    run_rag_pipeline(
                        model=model,
                        tokenizer=tokenizer,
                        queries=queries,
                        passage_metadata=passage_metadata,
                        passage_index=passage_index,
                        embed_model=embed_model,
                        gen_params=gen_params,
                        out_path=out_path,
                        debug_path=debug_path,
                        top_k=MAX_RETRIEVED_PASSAGES,
                        alpha=DEFAULT_HYBRID_ALPHA,
                        scale_idk_penalty=rc["scale_idk_penalty"],
                        score_NLI_result=rc["score_NLI_result"],
                        use_confidence=rc["use_confidence"]
                    )

                    print(f"Results being appended to {out_path} as they are generated.")

                # Free model GPU memory
                del model  
                free_gpu_memory()

            # Explicitly free FAISS + passage metadata if memory is tight
            del passage_metadata
            del passage_index
            torch.cuda.empty_cache()

    print("\nPipeline complete for all datasets, splits, models, and reward configs!")
