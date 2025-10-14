"""


cd C:/Users/jeiwi/Data_projects/RAG_abstentation
./venv/Scripts/python.exe -m src.b_answer_gen



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
from .x_utils import load_jsonl, save_jsonl, append_jsonl, load_model_8bit, dataset_results_paths, clean_text, dataset_rep_paths
from .z_configs import SEEDS, reward_configs, reward_scheme, gen_params
from .a_datasets_representations import extract_keywords
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer




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
    """

    """
    torch.cuda.empty_cache()
    gc.collect()






### dense representation


def load_faiss_index(path):
    """

    """
    index = faiss.read_index(path)
    print(f"[FAISS] Loaded {index.ntotal} vectors from {path}")
    return index



def faiss_search_topk(
        query_emb, 
        index, 
        top_k = 50
        ):
    """

    
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
    query_vec,
    query_keywords,
    metadata,
    index,
    top_k,
    alpha = DEFAULT_HYBRID_ALPHA,
    *,
    keyword_field,
    get_keywords_fn,
    filter_fn,
):
    """


    """
    if keyword_field is None and metadata:
        keyword_field = "keywords"

    query_vec = query_vec.reshape(1, -1)
    faiss_idxs, faiss_scores = faiss_search_topk(query_vec, index, top_k=top_k)
    n_meta = len(metadata)

    valid_pairs = [
        (int(idx), float(score))
        for idx, score in zip(faiss_idxs, faiss_scores)
        if idx != -1 and 0 <= int(idx) < n_meta
    ]

    # Optionally filter candidates
    candidate_idxs = [idx for idx, _ in valid_pairs]
    if filter_fn is not None:
        candidate_idxs = [i for i in candidate_idxs if filter_fn(i)]

    faiss_dict = {idx: score for idx, score in valid_pairs if idx in candidate_idxs}
    results = []

    for idx in candidate_idxs:
        sim_cos = faiss_dict[idx]

        # Get keywords
        if get_keywords_fn is not None:
            passage_keywords = get_keywords_fn(idx)
        else:
            passage_keywords = set(metadata[idx].get(keyword_field, []))

        sim_jac = jaccard_similarity(query_keywords, passage_keywords) if query_keywords else 0.0
        sim_hybrid = alpha * sim_cos + (1.0 - alpha) * sim_jac

        results.append({
            "idx": idx,
            "sim_cos": float(sim_cos),
            "sim_jac": float(sim_jac),
            "sim_hybrid": float(sim_hybrid)
        })

    results.sort(key=lambda x: x["sim_hybrid"], reverse=True)
    return results[:top_k]















def fill_prompt(
        template_path, 
        query, 
        passages, 
        max_passages=MAX_RETRIEVED_PASSAGES
        ):
    """

    
    """
    # Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Truncate or pad passages to fixed size
    filled_passages = passages[:max_passages] + [""] * (max_passages - len(passages)) 
    passage_map = {f"passage_{i+1}": filled_passages[i] for i in range(max_passages)}

    # Format template
    return template.format(query=query, **passage_map)




def parse_llm_json(output_str):
    """

    
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




def generate_answer(
        model, 
        tokenizer, 
        query, 
        passages, 
        gen_params, 
        prompt_path=ANSWER_PROMPT
        ):
    prompt = fill_prompt(prompt_path, query, passages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        **gen_params,
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













def score_NLI_fn(
        answer, 
        passage, 
        reward_scheme
        ):
    """

    
    """
    inputs = tokenizer_nli(passage, answer, return_tensors="pt", truncation=True).to("cuda")
    with torch.no_grad():
        logits = model_nli(**inputs).logits
        pred_class = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()

    mnli_labels = ["contradiction", "neutral", "entailment"]
    label = mnli_labels[pred_class]

    nli_rewards = reward_scheme.get("nli", {"entailment": 0.5, "neutral": -0.2, "contradiction": -1.0})
    return {
        "entailment": nli_rewards.get("entailment", 0.5) if label == "entailment" else 0.0,
        "neutral": nli_rewards.get("neutral", -0.2) if label == "neutral" else 0.0,
        "contradiction": nli_rewards.get("contradiction", -1.0) if label == "contradiction" else 0.0,
    }


def rerank_answers(
    candidates,
    passages,
    embed_model,
    reward_scheme
):
    """

    
    """
    scores = []

    for cand in candidates:
        answer = cand["answer"].strip()
        cited = cand.get("cited_passages", [])
        conf = cand.get("confidence", 0.0)

        # Deprioritize IDK unless all fail
        if answer.lower() in ["i don't know", "idk"]:
            scores.append((-1.0, cand))
            continue

        entail_scores, sim_scores = [], []
        for idx in cited:
            if 0 <= idx < len(passages):
                passage = passages[idx]

                # NLI scoring
                nli = score_NLI_fn(answer, passage, reward_scheme)
                entail_scores.append(nli.get("entailment", 0.0) - nli.get("contradiction", 0.0))

                # Semantic similarity
                ans_emb = embed_model.encode([answer])
                pas_emb = embed_model.encode([passage])
                sim = cosine_similarity(ans_emb, pas_emb)[0][0]
                sim_scores.append(sim)

        mean_entail = np.mean(entail_scores) if entail_scores else 0.0
        mean_sim = np.mean(sim_scores) if sim_scores else 0.0

        rerank_score = 0.5 * mean_entail + 0.3 * mean_sim + 0.2 * conf
        scores.append((rerank_score, cand))

    best = max(scores, key=lambda x: x[0])[1]
    return best


















""" EVALUATION METRICS + REWARD FUNCTION"""



def NLI_check_answer_per_passage(
        answer, 
        passages, 
        reward_scheme
        ):
    """

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




def normalise_for_em(s): # what is s in this case? the output???
    """

    
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

    """
    return int(normalise_for_em(prediction) == normalise_for_em(gold_answer))



def compute_f1(
        prediction, 
        gold_answer
        ):
    """

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
    correctness = "hybrid",
    *,
    scale_idk_penalty = True,
    use_confidence = True,
    multi_answers,
    NLI_score_passages = "retrieved"  # "retrieved" | "cited" | "none" ################## is this updated according to the configs???
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
    if NLI_score_passages.lower() == "retrieved":
        nli_targets = retrieved_texts
    elif NLI_score_passages.lower() == "cited":
        nli_targets = cited_texts
    else:
        nli_targets = []

    if nli_targets:
        nli_scores = [score_NLI_fn(answer_info["answer"], p, reward_scheme) for p in nli_targets]
        entail_scores = [n.get("entailment", 0.0) - n.get("contradiction", 0.0) for n in nli_scores]
        mean_entail = np.mean(entail_scores)
        reward += mean_entail
        if use_confidence:
            conf = float(answer_info.get("confidence", 0.5))
            reward += conf * mean_entail

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
    queries,
    passage_metadata,
    passage_index,
    embed_model,
    gen_params,
    reward_scheme,
    SEEDS=[1, 2, 3], ########### isn't this defined in configs???
    top_k = MAX_RETRIEVED_PASSAGES,
    alpha = DEFAULT_HYBRID_ALPHA, 
    scale_idk_penalty = True,
    score_NLI_result = True,
    use_confidence = True,
    NLI_score_passages = "retrieved",
    *,
    out_path,
    debug_path,
    out_path_dpo, 
):
    """

    
    """
    sparse_path = dataset_rep_paths(dataset, split)["passages_sparse_jsonl"]
    with open(sparse_path, "r", encoding="utf-8") as f:
            sparse_passages = [json.loads(line) for line in f]

    # Map passage_id -> keywords
    id_to_keywords = {p["passage_id"]: set(p.get("keywords", [])) for p in sparse_passages}

    for seed in SEEDS:

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        reward_name = rc.get("name", f"reward_config_{rc_idx+1}")
        #paths = dataset_results_paths(dataset, split, model_name, reward_name, seed)

        for qdata in tqdm(queries, desc="Processing queries", unit="query"):
            query_text = qdata["question"]
            query_id = qdata.get("question_id", None)
            gold_passage_ids = qdata.get("gold_passages", [])
            gold_answer = qdata.get("gold_answer", "")

            query_keywords = set(extract_keywords(
            query_text,
        ))
            
            query_for_embedding = clean_text(query_text)
            # Embed query
            query_emb = embed_model.encode(
                [query_for_embedding],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0]



            # Define function for retrieval
            def get_keywords(pid):
                pid_str = passage_metadata[pid]["passage_id"]
                return id_to_keywords.get(pid_str, set())

            retrieved_info = retrieve_passages(
                query_vec=query_emb,
                query_keywords=query_keywords,
                metadata=passage_metadata,
                index=passage_index,
                top_k=top_k,
                alpha=alpha,
                get_keywords_fn=get_keywords
            )

            retrieved_ids = [passage_metadata[r["idx"]]["passage_id"] for r in retrieved_info]
            retrieved_texts = [passage_metadata[r["idx"]]["text"] for r in retrieved_info]


            # --- Multi-answer generation ---
            candidates = []
            for _ in range(gen_params["num_return_sequences"]):
                cand = generate_answer(model, tokenizer, query_text, retrieved_texts, gen_params)
                cand["seed"] = seed
                candidates.append(cand)

            # Collect all answers for agreement reward
            multi_answers = [c["answer"].strip() for c in candidates]

            # --- Rerank candidates ---
            final_answer = rerank_answers(candidates, retrieved_texts, embed_model, reward_scheme)

            preferences = [(final_answer, cand) for cand in candidates if cand != final_answer]


            # --- Compute rewards and save scaled version ---
            for c in candidates:
                # Compute raw reward
                raw_reward = compute_total_reward(
                    c,
                    gold_answer,
                    retrieved_ids,
                    retrieved_texts,
                    gold_passage_ids,
                    reward_scheme,
                    correctness="hybrid",
                    scale_idk_penalty=scale_idk_penalty,
                    use_confidence=use_confidence,
                    multi_answers=multi_answers,
                    NLI_score_passages=NLI_score_passages
                )

                # Save raw and scaled reward
                c["reward"] = raw_reward
                c["reward_scaled"] = np.tanh(raw_reward)  # keeps reward in [-1, 1]

            # --- Build DPO preference pairs ---
            preferences = [(final_answer, cand) for cand in candidates if cand != final_answer]

            if out_path_dpo:
                for chosen, rejected in preferences:
                    append_jsonl(out_path_dpo, {
                        "query": query_text,
                        "chosen": chosen["answer"],
                        "rejected": rejected["answer"],
                        "chosen_reward_scaled": chosen.get("reward_scaled", chosen.get("reward", 0.0)),
                        "rejected_reward_scaled": rejected.get("reward_scaled", rejected.get("reward", 0.0))
                    })

            # --- Save final result per seed ---
            if out_path:
                result_entry = {
                    "question_id": query_id,
                    "query": query_text,
                    "gold_answer": gold_answer,
                    "final_answer": final_answer["answer"],
                    "final_confidence": final_answer.get("self_conf", 0.0),
                    "final_idk": final_answer.get("idk", False),
                    "final_cited_passages": [retrieved_texts[i] for i in final_answer.get("cited_passages", []) if 0 <= i < len(retrieved_texts)],
                    "final_cited_ids": [retrieved_ids[i] for i in final_answer.get("cited_passages", []) if 0 <= i < len(retrieved_ids)],
                    "seed": seed,
                    "retrieved_texts": retrieved_texts,
                    "retrieved_ids": retrieved_ids,
                    "answers": [
                        {
                            "answer": c["answer"],
                            "idk": c.get("idk", False),
                            "confidence": c.get("self_conf", 0.0),
                            "cited_texts": [retrieved_texts[i] for i in c.get("cited_passages", []) if 0 <= i < len(retrieved_texts)],
                            "cited_ids": [retrieved_ids[i] for i in c.get("cited_passages", []) if 0 <= i < len(retrieved_ids)],
                            "reward": c["reward"],
                            "reward_scaled": c["reward_scaled"]
                        }
                        for c in candidates
                    ]

                }
                append_jsonl(out_path, result_entry)


                                # Save multi-answer debug info
                if debug_path:
                    debug_entry = {
                        "question_id": query_id,
                        "query": query_text,
                        "answers": result_entry["answers"],
                        "seed": seed
                    }
                    append_jsonl(debug_path, debug_entry)













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

                for config_name, config_list in reward_configs.items():  # RLHF, PPO, DPO
                    for rc_idx, rc in enumerate(config_list):
                        print(f"\nReward config {config_name} #{rc_idx + 1}: {rc}")

                        # Build proper folder structure: <RLHF|PPO|DPO>/<reward_config_#>
                        out_folder = f"data/results/{dataset}/{split}/{model_name}/{config_name}/reward_config_{rc_idx+1}"
                        os.makedirs(out_folder, exist_ok=True)

                        out_path = f"{out_folder}/results.jsonl"
                        debug_path = f"{out_folder}/debug.jsonl"


                        # Run pipeline (streaming per query)
                        run_rag_pipeline(
                            dataset=dataset,
                            split=split,
                            model_name=model_name,
                            model=model,
                            tokenizer=tokenizer,
                            queries=queries,
                            passage_metadata=passage_metadata,
                            passage_index=passage_index,
                            embed_model=embed_model,
                            gen_params=gen_params,
                            reward_scheme=reward_scheme,
                            out_path=out_path,
                            debug_path=debug_path,
                            top_k=MAX_RETRIEVED_PASSAGES,
                            alpha=DEFAULT_HYBRID_ALPHA,
                            scale_idk_penalty=rc["scale_idk_penalty"],
                            use_confidence=rc["use_confidence"],
                            NLI_score_passages=rc["NLI_score_passages"]  # just a string now
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
