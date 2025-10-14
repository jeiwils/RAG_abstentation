"""


types (all affect retrieval and therefore NLI/IDK stuff):
- hybrid
- dense
- BM25


cd C:/Users/jeiwi/Data_projects/RAG_abstentation
./venv/Scripts/python.exe -m src.a_datasets_representations



"""

from collections import Counter
import math
import numpy as np
import re
import json
import os
import faiss
from .x_utils import (
    append_jsonl,
    clean_text,
    compute_resume_sets,
    load_jsonl,
    normalise_text,
    processed_dataset_paths,
    dataset_rep_paths,
    load_models,
    SPACY_MODEL,
    nlp,
)
from .z_configs import FIELD_MAPS



"""             GENERAL DATASET PROCESSING FUNCTION                """



def process_dataset( 
    dataset,
    split,
    file_path,
    field_map,
    max_examples,
    resume = True,
):
    """
    just provide the field map and file path of the dataset

    field map needs to have the following keys:
    - essential: get_qid, get_question_text, iter_passages, 
    - optional: get_answer_text, gold_passage_ids

    """

    # ---- Load raw examples 
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for i, line in enumerate(f):
                if isinstance(max_examples, int) and i >= max_examples: 
                    break # reads entire file if max_examples is None 
                examples.append(json.loads(line))
        else:
            examples = json.load(f)
            if isinstance(max_examples, int):
                examples = examples[:max_examples]

    # ---- output paths
    output_paths = processed_dataset_paths(dataset, split) 

    qa_output_path = str(output_paths["questions"])
    passages_output_path = str(output_paths["passages"])


    # ---- field map functions 
    get_qid = field_map["get_qid"]
    get_question_text = field_map["get_question_text"]
    get_answer_text = field_map.get("get_answer_text", lambda ex: "")
    gold_ids_fn = field_map.get("gold_passage_ids", lambda ex: [])  

    # passage info (id, title, text)
    iter_passages_fn = field_map["iter_passages"] 


    # ---- Determine resume state 
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_output_path,
        items=examples,
        get_id=lambda ex, i: get_qid(ex),
        phase_label=f"{dataset} {split} questions",
        id_field="question_id",
    )

    def iter_pids(): 
        for ex in examples:
            for pid, title, text in iter_passages_fn(ex): 
                yield pid

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_output_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"{dataset} {split} passages",
        id_field="passage_id",
    )

    # ---- Write processed files 
    for ex in examples:
        qid = get_qid(ex)

        if qid not in done_qids:
            gold_ids, seen = [], set() 
            for gold_pid in gold_ids_fn(ex):
                if gold_pid not in seen:
                    gold_ids.append(gold_pid)
                    seen.add(gold_pid)
            append_jsonl(
                qa_output_path,
                {
                    "question_id": qid,
                    "dataset": dataset,
                    "split": split,
                    "question": clean_text(get_question_text(ex)),
                    "gold_answer": clean_text(get_answer_text(ex)),
                    "gold_passages": gold_ids,
                },
            )

        for pid, title, text in iter_passages_fn(ex):
            if pid in done_pids:
                continue
            append_jsonl(
                passages_output_path,
                {
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(text),
                },
            )









"""         DENSE: make embeddings + build faiss database       """  





def build_and_save_faiss_index( 
    embeddings,
    dataset_name,
    index_type,
    new_vectors = None, 
    output_dir = ".",
):
    """

    new_vectors = for additing to an existing index 

    ##################################################################

    """

    # ---- validate inputs 
    if not index_type or index_type not in {"passages"}:
        raise ValueError(
            "index_type must be provided and set to 'passages'"
        )

    # ---- normalise and format embeddings - initialise vectors 
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    if new_vectors is not None:
        new_vectors = np.ascontiguousarray(new_vectors, dtype=np.float32)
        faiss.normalize_L2(new_vectors)

    # ---- create or update FAISS index
    faiss_path = os.path.join(output_dir, f"{dataset_name}_faiss_{index_type}.faiss")

    if new_vectors is not None and os.path.exists(faiss_path): # adds to existing index
        index = faiss.read_index(faiss_path)
        assert new_vectors.shape[1] == index.d, (
            f"Dimension mismatch: new_vectors.shape[1]={new_vectors.shape[1]}, index.d={index.d}"
        )
        index.add(new_vectors)
    else: # makes new index 
        index = faiss.IndexFlatIP(embeddings.shape[1])
        assert embeddings.shape[1] == index.d, (
            f"Dimension mismatch: embeddings.shape[1]={embeddings.shape[1]}, index.d={index.d}"
        )
        index.add(embeddings)

    faiss.write_index(index, faiss_path)

    print(f"[FAISS] Saved {index_type} index to {faiss_path} with {index.ntotal} vectors.")













"""         SPARSE       """ 


# gets noun chunks - makes ngrams according to named entities - filters according to tf-idf 


ALIAS = {
    "us": "united_states", "u_s": "united_states", "u_s_a": "united_states",
    "united_states_of_america": "united_states", "the_united_states": "united_states",
    "u_s_navy": "united_states_navy", "u_s_air_force": "united_states_air_force",
    "u_s_army": "united_states_army", "u_s_marine_corps": "united_states_marine_corps",
    "uk": "united_kingdom", "u_k": "united_kingdom", "great_britain": "united_kingdom",
    "un": "united_nations", "u_n": "united_nations",
    "what_year": None, "the_years": None, "year": None,
}

NOISE_PATTERNS = [
    #re.compile(r"^\d{4}$"),          # years
    re.compile(r"^\d+$"),            # pure numbers
    re.compile(r"^(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)$")
]

KEEP_ENTS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT", "WORK_OF_ART", "EVENT", "LAW", "DATE"} 




def compute_tfidf_stats(all_passages, min_df=2):
    """
    Compute DF counts and IDF values across passages.
    Returns: dict {term: idf}
    """
    df_counter = Counter()
    N = len(all_passages)
    for passage in all_passages:
        kws = set(extract_keywords(passage["text"]))
        df_counter.update(kws)

    idf = {kw: math.log(N / df) for kw, df in df_counter.items() if df >= min_df}
    return idf



def filter_keyword(
        kw, 
        remove_numbers=True
        ):
    """
    canonicalises keywords - replaces with alias
    removes nosie patterns 

    returns the same word if there's no alias or noise 
    """
    kw = ALIAS.get(kw, kw)
    if kw is None:
        return None
    if remove_numbers:
        for pattern in NOISE_PATTERNS:
            if pattern.match(kw):
                return None
    return kw


def extract_keywords(
        text, 
        remove_numbers=False, 
        ngram_length=3,
        idf_dict=None,
        min_idf=0.5
):
    if not text:
        return []

    doc = nlp(text)
    out = set()

    ner_spans = [ent for ent in doc.ents if ent.label_ in KEEP_ENTS and ent.text.strip()]

    for chunk in doc.noun_chunks:
        if not any(chunk.start < ent.end and chunk.end > ent.start for ent in ner_spans):
            continue

        tokens = [t.lemma_.lower() for t in chunk if t.is_alpha and not t.is_stop]
        n_tokens = len(tokens)
        if n_tokens == 0:
            continue

        for n in range(1, min(ngram_length, n_tokens) + 1):
            for i in range(n_tokens - n + 1):
                gram_tokens = tokens[i:i+n]
                phrase = "_".join(gram_tokens)
                canon = filter_keyword(phrase, remove_numbers=remove_numbers)
                if canon:
                    if idf_dict and canon in idf_dict:
                        if idf_dict[canon] >= min_idf:  # keep only high-IDF terms
                            out.add(canon)
                    else:
                        out.add(canon)

    return sorted(out)






"""         BATCH PROCESSING       """






def process_batch(
        batch_texts, 
        batch_entries, 
        bge_model, 
        embeddings_all, 
        vec_offset, 
        sparse_dir,
        idf_dict=None,
        ):
    """
    
    ##################################################################

    """
    
    if not batch_texts:
        return embeddings_all, vec_offset

    # Dense embeddings
    batch_embs = bge_model.encode(
        batch_texts,
        normalize_embeddings=True,
        batch_size=len(batch_texts),
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")
    embeddings_all = np.vstack([embeddings_all, batch_embs])

    # Sparse keyword extraction
    for i, ent in enumerate(batch_entries):
        ent["keywords"] = extract_keywords(
            batch_texts[i],
            idf_dict=idf_dict,
            remove_numbers=True
        )
        ent["vec_id"] = vec_offset
        vec_offset += 1
        sparse_dir.write(json.dumps(ent, ensure_ascii=False) + "\n")

    return embeddings_all, vec_offset







"""         MAIN         """





if __name__ == "__main__":
    RESUME = True
    MAX_EXAMPLES = 100
    PHASES = ["datasets", "representations"]
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train", "dev"]
    BATCH_SIZE = 128  

    nlp, bge_model = load_models()



    # ---------------------- File path helper ----------------------
    def get_file_path(dataset, split):
        """
        
        ##################################################################

        
        """
        if dataset == "hotpotqa":
            return (
                "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json" if split == "train" 
                else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
            )
        elif dataset == "2wikimultihopqa":
            return f"data/raw_datasets/2wikimultihopqa/{split}.json"
        elif dataset == "musique":
            return f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    # ---------------------- Main loop ----------------------
    for phase in PHASES:

        if phase == "datasets":
            print("\n*  *   * DATASET PROCESSING *  *   *\n")
            for dataset in DATASETS:
                for split in SPLITS:
                    print(f"\n=== DATASET: {dataset} ({split}) ===")
                    field_map = FIELD_MAPS[dataset]
                    file_path = get_file_path(dataset, split)

                    process_dataset(
                        dataset,
                        split,
                        file_path,
                        field_map,
                        MAX_EXAMPLES,
                        RESUME,
                    )

        elif phase == "representations":
            print("\n*  *   * REPRESENTATIONS PROCESSING *  *   *\n")
            print("[Phase] Building passage embeddings & FAISS indexes...")

            for dataset in DATASETS:
                for split in SPLITS:
                    print(f"\n=== DATASET: {dataset} ({split}) ===")

                    proc_paths = processed_dataset_paths(dataset, split)
                    rep_paths = dataset_rep_paths(dataset, split)

                    passages_jsonl_src = proc_paths["passages"]
                    passages_jsonl = rep_paths["passages_jsonl"]
                    passages_npy = rep_paths["passages_emb"]
                    sparse_jsonl = rep_paths["passages_sparse_jsonl"]
                    dataset_dir = os.path.dirname(passages_jsonl)

                    all_passages = list(load_jsonl(passages_jsonl_src))
                    # Compute TF-IDF scores for filtering
                    idf_dict = compute_tfidf_stats(all_passages, min_df=2)
                    done_ids, _ = compute_resume_sets(
                        resume=RESUME,
                        out_path=passages_jsonl,
                        items=all_passages,
                        get_id=lambda x, i: x["passage_id"],
                        phase_label="passage embeddings",
                        required_field="vec_id",
                    )
                    vec_offset = len(done_ids)

                    sparse_mode = "a" if os.path.exists(sparse_jsonl) else "w"
                    with open(sparse_jsonl, sparse_mode, encoding="utf-8") as f_sparse:

                        if os.path.exists(passages_npy):
                            embeddings_all = np.load(passages_npy).astype("float32")
                        else:
                            embeddings_all = np.empty(
                                (0, bge_model.get_sentence_embedding_dimension()), dtype="float32"
                            )

                        # ---------------------- Batch processing ----------------------
                        batch_texts, batch_entries = [], []
                        for entry in all_passages:
                            pid = entry["passage_id"]
                            if pid in done_ids:
                                continue
                            batch_texts.append(entry["text"])
                            batch_entries.append(entry)

                            if len(batch_texts) >= BATCH_SIZE:
                                embeddings_all, vec_offset = process_batch(
                                    batch_texts, batch_entries, bge_model, embeddings_all, vec_offset, f_sparse, idf_dict=idf_dict
                                )
                                batch_texts, batch_entries = [], []

                        # Process any remaining batch
                        embeddings_all, vec_offset = process_batch(
                            batch_texts, batch_entries, bge_model, embeddings_all, vec_offset, f_sparse, idf_dict=idf_dict
                        )

                    np.save(passages_npy, embeddings_all)
                    print(f"[Done] Saved embeddings ({embeddings_all.shape[0]}) and sparse JSONL → {sparse_jsonl}")

                    # Build FAISS index
                    build_and_save_faiss_index(
                        embeddings=embeddings_all,
                        dataset_name=dataset,
                        index_type="passages",
                        output_dir=dataset_dir,
                    )



        else:
            print(f"[warn] Unknown phase: {phase}")

    print("[All Done] All datasets processed.")

