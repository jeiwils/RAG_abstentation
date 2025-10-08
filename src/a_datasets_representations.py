"""


types (all affect retrieval and therefore NLI/IDK stuff):
- hybrid
- dense
- BM25


cd C:/Users/jeiwi/Data_projects/RAG_abstentation
./venv/Scripts/python.exe -m src.a_datasets_representations



"""

from __future__ import annotations


import spacy
import re
from itertools import islice

from pathlib import Path
import re
import spacy
import json
from typing import Dict, Iterable, List, Set
import numpy as np
import os
import faiss
from .x_utils import (
    append_jsonl,
    clean_text,
    compute_resume_sets,
    load_jsonl,
    pid_plus_title,
    normalise_text
)
import torch 
from sentence_transformers import SentenceTransformer






# ---------------------------------------------------------------------------
# Generic dataset processing




def processed_dataset_paths(dataset: str, split: str) -> Dict[str, Path]:
    """
    Return standard paths for processed dataset files.

    Creates ``data/processed_datasets/{dataset}/{split}/`` if necessary and
    returns paths for ``questions.jsonl`` and ``passages.jsonl``.

    """
    base = Path(f"data/processed_datasets/{dataset}/{split}")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "questions": base / "questions.jsonl",
        "passages": base / "passages.jsonl",
    }



def process_dataset(
    *, ############ ???????
    dataset,
    split,
    file_path,
    field_map,
    max_examples,
    overwrite = False,
    resume,
):
    """Process ``file_path`` using ``field_map``.

    Parameters
    ----------
    dataset:
        Name of the dataset being processed.
    split:
        Dataset split (``train``, ``dev`` ...).
    file_path:
        Path to the raw dataset file.  JSON or JSONL files are supported.
    field_map:
        Mapping of callables that extract fields from each example.  Required
        keys are ``get_id``, ``get_question``, ``get_answer``,
        ``iter_passages`` and ``gold_passage_ids``.  The callables operate on a
        single example and either return a value or an iterable of values.
    max_examples:
        Optional limit for the number of examples processed.
    overwrite:
        Unused but kept for backward compatibility.
    resume:
        Whether to resume from existing processed files.
    """

    # ---- Load raw examples -------------------------------------------------
    examples: List[Dict] ######## what's this???? 
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            examples = []
            for i, line in enumerate(f):
                if isinstance(max_examples, int) and i >= max_examples: # like this, allows for None == no limit
                    break
                examples.append(json.loads(line))
        else:
            examples = json.load(f)
            if isinstance(max_examples, int):
                examples = examples[:max_examples]

    paths = processed_dataset_paths(dataset, split)
    qa_path = str(paths["questions"])
    passages_path = str(paths["passages"])

    get_id = field_map["get_id"]
    get_question = field_map["get_question"]
    get_answer = field_map.get("get_answer", lambda ex: "")
    iter_passages_fn = field_map["iter_passages"] # what does fn stand for here? 
    gold_ids_fn = field_map.get("gold_passage_ids", lambda ex: []) # what does fn stand for here? 

    # ---- Determine resume state --------------------------------------------
    done_qids, _ = compute_resume_sets(
        resume=resume,
        out_path=qa_path,
        items=examples,
        get_id=lambda ex, i: get_id(ex),
        phase_label=f"{dataset} {split} questions",
        id_field="question_id",
    )

    def iter_pids(): ############ why is this defined here? 
        for ex in examples:
            for pid, _title, _text in iter_passages_fn(ex):
                yield pid

    done_pids, _ = compute_resume_sets(
        resume=resume,
        out_path=passages_path,
        items=iter_pids(),
        get_id=lambda pid, i: pid,
        phase_label=f"{dataset} {split} passages",
        id_field="passage_id",
    )

    # ---- Write processed files ---------------------------------------------
    for ex in examples:
        qid = get_id(ex)


        if qid not in done_qids:
            gold_ids, seen = [], set()
            for pid in gold_ids_fn(ex):
                if pid not in seen:
                    gold_ids.append(pid)
                    seen.add(pid)
            append_jsonl(
                qa_path,
                {
                    "question_id": qid,
                    "dataset": dataset,
                    "split": split,
                    "question": clean_text(get_question(ex)),
                    "gold_answer": clean_text(get_answer(ex)),
                    "gold_passages": gold_ids,
                },
            )

        for pid, title, text in iter_passages_fn(ex):
            if pid in done_pids:
                continue
            append_jsonl(
                passages_path,
                {
                    "passage_id": pid,
                    "title": title,
                    "text": clean_text(text),
                },
            )






















"""         DENSE: make embeddings + build faiss database       """  



def embed_and_save( ########### CHECK THIS - LOOKS A BIT RIDICULOUS 
    input_jsonl,
    output_npy,
    output_jsonl,
    model,
    text_key,
    *,
    id_field="passage_id",
    done_ids: Set[str] | None = None,
    output_jsonl_input: str | None = None,
):
    """
    Embed texts from ``input_jsonl`` and save results.

    """

    if not text_key:
        raise ValueError("You must provide a valid text_key (e.g., 'text' or 'question').")


    if output_jsonl_input is None:
        output_jsonl_input = input_jsonl

    by_id = {}
    if output_jsonl_input != input_jsonl:
        with open(output_jsonl_input, "rt", encoding="utf-8") as f_clean:
            for line in f_clean:
                entry = json.loads(line)
                by_id[entry[id_field]] = entry

    data, texts = [], []

    for entry in load_jsonl(input_jsonl):
        entry_id = entry.get(id_field)
        if done_ids and entry_id in done_ids:
            continue
        texts.append(entry[text_key])
        if by_id:
            if entry_id not in by_id:
                raise KeyError(
                    f"{id_field} {entry_id} from {input_jsonl} not found in {output_jsonl_input}"
                )
            data.append(by_id[entry_id])
        else:
            data.append(entry)

    existing_embs = None
    vec_offset = 0
    if os.path.exists(output_npy):
        existing_embs = np.load(output_npy).astype("float32") 
        vec_offset = existing_embs.shape[0]
        if os.path.exists(output_jsonl):
            with open(output_jsonl, "rt", encoding="utf-8") as f_old:
                idx = -1
                for idx, line in enumerate(f_old):
                    if json.loads(line).get("vec_id") != idx:
                        raise AssertionError(
                            f"vec_id mismatch at line {idx} in {output_jsonl}"
                        )
                if vec_offset != idx + 1:
                    raise AssertionError(
                        f"Embedding count {vec_offset} does not match JSONL entries {idx + 1}"
                    )

    if not data:
        if existing_embs is not None:
            embs_all = existing_embs
        else:
            embs_all = np.empty(
                (0, model.get_sentence_embedding_dimension()), dtype="float32"
            )
        print(f"[Embeddings] No new items for {input_jsonl}; skipping.")
        return embs_all, np.empty(
            (0, embs_all.shape[1] if embs_all.size else 0), dtype="float32"
        )

    new_embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=128,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    for i, entry in enumerate(data):
        entry["vec_id"] = i + vec_offset

    dir_path = os.path.dirname(output_npy)
    os.makedirs(dir_path or ".", exist_ok=True)
    if existing_embs is not None:
        embs_all = np.vstack([existing_embs, new_embs])
    else:
        embs_all = new_embs
    np.save(output_npy, embs_all)

    mode = "a" if vec_offset > 0 else "w"
    dir_path = os.path.dirname(output_jsonl)
    os.makedirs(dir_path or ".", exist_ok=True)
    with open(output_jsonl, mode + "t", encoding="utf-8") as f_out:
        for d in data:

            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(
        f"[Embeddings] Saved {len(data)} new vectors to {output_npy} and updated JSONL {output_jsonl}"
    )
    return embs_all, new_embs



def build_and_save_faiss_index(
    embeddings: np.ndarray,
    dataset_name: str,
    index_type: str,
    output_dir: str = ".",
    new_vectors: np.ndarray | None = None,
):
    """Build or update a FAISS cosine-similarity index.

    If ``new_vectors`` is provided and an existing index file is found, the new
    vectors are appended to that index. Otherwise, a fresh index is built from
    ``embeddings``. 
    """
    if not index_type or index_type not in {"passages", "iqoq", "iq"}:
        raise ValueError(
            "index_type must be provided and set to 'passages', 'iqoq', or 'iq'."
        )

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    if new_vectors is not None:
        new_vectors = np.ascontiguousarray(new_vectors, dtype=np.float32)
        faiss.normalize_L2(new_vectors)

    faiss_path = os.path.join(output_dir, f"{dataset_name}_faiss_{index_type}.faiss")

    if new_vectors is not None and os.path.exists(faiss_path):
        index = faiss.read_index(faiss_path)
        assert new_vectors.shape[1] == index.d, (
            f"Dimension mismatch: new_vectors.shape[1]={new_vectors.shape[1]}, index.d={index.d}"
        )
        index.add(new_vectors)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        assert embeddings.shape[1] == index.d, (
            f"Dimension mismatch: embeddings.shape[1]={embeddings.shape[1]}, index.d={index.d}"
        )
        index.add(embeddings)

    faiss.write_index(index, faiss_path)

    print(f"[FAISS] Saved {index_type} index to {faiss_path} with {index.ntotal} vectors.")








"""         SPARSE: Spacy NER + keyword extraction        """ 




ALIAS = {
    "us": "united_states", "u_s": "united_states", "u_s_a": "united_states",
    "united_states_of_america": "united_states", "the_united_states": "united_states",
    "u_s_navy": "united_states_navy", "u_s_air_force": "united_states_air_force",
    "u_s_army": "united_states_army", "u_s_marine_corps": "united_states_marine_corps",
    "uk": "united_kingdom", "u_k": "united_kingdom", "great_britain": "united_kingdom",
    "un": "united_nations", "u_n": "united_nations",
    "what_year": None, "the_years": None, "year": None,
}

_NOISE_PATTERNS = [
    re.compile(r"^\d{4}$"),          # years
    re.compile(r"^\d+$"),            # pure numbers
    re.compile(r"^(one|two|three|four|five|six|seven|eight|nine|ten|first|second|third)$")
]

KEEP_ENTS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
             "WORK_OF_ART", "EVENT", "LAW"}

SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL, disable=["parser", "textcat"])






def filter_keyword(kw, remove_numbers=True):
    kw = ALIAS.get(kw, kw)
    if kw is None:
        return None
    if remove_numbers:
        for pat in _NOISE_PATTERNS:
            if pat.match(kw):
                return None
    return kw




def generate_ngrams(tokens, n_max=3):
    """Return all n-grams (up to n_max) from tokens, skipping stopwords and punctuation."""
    ngrams = set()
    n_tokens = len(tokens)
    for n in range(2, n_max+1):  # start at bigrams
        for i in range(n_tokens - n + 1):
            gram_tokens = tokens[i:i+n]
            # Skip if any token is punctuation or whitespace
            if any(t.is_punct or t.is_space for t in gram_tokens):
                continue
            phrase = "_".join([t.text.lower() for t in gram_tokens])
            ngrams.add(phrase)
    return ngrams





def extract_keywords(text: str, include_ngrams=True, n_max=3, remove_numbers=True) -> list[str]:
    """
    Extract canonical keywords from text:
      1. spaCy NER entities (filtered + aliased)
      2. Optional n-grams (filtered)
    
    Returns a sorted list of unique keywords.
    """
    if not text:
        return []

    doc = nlp(text)
    out = set()

    # NER entities
    for ent in doc.ents:
        if ent.label_ in KEEP_ENTS and ent.text.strip():
            norm = normalise_text(ent.text)
            if norm:
                canon = filter_keyword(norm, remove_numbers=remove_numbers)
                if canon:
                    out.add(canon)

    # Optional n-grams from tokens
    if include_ngrams:
        tokens = [t for t in doc if not t.is_stop and not t.is_space]
        ngrams = generate_ngrams(tokens, n_max=n_max)
        for gram in ngrams:
            canon = filter_keyword(gram, remove_numbers=remove_numbers)
            if canon:
                out.add(canon)

    return sorted(out)






"""        """





def dataset_rep_paths(dataset, split):
    """
    Return paths for dataset-level passage representations, embeddings, FAISS index, and sparse keywords.
    """
    base = os.path.join("data", "representations", "datasets", dataset, split)
    os.makedirs(base, exist_ok=True)
    return {
        "passages_jsonl": os.path.join(base, f"{dataset}_passages.jsonl"),
        "passages_emb": os.path.join(base, f"{dataset}_passages_emb.npy"),
        "passages_index": os.path.join(base, f"{dataset}_faiss_passages.faiss"),
        "passages_sparse_jsonl": os.path.join(base, f"{dataset}_passages_sparse.jsonl"),
    }





"""         MAIN         """
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb





def load_models():
    print(f"[spaCy] Using: {SPACY_MODEL}")
    bge_model = SentenceTransformer(BGE_MODEL, device=DEVICE)
    print(f"[BGE] Loaded {BGE_MODEL} on {DEVICE}")
    return nlp, bge_model







if __name__ == "__main__":
    RESUME = True
    MAX_EXAMPLES = 100
    PHASES = ["datasets", "representations"]
    DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]
    SPLITS = ["train", "dev"]
    BATCH_SIZE = 128  


    SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    
    BGE_MODEL = os.environ.get("BGE_MODEL", "all-MiniLM-L6-v2")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    nlp, bge_model = load_models()


    for phase in PHASES:


        """         Dataset processing       """


        if phase == "datasets":
            print("\n*  *   * DATASET PROCESSING *  *   *\n")
            print("[Phase] Processing datasets...")
            for dataset in DATASETS:
                for split in SPLITS:
                    print(f"\n=== DATASET: {dataset} ({split}) ===")

                    proc_paths = processed_dataset_paths(dataset, split) # make var name clearer
                    qa_path = proc_paths["questions"]
                    passages_path = proc_paths["passages"]

                    # Define field maps per dataset
                    if dataset == "hotpotqa":
                        file_path = (
                            "data/raw_datasets/hotpotqa/hotpot_train_v1.1.json" if split == "train" ######## what about the distractor set? 
                            else "data/raw_datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
                        )
                        field_map = {
                            "get_id": lambda ex: ex["_id"],
                            "get_question": lambda ex: ex["question"],
                            "get_answer": lambda ex: ex.get("answer", ""),
                            "iter_passages": lambda ex: [
                                (pid_plus_title(ex["_id"], title, i), title, sent)
                                for title, sents in ex["context"]
                                for i, sent in enumerate(sents)
                            ],
                            "gold_passage_ids": lambda ex: [
                                pid_plus_title(ex["_id"], title, idx)
                                for title, idx in ex.get("supporting_facts", [])
                            ],
                        }
                    elif dataset == "2wikimultihopqa":
                        file_path = f"data/raw_datasets/2wikimultihopqa/{split}.json"
                        field_map = {
                            "get_id": lambda ex: ex["_id"],
                            "get_question": lambda ex: ex["question"],
                            "get_answer": lambda ex: ex.get("answer", ""),
                            "iter_passages": lambda ex: [
                                (pid_plus_title(ex["_id"], title, i), title, sent)
                                for title, sents in ex["context"]
                                for i, sent in enumerate(sents)
                            ],
                            "gold_passage_ids": lambda ex: [
                                pid_plus_title(ex["_id"], title, idx)
                                for title, idx in ex.get("supporting_facts", [])
                            ],
                        }
                    elif dataset == "musique":
                        file_path = f"data/raw_datasets/musique/musique_ans_v1.0_{split}.jsonl" ### what's the difference between ans and full?
                        field_map = {
                            "get_id": lambda ex: ex["id"],
                            "get_question": lambda ex: ex.get("question", ""),
                            "get_answer": lambda ex: ex.get("answer", ""),
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

                    process_dataset(
                        dataset=dataset,
                        split=split,
                        file_path=file_path,
                        field_map=field_map,
                        max_examples=MAX_EXAMPLES,
                        resume=RESUME,
                    )


            ##### REP PHASE #####

        elif phase == "representations":
            print("\n*  *   * REPRESENTATIONS PROCESSING *  *   *\n")
            print("[Phase] Building passage embeddings & FAISS indexes...")
            for dataset in DATASETS:
                for split in SPLITS:
                    print(f"\n=== DATASET: {dataset} ({split}) ===")


                    # Standardize paths ---- CHANGE NAMING HERE - DON'T LIKE SRC 
                    proc_paths = processed_dataset_paths(dataset, split)
                    rep_paths = dataset_rep_paths(dataset, split)

                    passages_jsonl_src = proc_paths["passages"] 
                    passages_jsonl = rep_paths["passages_jsonl"] 
                    passages_npy = rep_paths["passages_emb"]
                    sparse_jsonl = rep_paths["passages_sparse_jsonl"]
                    dataset_dir = os.path.dirname(passages_jsonl) # WHY IS THIS DIFFERENT???

                    # Resume sets
                    all_passages = list(load_jsonl(passages_jsonl_src))
                    done_ids, _ = compute_resume_sets(
                        resume=RESUME,
                        out_path=passages_jsonl,
                        items=all_passages,
                        get_id=lambda x, i: x["passage_id"],
                        phase_label="passage embeddings",
                        required_field="vec_id",
                    )
                    vec_offset = len(done_ids)

                    # Open sparse JSONL for appending ############################ ?????????????????????????????
                    sparse_mode = "a" if os.path.exists(sparse_jsonl) else "w"
                    f_sparse = open(sparse_jsonl, sparse_mode, encoding="utf-8")

                    # Load existing embeddings if any
                    if os.path.exists(passages_npy):
                        embeddings_all = np.load(passages_npy).astype("float32")
                    else: ######## ?????????????????
                        embeddings_all = np.empty(
                            (0, bge_model.get_sentence_embedding_dimension()), dtype="float32"
                        )






                    # Batch processing
                    batch_texts, batch_entries = [], []
                    for entry in all_passages:
                        pid = entry["passage_id"]
                        if pid in done_ids:
                            continue
                        batch_texts.append(entry["text"])
                        batch_entries.append(entry)

                        if len(batch_texts) >= BATCH_SIZE:


                            # Dense embeddings
                            batch_embs = bge_model.encode(
                                batch_texts, normalize_embeddings=True, batch_size=BATCH_SIZE,
                                convert_to_numpy=True, show_progress_bar=False
                            ).astype("float32")
                            embeddings_all = np.vstack([embeddings_all, batch_embs])

                            # Sparse keywords
                            for i, ent in enumerate(batch_entries):
                                ent["keywords"] = extract_keywords(batch_texts[i])
                                ent["vec_id"] = vec_offset
                                vec_offset += 1
                                f_sparse.write(json.dumps(ent, ensure_ascii=False) + "\n")

                            batch_texts, batch_entries = [], []

                    # Process remaining batch
                    if batch_texts:
                        batch_embs = bge_model.encode(
                            batch_texts, normalize_embeddings=True, batch_size=BATCH_SIZE,
                            convert_to_numpy=True, show_progress_bar=False
                        ).astype("float32")
                        embeddings_all = np.vstack([embeddings_all, batch_embs])

                        for i, ent in enumerate(batch_entries):
                            ent["keywords"] = extract_keywords(batch_texts[i])
                            ent["vec_id"] = vec_offset
                            vec_offset += 1
                            f_sparse.write(json.dumps(ent, ensure_ascii=False) + "\n")

                    f_sparse.close()
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

