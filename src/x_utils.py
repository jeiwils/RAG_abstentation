"""

query_llm
- put inferencce wall time in this as a optional arg?


get RESUME stuff
- for normal rows + FAISS


"""



from typing import Iterator, Dict, List, Any, Hashable, Set, Tuple, Callable, Iterable
import json
import os
import re
import unicodedata
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from .z_configs import BGE_MODEL, SPACY_MODEL, DEVICE
import spacy

SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL, disable=["textcat"])


"""          PATHS           """


def processed_dataset_paths(dataset, split):
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




def dataset_results_paths(dataset, split, model_name, reward_scheme, seed, paradigm=None):
    parts = ["data", "results", dataset, split, model_name]
    if paradigm:
        parts.append(paradigm)
    parts.append(reward_scheme)
    parts.append(f"seed_{seed}")
    base = os.path.join(*parts)
    os.makedirs(base, exist_ok=True)
    return {
        "base": base,
        "answers": os.path.join(base, "answers.jsonl"),
        "debug": os.path.join(base, "debug.jsonl"),
        "log": os.path.join(base, "training.log"),
    }





"""        INDEXING         """



def pid_plus_title( 
        qid, 
        title, 
        sent_idx
        ):
    """

    ##########################################################

    """
    if not title:
        safe = "no_title"
    else:

        safe = re.sub(r"[^0-9A-Za-z]+", "_", title.lower()).strip("_")
        if not safe:
            safe = "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"






"""         JSON            """








def load_jsonl(path: str, log_skipped: bool = False) -> Iterator[Dict]:
    """Yield objects from a JSONL file one by one.

    Lines that are empty or fail to parse as JSON are skipped. If
    ``log_skipped`` is ``True``, the number of skipped lines is printed.
    """
    skipped = 0
    with open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if log_skipped:
                    print(f"Skipping malformed JSON on line {line_no} in {path}")
    if log_skipped and skipped:
        print(f"Skipped {skipped} empty or malformed lines in {path}")

def save_jsonl(path: str, data: List[Dict]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "wt", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, obj: Dict) -> None:
    """Append a single JSON serialisable object to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")








"""       CLEANING          """


def clean_text(text):
    """


    """
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = unicodedata.normalize("NFKC", text)
    return text


def strip_accents(t: str) -> str:
    """Transliterate `t` to ASCII by stripping accents and special letters."""
    t = unicodedata.normalize("NFKD", t)
    replacements = {"ß": "ss", "æ": "ae", "œ": "oe"}
    for src, tgt in replacements.items():
        t = t.replace(src, tgt)
    return t.encode("ascii", "ignore").decode("ascii")


def normalise_text(s: str) -> str:
    """
    Return a normalised token string suitable for comparisons.
    This handles punctuation, casing, diacritics, and possessives.
    """
    if not s:
        return ""
    t = s.lower()
    t = t.replace("’", "'")       # unify curly quote
    t = strip_accents(t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\W+", "_", t.strip())
    t = re.sub(r"_s_", "_", t)    # drop possessive
    t = re.sub(r"_s$", "", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t













"""          RESUME               """



def existing_ids(path, id_field="passage_id", required_field: str | None = None):
    """Return IDs from ``path`` when available.

    Parameters
    ----------
    path:
        JSONL file to scan.
    id_field:
        Name of the identifier field whose values should be collected.
    required_field:
        Optional field that must also be present in a line for it to
        contribute an ID. This is useful when resuming an embedding step:
        rows that have been written but lack the embedding field (e.g.
        ``vec_id``) will then be ignored, keeping them eligible for
        processing.
    """
    if not Path(path).exists():
        return set()
    done = set()
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get(id_field)
                if pid is not None and (
                    required_field is None or obj.get(required_field) is not None
                ):
                    done.add(pid)
            except Exception:
                # tolerate a possibly truncated last line ################################################ ?
                continue
    return done

def compute_resume_sets(
    *,
    resume: bool,
    out_path: str,
    items: Iterable[Any],
    get_id: Callable[[Any, int], Hashable],
    phase_label: str,
    id_field: str = "passage_id",
    required_field: str | None = None,
) -> Tuple[Set[Hashable], Set[Hashable]]:
    
    """Return ``(done_ids, shard_ids)`` for a single shard.

    When ``resume`` is ``True``, :func:`existing_ids` reads ``out_path`` and the
    function prints a message describing how many items are skipped for *this*
    shard. Pipelines that split work across multiple shards should call this
      function separately for each shard's output file - resumption is per shard
    only. The ``items`` iterable is fully consumed to build ``shard_ids``; pass a
    list or other re-iterable sequence if it will be reused later.

    Parameters
    ----------
    resume:
        Whether to check ``out_path`` and report existing IDs.
    out_path:
        JSONL file produced by the current shard.
    items:
        Input sequence for the shard.
    get_id:
        Callable extracting an identifier from ``items`` with signature
        ``(item, index) -> Hashable``.
    phase_label:
        Human-readable label used in log messages.
    id_field:
        Name of the identifier field inside ``out_path`` JSON objects.
    required_field:
        Optional field that must exist in a JSON object for the corresponding
        ID to be considered "done". This allows partially written records to
        be retried on resume.

    Returns
    -------
    Tuple[Set[Hashable], Set[Hashable]]
        ``done_ids``: IDs already present in ``out_path`` for this shard.
        ``shard_ids``: IDs for all items in the shard.
    """
    shard_ids = {get_id(x, i) for i, x in enumerate(items)}
    if not resume:
        return set(), shard_ids

    done_all = existing_ids(
        out_path, id_field=id_field, required_field=required_field
    )  # only this shard's file; caller handles other shards
    done_ids = done_all & shard_ids  # defensive intersection
    print(
          f"[resume] {phase_label}: {len(done_ids)}/{len(shard_ids)} already present in this shard - skipping those"
      )
    return done_ids, shard_ids













"""      MODEL LOADING       """



def load_model_8bit(model_path, enable_fp32_cpu_offload=False):
    """
    Load a causal LM in 8-bit with optional FP32 CPU offload for large models.
    
    Args:
        model_path (str): path to HuggingFace model
        enable_fp32_cpu_offload (bool): whether to enable llm_int8_enable_fp32_cpu_offload
    
    Returns:
        tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=enable_fp32_cpu_offload
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    return tokenizer, model




def load_models(): ### should this be here or in utils??? what else would I use this for? would I use this when making embeddings for the queries?? 
    print(f"[spaCy] Using: {SPACY_MODEL}")
    bge_model = SentenceTransformer(BGE_MODEL, device=DEVICE)
    print(f"[BGE] Loaded {BGE_MODEL} on {DEVICE}")
    return nlp, bge_model