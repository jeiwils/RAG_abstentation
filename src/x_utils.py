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




def pid_plus_title(qid: str, title: str, sent_idx: int) -> str:
    """Create a safe passage identifier using question id and title.


    only used with 2wikimulihop and hotpotqa - musique already has a unique identifier for each passage

    The title is normalised by converting to lowercase and replacing any
    non-alphanumeric characters with underscores.  If the provided title is
    empty or sanitisation results in an empty string, ``"no_title"`` is used
    instead.

    Parameters
    ----------
    qid:
        The base identifier, typically the question or passage id.
    title:
        Title text associated with the passage.
    sent_idx:
        Sentence index within the passage.

    Returns
    -------
    str
        A combined identifier ``"{qid}__{safe}_sent{sent_idx}"``.
    """
    if not title:
        safe = "no_title"
    else:
        # Replace any non-word characters with underscores and collapse
        # repeated underscores.  ``\w`` matches alphanumerics and ``_``.
        safe = re.sub(r"[^0-9A-Za-z]+", "_", title.lower()).strip("_")
        if not safe:
            safe = "no_title"
    return f"{qid}__{safe}_sent{sent_idx}"


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








def clean_text(text: str) -> str:
    """Normalise whitespace and remove simple markup for clean text."""
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













