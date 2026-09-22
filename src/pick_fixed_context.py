"""
Pick C4's fixed guideline context: the chunks retrieved most often in the C2/C3 logs.

Derived from the logs rather than hand-picked, so the choice is defensible. n = TOP_K keeps
C4's prompt length equal to C2's. Repeat runs (*_rep2) are excluded by default to avoid
double-counting the same queries.

Usage:
  python src/pick_fixed_context.py
"""
import argparse
import json
from collections import Counter

from common import resolve
from config import CORPUS_FILE, RESULTS_PATH, TOP_K

FIXED_CONTEXT_FILE = CORPUS_FILE.parent / "fixed_context.json"


def find_log_files(include_repeats: bool = False):
    """Retrieval logs to count, in stable order. Smoke logs are never counted."""
    logs = sorted(p for p in RESULTS_PATH.glob("retrieval_logs*.jsonl"))
    if not include_repeats:
        logs = [p for p in logs if not p.stem.endswith("_rep2")]
    return logs


def count_retrievals(log_files):
    """(Counter of chunk_id -> times retrieved, total queries, per-file query counts)."""
    counts, per_file, n_queries = Counter(), {}, 0
    for path in log_files:
        n_file = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                counts.update(entry.get("retrieved_chunk_ids", []))
                n_file += 1
        per_file[path.name] = n_file
        n_queries += n_file
    return counts, n_queries, per_file


def load_corpus_texts():
    id_to_chunk = {}
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_to_chunk[obj["chunk_id"]] = obj
    return id_to_chunk


def select_chunk_ids(counts: Counter, n: int):
    """Top-n chunk ids by retrieval count; ties broken by chunk_id for reproducibility."""
    return [cid for cid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def main():
    parser = argparse.ArgumentParser(description="Derive C4's fixed guideline context from the retrieval logs.")
    parser.add_argument("--n-chunks", type=int, default=TOP_K, help=f"Chunks to fix (default: TOP_K = {TOP_K}).")
    parser.add_argument("--include-repeats", action="store_true", help="Also count *_rep2 logs.")
    parser.add_argument("--output", default=None, help=f"Default: {FIXED_CONTEXT_FILE}")
    args = parser.parse_args()

    log_files = find_log_files(args.include_repeats)
    if not log_files:
        raise SystemExit(f"No retrieval logs found in {RESULTS_PATH}")
    counts, n_queries, per_file = count_retrievals(log_files)
    if not counts:
        raise SystemExit("Retrieval logs contain no retrieved_chunk_ids.")

    id_to_chunk = load_corpus_texts()
    selected = select_chunk_ids(counts, args.n_chunks)
    missing = [cid for cid in selected if cid not in id_to_chunk]
    if missing:
        raise SystemExit(f"Chunk ids in logs but not in corpus: {missing}")

    print(f"Counted {n_queries} queries from {len(log_files)} log file(s):")
    for name, n in per_file.items():
        print(f"  {name}: {n}")
    print(f"\nRetrieval frequency (all {len(counts)} retrieved chunks):")
    for cid, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = "*" if cid in selected else " "
        print(f" {mark} {cid:<32} {c:>5} / {n_queries}  ({c / n_queries:.1%})")
    print(f"\nSelected {len(selected)} chunk(s) for the C4 fixed context:")
    for cid in selected:
        print(f"  {cid:<32} {counts[cid]:>5} / {n_queries}  ({counts[cid] / n_queries:.1%})")

    payload = {
        "description": "Fixed guideline context for C4. Derived from retrieval logs, not hand-picked.",
        "n_queries": n_queries,
        "log_files": per_file,
        "include_repeats": args.include_repeats,
        "n_chunks": len(selected),
        "chunk_ids": selected,
        "retrieval_counts": {cid: counts[cid] for cid in selected},
        "all_retrieval_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "chunks": [
            {
                "chunk_id": cid,
                "doc_id": id_to_chunk[cid].get("doc_id"),
                "source": id_to_chunk[cid].get("source"),
                "section": id_to_chunk[cid].get("section"),
                "retrieval_count": counts[cid],
                "text": id_to_chunk[cid]["text"],
            }
            for cid in selected
        ],
    }
    out = resolve(args.output) if args.output else FIXED_CONTEXT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
