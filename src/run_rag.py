"""
C2 RAG, C3 RAG + demographic masking, C4 fixed-context control.

C4 (--fixed-context) separates "retrieval helped" from "guideline text helped": same prompt and
settings, but {retrieved_context} holds the same chunks every time, from fixed_context.json
(derived by pick_fixed_context.py). No embedding, no FAISS, no ranking.

Usage:
  python src/run_rag.py [--mask-demographics | --fixed-context] [--cases-file F] [--output F] [--log-file F]
"""
import argparse
import json

import numpy as np

from common import (build_case_block, call_model, load_client, load_prompt, read_cases, resolve,
                    row_key, run_predictions)
from config import CORPUS_FILE, EMBEDDING_MODEL, FAISS_INDEX, ID_MAP_FILE, TOP_K
from pick_fixed_context import FIXED_CONTEXT_FILE

# How retrieved chunks are pasted into the {retrieved_context} slot of rag.txt.
# C4 must use this exact separator so the only difference from C2 is which text is pasted.
CHUNK_SEPARATOR = "\n\n---\n\n"


def load_retriever():
    """Load FAISS index, chunk ids (index order), corpus texts and the embedding model."""
    import faiss
    from sentence_transformers import SentenceTransformer

    index = faiss.read_index(str(FAISS_INDEX))
    chunk_ids = json.loads(ID_MAP_FILE.read_text(encoding="utf-8"))
    id_to_text = {}
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_to_text[obj["chunk_id"]] = obj["text"]
    texts = [id_to_text[cid] for cid in chunk_ids]
    assert len(texts) == index.ntotal, "Index size does not match corpus; re-run build_index.py"
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, chunk_ids, texts, model


def load_fixed_context():
    """C4: the same chunk ids and texts for every case, in the order stored by pick_fixed_context.py."""
    if not FIXED_CONTEXT_FILE.exists():
        raise SystemExit(f"{FIXED_CONTEXT_FILE} not found; run: python src/pick_fixed_context.py")
    payload = json.loads(FIXED_CONTEXT_FILE.read_text(encoding="utf-8"))
    chunks = payload["chunks"]
    if len(chunks) != payload["n_chunks"] or not chunks:
        raise SystemExit(f"{FIXED_CONTEXT_FILE} is inconsistent; re-run src/pick_fixed_context.py")
    return [c["chunk_id"] for c in chunks], [c["text"] for c in chunks], payload


def main():
    parser = argparse.ArgumentParser(description="Run RAG LLM triage on a cases CSV.")
    parser.add_argument("--cases-file", default="data/cases/main.csv")
    parser.add_argument("--output", default=None, help="Default: results/rag[_masked][_fixed]_predictions.csv")
    parser.add_argument("--log-file", default=None, help="Default: results/retrieval_logs[_masked][_fixed].jsonl")
    parser.add_argument("--mask-demographics", action="store_true",
                        help="C3: remove age/gender from the retrieval query and the generation prompt.")
    parser.add_argument("--fixed-context", action="store_true",
                        help="C4: paste the same guideline chunks into every prompt; no embedding, no FAISS search.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    suffix = ("_masked" if args.mask_demographics else "") + ("_fixed" if args.fixed_context else "")
    output = resolve(args.output or f"results/rag{suffix}_predictions.csv")
    log_path = resolve(args.log_file or f"results/retrieval_logs{suffix}.jsonl")

    client = load_client()
    template = load_prompt("rag.txt")
    cases = read_cases(resolve(args.cases_file), args.limit)
    if args.fixed_context:
        # C4: no index, no embedder - the context is the same for every case.
        fixed_ids, fixed_texts, fixed_meta = load_fixed_context()
        print(f"Fixed context: {len(fixed_ids)} chunk(s) from {FIXED_CONTEXT_FILE.name} "
              f"(derived from {fixed_meta['n_queries']} logged queries): {', '.join(fixed_ids)}")
    else:
        index, chunk_ids, texts, embedder = load_retriever()

    # Keep log entries of already-finished rows when resuming
    existing = {}
    if log_path.exists() and not args.no_resume:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                existing[e.get("variant_id") or e["case_id"]] = e
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f_log = open(log_path, "w", encoding="utf-8")
    for e in existing.values():
        f_log.write(json.dumps(e, ensure_ascii=False) + "\n")

    def predict(case):
        case_block = build_case_block(case, mask=args.mask_demographics)
        if args.fixed_context:
            ids, retrieved, scores_out = fixed_ids, fixed_texts, None
        else:
            query_vec = embedder.encode([case_block], normalize_embeddings=True).astype(np.float32)
            scores, idx = index.search(query_vec, args.top_k)
            ids = [chunk_ids[i] for i in idx[0]]
            retrieved = [texts[i] for i in idx[0]]
            scores_out = [float(s) for s in scores[0]]
        prompt = template.format(retrieved_context=CHUNK_SEPARATOR.join(retrieved), case_block=case_block)
        entry = {
            "case_id": case.get("case_id"),
            "variant_id": case.get("variant_id", ""),
            "masked": args.mask_demographics,
            "fixed_context": args.fixed_context,
            "query": case_block,
            "retrieved_chunk_ids": ids,
            "retrieved_scores": scores_out,
            "retrieved_texts": retrieved,
        }
        if row_key(case) not in existing:
            f_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_log.flush()
        return call_model(client, prompt)

    try:
        desc = "RAG-fixed" if args.fixed_context else "RAG"
        if args.mask_demographics:
            desc += "-masked"
        run_predictions(cases, output, predict, desc=desc,
                        resume=not args.no_resume, sleep=args.sleep)
    finally:
        f_log.close()
    print(f"Retrieval logs written to {log_path}")


if __name__ == "__main__":
    main()
