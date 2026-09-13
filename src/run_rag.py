"""
C2 RAG and C3 RAG + demographic masking.

Examples:
  python src/run_rag.py --limit 20
  python src/run_rag.py --mask-demographics --output results/rag_masked_predictions.csv --log-file results/retrieval_logs_masked.jsonl
  python src/run_rag.py --cases-file data/cases/fairness_variants.csv --output results/rag_variants_predictions.csv --log-file results/retrieval_logs_variants.jsonl
"""
import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from common import (build_case_block, call_model, load_client, load_prompt, read_cases, resolve,
                    row_key, run_predictions)
from config import CORPUS_FILE, EMBEDDING_MODEL, FAISS_INDEX, ID_MAP_FILE, TOP_K


def load_retriever():
    """Load FAISS index, chunk ids (index order), corpus texts and the embedding model."""
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


def main():
    parser = argparse.ArgumentParser(description="Run RAG LLM triage on a cases CSV.")
    parser.add_argument("--cases-file", default="data/cases/main.csv")
    parser.add_argument("--output", default=None, help="Default: results/rag_predictions.csv (or rag_masked_predictions.csv)")
    parser.add_argument("--log-file", default=None, help="Default: results/retrieval_logs[_masked].jsonl")
    parser.add_argument("--mask-demographics", action="store_true",
                        help="C3: remove age/gender from the retrieval query and the generation prompt.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    suffix = "_masked" if args.mask_demographics else ""
    output = resolve(args.output or f"results/rag{suffix}_predictions.csv")
    log_path = resolve(args.log_file or f"results/retrieval_logs{suffix}.jsonl")

    client = load_client()
    template = load_prompt("rag.txt")
    cases = read_cases(resolve(args.cases_file), args.limit)
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
        query_vec = embedder.encode([case_block], normalize_embeddings=True).astype(np.float32)
        scores, idx = index.search(query_vec, args.top_k)
        retrieved = [texts[i] for i in idx[0]]
        prompt = template.format(retrieved_context="\n\n---\n\n".join(retrieved), case_block=case_block)
        entry = {
            "case_id": case.get("case_id"),
            "variant_id": case.get("variant_id", ""),
            "masked": args.mask_demographics,
            "query": case_block,
            "retrieved_chunk_ids": [chunk_ids[i] for i in idx[0]],
            "retrieved_scores": [float(s) for s in scores[0]],
            "retrieved_texts": retrieved,
        }
        if row_key(case) not in existing:
            f_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_log.flush()
        return call_model(client, prompt)

    try:
        run_predictions(cases, output, predict, desc="RAG-masked" if args.mask_demographics else "RAG",
                        resume=not args.no_resume, sleep=args.sleep)
    finally:
        f_log.close()
    print(f"Retrieval logs written to {log_path}")


if __name__ == "__main__":
    main()
