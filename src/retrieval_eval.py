"""
Small retrieval evaluation (proposal 6.3).

Step 1 - create a rating sheet from a retrieval log:
  python src/retrieval_eval.py sample --log-file results/retrieval_logs.jsonl --n 25
  -> results/retrieval_rating_sheet.csv  (fill the `relevant` column with 1 / 0)

Step 2 - score the filled sheet:
  python src/retrieval_eval.py score
  -> prints P@k, MRR, nDCG@k and writes results/retrieval_eval.json

Note: the guideline corpus has only ~14 chunks, so these numbers are coarse; report as a limitation.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict

from common import resolve
from config import RESULTS_PATH

SHEET = RESULTS_PATH / "retrieval_rating_sheet.csv"
FIELDS = ["case_id", "rank", "chunk_id", "score", "query_excerpt", "chunk_excerpt", "relevant"]


def sample(log_file, n, seed):
    with open(resolve(log_file), "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    random.seed(seed)
    picked = random.sample(entries, min(n, len(entries)))
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in picked:
            for rank, (cid, score, text) in enumerate(zip(e["retrieved_chunk_ids"], e["retrieved_scores"], e["retrieved_texts"]), 1):
                w.writerow({"case_id": e["case_id"], "rank": rank, "chunk_id": cid, "score": round(score, 4),
                            "query_excerpt": e["query"][:600], "chunk_excerpt": text[:600], "relevant": ""})
    print(f"Wrote {len(picked)} cases to {SHEET}. Fill `relevant` with 1/0, then run `score`.")


def score(sheet=SHEET):
    by_case = defaultdict(list)
    with open(resolve(str(sheet)), "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["relevant"].strip() not in {"0", "1"}:
                continue
            by_case[r["case_id"]].append((int(r["rank"]), int(r["relevant"])))
    if not by_case:
        print("No rated rows found.")
        return
    p_at_k, rr, ndcg = [], [], []
    for rels in by_case.values():
        rels = [rel for _, rel in sorted(rels)]
        k = len(rels)
        p_at_k.append(sum(rels) / k)
        rr.append(next((1 / (i + 1) for i, rel in enumerate(rels) if rel), 0.0))
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(sorted(rels, reverse=True)))
        ndcg.append(dcg / idcg if idcg else 0.0)
    res = {"n_cases_rated": len(by_case), "precision_at_k": sum(p_at_k) / len(p_at_k),
           "mrr": sum(rr) / len(rr), "ndcg_at_k": sum(ndcg) / len(ndcg)}
    print(json.dumps(res, indent=2))
    (RESULTS_PATH / "retrieval_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--log-file", default="results/retrieval_logs.jsonl")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score")
    sc.add_argument("--sheet", default=str(SHEET), help="Rated CSV (default: results/retrieval_rating_sheet.csv)")
    args = parser.parse_args()
    sample(args.log_file, args.n, args.seed) if args.cmd == "sample" else score(args.sheet)


if __name__ == "__main__":
    main()
