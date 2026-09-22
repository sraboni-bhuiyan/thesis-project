"""
Build the FAISS index over the guideline corpus.

Re-run after every corpus rebuild: a stale index paired with a new corpus gives wrong
retrievals silently, not an error.

Usage:
  python src/build_index.py
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_FILE = PROJECT_ROOT / "data" / "guidelines_clean" / "corpus.jsonl"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
INDEX_FILE = INDEX_DIR / "guidelines.index"
ID_MAP_FILE = INDEX_DIR / "id_map.json"  # mapping from index position to chunk_id

# Must match config.EMBEDDING_MODEL: query and document encoders have to be the same model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_corpus():
    """Load corpus.jsonl and return list of (chunk_id, text)."""
    records = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append((obj["chunk_id"], obj["text"]))
    return records

def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading corpus from {CORPUS_FILE}...")
    records = load_corpus()
    chunk_ids, texts = zip(*records) if records else ([], [])
    print(f"Loaded {len(texts)} chunks.")

    print(f"Loading embedding model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding texts...")
    # Normalize embeddings for inner product = cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Embeddings shape: {embeddings.shape}")

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors.")

    faiss.write_index(index, str(INDEX_FILE))
    print(f"Saved index to {INDEX_FILE}")

    # index position -> chunk_id
    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(list(chunk_ids), f, ensure_ascii=False, indent=2)
    print(f"Saved ID map to {ID_MAP_FILE}")

if __name__ == "__main__":
    build_index()