import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

# Handle both old and new openai versions
try:
    from openai import OpenAI
    _has_new_openai = True
except ImportError:
    import openai
    _has_new_openai = False

from config import MODEL_NAME, TEMPERATURE

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "data" / "cases" / "main.csv"
PROMPT_FILE = PROJECT_ROOT / "src" / "prompts" / "rag.txt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "rag_predictions.csv"
LOG_FILE = RESULTS_DIR / "retrieval_logs.jsonl"

INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
FAISS_INDEX = INDEX_DIR / "guidelines.index"
ID_MAP_FILE = INDEX_DIR / "id_map.json"
CORPUS_FILE = PROJECT_ROOT / "data" / "guidelines_clean" / "corpus.jsonl"

# Retrieval parameters
TOP_K = 3  # number of retrieved passages to use


def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    if _has_new_openai:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        return client
    else:
        openai.api_key = api_key
        openai.api_base = base_url
        return None  # We'll use the module directly


def load_rag_template() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_cases(limit=None):
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
        if limit is not None:
            cases = cases[:limit]
    return cases


def build_case_query(case: dict) -> str:
    """Create the query text for retrieval (same format as baseline prompt without JSON instruction)."""
    symptoms = (case.get('symptoms') or '').strip()
    vitals = (case.get('vitals') or '').strip()
    age = (case.get('age') or '').strip()
    gender = (case.get('gender') or '').strip()

    return (
        "Patient case:\n"
        f"- Symptoms: {symptoms}\n"
        f"- Vitals: {vitals}\n"
        f"- Age: {age}\n"
        f"- Gender: {gender}"
    )


def build_rag_prompt(template: str, case: dict, retrieved_context: str) -> str:
    """Fill in the RAG template with retrieved context and case fields."""
    symptoms = (case.get('symptoms') or '').strip()
    vitals = (case.get('vitals') or '').strip()
    age = (case.get('age') or '').strip()
    gender = (case.get('gender') or '').strip()

    return template.format(
        retrieved_context=retrieved_context,
        symptoms=symptoms,
        vitals=vitals,
        age=age,
        gender=gender
    )


def load_index_and_resources():
    """Load FAISS index, ID map, embedding model, and corpus texts."""
    print(f"Loading FAISS index from {FAISS_INDEX}...")
    index = faiss.read_index(str(FAISS_INDEX))

    print(f"Loading ID map from {ID_MAP_FILE}...")
    with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)  # list of chunk_id strings in index order

    # Use a lightweight embedding model for retrieval (same as used in build_index.py)
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model {embedding_model_name}...")
    model = SentenceTransformer(embedding_model_name)

    # Load corpus texts in the same order as chunk_ids (we assume the index was built in the same order as the corpus)
    # We'll read the corpus and create a mapping from chunk_id to text, then align with chunk_ids.
    print("Loading corpus texts...")
    id_to_text = {}
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_to_text[obj["chunk_id"]] = obj["text"]
    # Build a list of texts in the same order as chunk_ids
    texts = [id_to_text[cid] for cid in chunk_ids]
    # Sanity check
    assert len(texts) == index.ntotal, "Mismatch between index size and number of texts"

    return index, chunk_ids, model, texts


def call_model(client_or_none, prompt: str) -> str:
    if _has_new_openai:
        client = client_or_none
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are a careful clinical triage assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise RuntimeError(f"Model returned no choices. Response: {response}")
        choice = response.choices[0]
        if not choice.message or not choice.message.content:
            raise RuntimeError(f"Model returned no content. Choice: {choice}")
        return choice.message.content.strip()
    else:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": "You are a careful clinical triage assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if not response['choices']:
            raise RuntimeError(f"Model returned no choices. Response: {response}")
        choice = response['choices'][0]
        if not choice['message'] or not choice['message']['content']:
            raise RuntimeError(f"Model returned no content. Choice: {choice}")
        return choice['message']['content'].strip()


def extract_json_block(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(inner).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_model_output(text: str):
    try:
        json_str = extract_json_block(text)
        obj = json.loads(json_str)
        urgency = str(obj.get("urgency", "")).strip()
        specialty = str(obj.get("specialty", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        # Map numeric urgency to color if needed (1:red,2:orange,3:yellow,4:green,5:blue)
        urgency_lower = urgency.lower()
        if urgency_lower.isdigit():
            num = int(urgency_lower)
            mapping = {1: "red", 2: "orange", 3: "yellow", 4: "green", 5: "blue"}
            urgency = mapping.get(num, urgency)  # fallback to original if not 1-5
        else:
            # Ensure lowercase for consistency
            urgency = urgency_lower
        return urgency, specialty, rationale
    except Exception as e:
        return "PARSE_ERROR", "PARSE_ERROR", f"Could not parse model output: {e}"


def run_rag(limit=None):
    client = load_api_key()
    template = load_rag_template()
    cases = read_cases(limit=limit)

    # Load retrieval resources
    index, chunk_ids, model, texts = load_index_and_resources()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "input_symptoms",
        "input_vitals",
        "input_age",
        "input_gender",
        "pred_urgency",
        "pred_specialty",
        "pred_rationale",
        "raw_model_output",
    ]

    # Open retrieval log file for appending
    log_entries = []

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            case_id = case.get("case_id")
            try:
                # 1. Build query for retrieval
                query_text = build_case_query(case)
                # 2. Encode query
                query_vec = model.encode([query_text], normalize_embeddings=True)
                # 3. Search
                scores, indices = index.search(query_vec.astype(np.float32), TOP_K)
                scores = scores[0]  # shape (TOP_K,)
                indices = indices[0]  # shape (TOP_K,)

                # 4. Retrieve texts
                retrieved_texts = [texts[i] for i in indices]
                retrieved_chunk_ids = [chunk_ids[i] for i in indices]

                # 5. Build retrieved context (simple concatenation with separator)
                retrieved_context = "\n\n---\n\n".join(retrieved_texts)

                # 6. Build final prompt
                prompt = build_rag_prompt(template, case, retrieved_context)

                # 7. Call model
                model_text = call_model(client, prompt)
                print(f"  Model raw output (first 100): {repr(model_text[:100])}")
                urgency, specialty, rationale = parse_model_output(model_text)

                # 8. Write prediction
                writer.writerow(
                    {
                        "case_id": case_id,
                        "input_symptoms": case.get("symptoms", ""),
                        "input_vitals": case.get("vitals", ""),
                        "input_age": case.get("age", ""),
                        "input_gender": case.get("gender", ""),
                        "pred_urgency": urgency,
                        "pred_specialty": specialty,
                        "pred_rationale": rationale,
                        "raw_model_output": model_text,
                    }
                )

                # 9. Log retrieval details
                log_entries.append(
                    {
                        "case_id": case_id,
                        "query": query_text,
                        "retrieved_chunk_ids": retrieved_chunk_ids,
                        "retrieved_scores": scores.tolist(),
                        "retrieved_texts": retrieved_texts,
                    }
                )

            except Exception as e:
                print(f"Error processing case {case_id}: {e}")
                # If we have a prompt and model_text from earlier steps, show them for debugging
                debug_info = {}
                if 'prompt' in locals():
                    debug_info['prompt'] = prompt[:200]  # truncate
                if 'model_text' in locals():
                    debug_info['model_text'] = model_text[:200]
                print(f"  Debug info: {debug_info}")
                writer.writerow(
                    {
                        "case_id": case_id,
                        "input_symptoms": case.get("symptoms", ""),
                        "input_vitals": case.get("vitals", ""),
                        "input_age": case.get("age", ""),
                        "input_gender": case.get("gender", ""),
                        "pred_urgency": "ERROR",
                        "pred_specialty": "ERROR",
                        "pred_rationale": str(e),
                        "raw_model_output": "",
                    }
                )
                # Still log the error? We'll skip logging for error cases for simplicity.

        # Write retrieval logs after processing all cases (or we could write incrementally)
        with open(LOG_FILE, "w", encoding="utf-8") as f_log:
            for entry in log_entries:
                f_log.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"RAG predictions written to {RESULTS_FILE}")
    print(f"Retrieval logs written to {LOG_FILE}")


if __name__ == "__main__":
    # Optional limit from command line: python src/run_rag.py 10
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass
    run_rag(limit=limit)