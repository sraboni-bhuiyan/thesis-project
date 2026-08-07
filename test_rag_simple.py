import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

# Reuse the LLM handling from run_baseline
def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    try:
        from openai import OpenAI
        _has_new_openai = True
    except ImportError:
        import openai
        _has_new_openai = False
    if _has_new_openai:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    else:
        openai.api_key = api_key
        openai.api_base = base_url
        return None

def call_model(client_or_none, prompt: str) -> str:
    if _has_new_openai:
        client = client_or_none
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0.0,
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
            model="openai/gpt-4o-mini",
            temperature=0.0,
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
        return urgency, specialty, rationale
    except Exception as e:
        return "PARSE_ERROR", "PARSE_ERROR", f"Could not parse model output: {e}"

def main():
    # Paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    CASES_FILE = PROJECT_ROOT / "data" / "cases" / "main.csv"
    PROMPT_FILE = PROJECT_ROOT / "src" / "prompts" / "rag.txt"
    RESULTS_FILE = PROJECT_ROOT / "results" / "rag_predictions_simple.csv"
    INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
    FAISS_INDEX = INDEX_DIR / "guidelines.index"
    ID_MAP_FILE = INDEX_DIR / "id_map.json"
    CORPUS_FILE = PROJECT_ROOT / "data" / "guidelines_clean" / "corpus.jsonl"
    TOP_K = 3

    # Load API key
    client = load_api_key()
    # Load RAG template
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        template = f.read().strip()
    # Load cases
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    # Limit to first 2 for test
    cases = cases[:2]

    # Load retrieval resources
    print("Loading FAISS index...")
    index = faiss.read_index(str(FAISS_INDEX))
    print("Loading ID map...")
    with open(ID_MAP_FILE, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Loading corpus texts...")
    id_to_text = {}
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            id_to_text[obj["chunk_id"]] = obj["text"]
    texts = [id_to_text[cid] for cid in chunk_ids]
    assert len(texts) == index.ntotal

    # Process
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            case_id = case.get("case_id")
            try:
                # Build query for retrieval
                symptoms = (case.get('symptoms') or '').strip()
                vitals = (case.get('vitals') or '').strip()
                age = (case.get('age') or '').strip()
                gender = (case.get('gender') or '').strip()
                query_text = (
                    "Patient case:\n"
                    f"- Symptoms: {symptoms}\n"
                    f"- Vitals: {vitals}\n"
                    f"- Age: {age}\n"
                    f"- Gender: {gender}"
                )
                # Encode and search
                query_vec = model.encode([query_text], normalize_embeddings=True)
                scores, indices = index.search(query_vec.astype(np.float32), TOP_K)
                scores = scores[0]
                indices = indices[0]
                retrieved_texts = [texts[i] for i in indices]
                retrieved_context = "\n\n---\n\n".join(retrieved_texts)
                # Build final prompt
                prompt = template.format(
                    retrieved_context=retrieved_context,
                    symptoms=symptoms,
                    vitals=vitals,
                    age=age,
                    gender=gender
                )
                # Call model
                model_text = call_model(client, prompt)
                print(f"  Model raw output: {repr(model_text)}")
                urgency, specialty, rationale = parse_model_output(model_text)
                writer.writerow({
                    "case_id": case_id,
                    "input_symptoms": symptoms,
                    "input_vitals": vitals,
                    "input_age": age,
                    "input_gender": gender,
                    "pred_urgency": urgency,
                    "pred_specialty": specialty,
                    "pred_rationale": rationale,
                    "raw_model_output": model_text,
                })
                print(f"Processed case {case_id}: urgency={urgency}, specialty={specialty}")
            except Exception as e:
                print(f"Error processing case {case_id}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                writer.writerow({
                    "case_id": case_id,
                    "input_symptoms": symptoms,
                    "input_vitals": vitals,
                    "input_age": age,
                    "input_gender": gender,
                    "pred_urgency": "ERROR",
                    "pred_specialty": "ERROR",
                    "pred_rationale": str(e),
                    "raw_model_output": "",
                })
    print(f"Done. Results written to {RESULTS_FILE}")

if __name__ == "__main__":
    main()