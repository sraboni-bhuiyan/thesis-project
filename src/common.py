"""
Shared inference utilities for all configs (C1 baseline, C2 RAG, C3 RAG+masking):
API client, model call with retries, JSON parsing, label normalization, resumable CSV runner.
"""
import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from config import MAX_TOKENS, MODEL_NAME, ROOT, TEMPERATURE, URGENCY_MAP, URGENCY_ORDER

csv.field_size_limit(10_000_000)

SYSTEM_PROMPT = "You are a careful clinical triage assistant. You always answer with a single JSON object."


# ============================================================
# API client
# ============================================================
def load_client() -> OpenAI:
    """
    Build an OpenAI-compatible client from .env.
    Key:  OPENROUTER_API_KEY | OPENAI_API_KEY | NIM_API_KEY
    Base: OPENROUTER_BASE_URL | NIM_BASE_URL | OPENAI_BASE_URL
    """
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("NIM_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("NIM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("No API key found in .env (expected OPENAI_API_KEY with OPENROUTER_BASE_URL).")
    return OpenAI(api_key=api_key.strip(), base_url=base_url.strip() if base_url else None, timeout=180)


_json_mode_supported = True


def _is_retryable(err: Exception) -> bool:
    s = str(err).lower()
    return any(t in s for t in ("429", "too many requests", "rate limit", "500", "502", "503", "504",
                                "service unavailable", "overload", "timeout", "timed out", "connection"))


def call_model(client: OpenAI, prompt: str, max_retries: int = 6) -> str:
    """Call the chat model in JSON mode with exponential backoff on rate limits / transient errors."""
    global _json_mode_supported
    delay = 2.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        kwargs = dict(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if _json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**kwargs)
            if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
                raise RuntimeError(f"Model returned no content: {response}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            s = str(e).lower()
            if _json_mode_supported and "response_format" in s and ("400" in s or "unsupported" in s):
                print("  JSON mode not supported by provider; falling back to prompt-only JSON.")
                _json_mode_supported = False
                continue
            if attempt < max_retries and _is_retryable(e):
                tqdm.write(f"  Transient error (attempt {attempt}/{max_retries}): {str(e)[:120]} - retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            break
    raise RuntimeError(f"Model call failed after {attempt} attempts: {last_err}")


# ============================================================
# Parsing + normalization
# ============================================================
URGENCY_SYNONYMS = [
    # (regex, colour) - checked in order, first match wins
    (r"\bnon[- ]?urgent\b|\bnot urgent\b|\bacuity 5\b|\bcategory 5\b|\blevel 5\b", "blue"),
    (r"\bvery urgent\b|\bemergent\b|\bacuity 2\b|\bcategory 2\b|\blevel 2\b", "orange"),
    (r"\bimmediate\b|\bresuscitation\b|\bacuity 1\b|\bcategory 1\b|\blevel 1\b|\b999\b|\blife[- ]threatening\b", "red"),
    (r"\bstandard\b|\bminors?\b|\bacuity 4\b|\bcategory 4\b|\blevel 4\b|\bless urgent\b", "green"),
    (r"\burgent\b|\bacuity 3\b|\bcategory 3\b|\blevel 3\b", "yellow"),
    (r"\bemergency\b", "orange"),
]


def normalize_urgency(value) -> str:
    """Map a raw urgency string to one of red/orange/yellow/green/blue, or 'INVALID'."""
    u = str(value or "").strip().lower()
    if not u:
        return "INVALID"
    for colour in URGENCY_ORDER:
        if re.search(rf"\b{colour}\b", u):
            return colour
    m = re.fullmatch(r"\s*([1-5])\s*", u) or re.match(r"^\s*([1-5])\b", u)
    if m:
        return URGENCY_MAP[int(m.group(1))]
    for pattern, colour in URGENCY_SYNONYMS:
        if re.search(pattern, u):
            return colour
    return "INVALID"


SPECIALTY_SYNONYMS = {
    "general surgery": ["surgery", "general surgical", "acute surgery", "surgical"],
    "gastroenterology": ["gi", "gastro", "gastrointestinal", "hepatology"],
    "orthopedics": ["orthopedic surgery", "orthopaedics", "orthopaedic surgery", "orthopedic", "ortho", "trauma and orthopedics"],
    "neurosurgery": ["neurological surgery", "neuro surgery"],
    "cardiology": ["cardiac", "cardiovascular", "cardiovascular medicine"],
    "urology": ["urologic surgery", "urological"],
    "neurology": ["neuro", "neurological", "stroke"],
    "vascular surgery": ["vascular"],
    "emergency medicine": ["emergency", "em", "a&e", "emergency department"],
    "oncology": ["hematology/oncology", "medical oncology"],
    "dermatology": ["derm"],
    "infectious disease": ["infectious diseases", "id"],
    "gynecology": ["obstetrics and gynecology", "obstetrics", "ob/gyn", "obgyn", "gynaecology", "obstetrics & gynecology"],
    "pulmonology": ["pulmonary", "respiratory", "respiratory medicine", "pulmonary medicine"],
    "otolaryngology": ["ent", "ear nose and throat", "ear, nose and throat"],
    "oral and maxillofacial surgery": ["maxillofacial surgery", "omfs", "oral surgery"],
    "nephrology": ["renal", "renal medicine"],
    "hematology": ["haematology"],
    "internal medicine": ["general medicine", "medicine", "general practice", "family medicine", "primary care"],
    "psychiatry": ["mental health", "psychiatric"],
    "endocrinology": ["endocrine"],
    "ophthalmology": ["eye", "ophthalmic"],
}
_SPECIALTY_LOOKUP = {}
for canon, syns in SPECIALTY_SYNONYMS.items():
    _SPECIALTY_LOOKUP[canon] = canon
    for syn in syns:
        _SPECIALTY_LOOKUP[syn] = canon


def normalize_specialty(value) -> str:
    """Lowercase, strip, take first item of a list-like string, map synonyms to a canonical name."""
    s = str(value or "").strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].split(",")[0]
    s = re.split(r"[;,(]| / | or ", s)[0]
    s = s.strip().strip("'\"").lower()
    s = re.sub(r"\s+", " ", s)
    return _SPECIALTY_LOOKUP.get(s, s)


def extract_json_block(text: str) -> str:
    """Strip <think> blocks / code fences and return the outermost {...} slice."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_model_output(text: str):
    """Return (urgency, specialty, rationale). Urgency is normalized; unparseable output -> INVALID."""
    try:
        obj = json.loads(extract_json_block(text))
        if not isinstance(obj, dict):
            raise ValueError("JSON is not an object")
    except Exception as e:
        return "INVALID", "INVALID", f"Could not parse model output: {e}"
    urgency = normalize_urgency(obj.get("urgency"))
    specialty = str(obj.get("specialty", "")).strip() or "INVALID"
    rationale = str(obj.get("rationale", "")).strip()
    return urgency, specialty, rationale


# ============================================================
# Case IO + resumable runner
# ============================================================
PRED_FIELDS = [
    "case_id", "variant_id", "input_age", "input_gender",
    "pred_urgency", "pred_specialty", "pred_rationale", "raw_model_output", "model",
]


def models_in(preds: list) -> str:
    """Model id(s) actually recorded in a prediction file."""
    return "|".join(sorted({r.get("model") for r in preds if r.get("model")})) or "unknown"


def build_case_block(case: dict, mask: bool = False) -> str:
    """Patient case text used in prompts (and as the retrieval query). mask=True drops age/gender (C3)."""
    symptoms = (case.get("symptoms") or "").strip()
    vitals = (case.get("vitals") or "").strip()
    lines = ["Patient case:", f"- Symptoms: {symptoms}", f"- Vitals: {vitals}"]
    if mask:
        from demographics import mask_demographics
        lines[1] = f"- Symptoms: {mask_demographics(symptoms)}"
    else:
        lines += [f"- Age: {(case.get('age') or '').strip()}", f"- Gender: {(case.get('gender') or '').strip()}"]
    return "\n".join(lines)


def load_prompt(name: str) -> str:
    from config import PROMPTS_PATH
    return (PROMPTS_PATH / name).read_text(encoding="utf-8").strip()


def read_cases(path: Path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    return cases[:limit] if limit is not None else cases


def row_key(case: dict) -> str:
    return case.get("variant_id") or case.get("case_id")


def load_done_keys(output: Path):
    """Keys of rows already predicted successfully (ERROR rows are retried on resume)."""
    if not output.exists():
        return set(), []
    with open(output, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("pred_urgency") != "ERROR"]
    return {row_key(r) for r in rows}, rows


def run_predictions(cases, output: Path, predict_fn, desc: str, resume: bool = True, sleep: float = 0.0):
    """
    predict_fn(case) -> (model_text, extra) ; writes one CSV row per case.
    With resume=True, successful rows already in `output` are kept and skipped.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    done, kept = load_done_keys(output) if resume else (set(), [])
    todo = [c for c in cases if row_key(c) not in done]
    if done:
        print(f"Resuming: {len(done)} rows already done, {len(todo)} to go.")

    n_err = 0
    with open(output, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=PRED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
        for case in tqdm(todo, desc=desc, unit="case"):
            base = {
                "case_id": case.get("case_id"),
                "variant_id": case.get("variant_id", ""),
                "input_age": case.get("age", ""),
                "input_gender": case.get("gender", ""),
                "model": MODEL_NAME,
            }
            try:
                model_text = predict_fn(case)
                urgency, specialty, rationale = parse_model_output(model_text)
                writer.writerow({**base, "pred_urgency": urgency, "pred_specialty": specialty,
                                 "pred_rationale": rationale, "raw_model_output": model_text})
            except Exception as e:
                n_err += 1
                tqdm.write(f"Error on {row_key(case)}: {e}")
                writer.writerow({**base, "pred_urgency": "ERROR", "pred_specialty": "ERROR",
                                 "pred_rationale": str(e), "raw_model_output": ""})
            f_out.flush()
            if sleep:
                time.sleep(sleep)
    print(f"Predictions written to {output} ({n_err} errors; re-run the same command to retry them).")


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
