import csv
import json
from pathlib import Path

from dotenv import load_dotenv
import os
from openai import OpenAI

from config import MODEL_NAME, TEMPERATURE

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "data" / "cases" / "main.csv"
PROMPT_FILE = PROJECT_ROOT / "src" / "prompts" / "baseline.txt"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "baseline_predictions.csv"


def load_api_key():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    # Create OpenAI client for OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client


def load_baseline_template() -> str:
    """
    Load the baseline prompt template from src/prompts/baseline.txt.
    """
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_cases():
    """
    Read evaluation cases from data/cases/main.csv.
    """
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cases = list(reader)
    return cases


def build_case_prompt(template: str, case: dict) -> str:
    """
    Build the prompt for one case by combining the template with case details.
    Handles None values in CSV fields.
    """
    # Safely get values, defaulting to empty string if None or missing
    symptoms = (case.get('symptoms') or '').strip()
    vitals = (case.get('vitals') or '').strip()
    age = (case.get('age') or '').strip()
    gender = (case.get('gender') or '').strip()

    case_block = (
        "Patient case:\n"
        f"- Symptoms: {symptoms}\n"
        f"- Vitals: {vitals}\n"
        f"- Age: {age}\n"
        f"- Gender: {gender}\n"
        "\n"
        "Return STRICT JSON with exactly these keys:\n"
        '  {"urgency": "...", "specialty": "...", "rationale": "..."}\n'
        "Do not include any extra keys, comments, or text outside the JSON."
    )
    return template + "\n\n" + case_block


def call_model(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a careful clinical triage assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract content from response
    if not response.choices:
        raise RuntimeError(f"Model returned no choices. Response: {response}")

    choice = response.choices[0]
    if not choice.message or not choice.message.content:
        raise RuntimeError(f"Model returned no content. Choice: {choice}")

    return choice.message.content.strip()


def extract_json_block(text: str) -> str:
    """
    Extract a JSON object from the model text output.
    Handles cases where the model wraps JSON in ```json fences.
    """
    # Remove code fences if present
    if text.startswith("```"):
        # e.g. ```json ... ```
        lines = text.splitlines()
        # drop first and last fence lines
        inner = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(inner).strip()

    # Heuristic: find first '{' and last '}' and keep that slice
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_model_output(text: str):
    """
    Parse model output into urgency, specialty, rationale.
    If parsing fails, return 'PARSE_ERROR' placeholders for debugging.
    """
    try:
        json_str = extract_json_block(text)
        obj = json.loads(json_str)
        urgency = str(obj.get("urgency", "")).strip()
        specialty = str(obj.get("specialty", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        return urgency, specialty, rationale
    except Exception as e:
        # For debugging: you can log e and text if needed
        return "PARSE_ERROR", "PARSE_ERROR", f"Could not parse model output: {e}"


def run_baseline():
    client = load_api_key()
    template = load_baseline_template()
    cases = read_cases()

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

    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            case_id = case.get("case_id")
            try:
                prompt = build_case_prompt(template, case)
                model_text = call_model(client, prompt)
                urgency, specialty, rationale = parse_model_output(model_text)

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
            except Exception as e:
                # Log error but continue processing other cases
                print(f"Error processing case {case_id}: {e}")
                # Write error row
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

    print(f"Baseline predictions written to {RESULTS_FILE}")


if __name__ == "__main__":
    run_baseline()