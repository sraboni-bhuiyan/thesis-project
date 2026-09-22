"""
C1 Baseline: prompt-only LLM triage, no retrieval.

Usage:
  python src/run_baseline.py [--cases-file F] [--output F] [--limit N]
"""
import argparse

from common import build_case_block, call_model, load_client, load_prompt, read_cases, resolve, run_predictions


def main():
    parser = argparse.ArgumentParser(description="Run baseline (prompt-only) LLM triage on a cases CSV.")
    parser.add_argument("--cases-file", default="data/cases/main.csv")
    parser.add_argument("--output", default="results/baseline_predictions.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases (testing).")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite output instead of resuming.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between calls (rate limits).")
    args = parser.parse_args()

    client = load_client()
    template = load_prompt("baseline.txt")
    cases = read_cases(resolve(args.cases_file), args.limit)

    def predict(case):
        prompt = template.format(case_block=build_case_block(case))
        return call_model(client, prompt)

    run_predictions(cases, resolve(args.output), predict, desc="Baseline",
                    resume=not args.no_resume, sleep=args.sleep)


if __name__ == "__main__":
    main()
