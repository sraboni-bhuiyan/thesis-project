"""
Aggregate all configurations into results/results_summary.csv and figures.

Expects (missing files are skipped):
  results/{cfg}_predictions.csv            on data/cases/main.csv
  results/{cfg}_variants_predictions.csv   on data/cases/fairness_variants.csv
for cfg in baseline, rag, rag_masked.

Writes per-config results/eval_{cfg}.json and results/fairness_{cfg}.json, plus
  results/results_summary.csv
  results/figures/fig1_performance.png
  results/figures/fig2_dpr_heatmap.png
  results/figures/fig3_consistency.png
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import models_in, read_cases, row_key
from config import CASES_PATH, FAIRNESS_PATH, RESULTS_PATH, TEMPERATURE, TOP_K
from evaluate import evaluate
from fairness_metrics import counterfactual_consistency, fairness_from_stats, group_stats

CONFIGS = [("baseline", "C1 Baseline"), ("rag", "C2 RAG"), ("rag_masked", "C3 RAG + masking")]
FIG_DIR = RESULTS_PATH / "figures"

SUMMARY_FIELDS = [
    "config", "model", "n_cases", "n_error", "n_invalid_urgency", "urgency_exact_acc", "urgency_adjacent_acc",
    "over_triage_rate", "under_triage_rate", "specialty_acc", "specialty_acc_any_listed",
    "urgency_dpr", "specialty_acc_gap", "urgency_tpr_gap", "urgency_fpr_gap", "urgency_eod",
    "variants_urgency_dpr", "urgency_consistency", "specialty_consistency", "joint_consistency",
    "urgency_gender_flip_rate", "urgency_age_flip_rate",
]


def fmt(v):
    return "" if v is None else (round(v, 4) if isinstance(v, float) else v)


def main():
    cases = {row_key(r): r for r in read_cases(CASES_PATH)}
    vcases = {row_key(r): r for r in read_cases(FAIRNESS_PATH)}
    rows, fairness_all = [], {}

    for cfg, label in CONFIGS:
        perf_file = RESULTS_PATH / f"{cfg}_predictions.csv"
        var_file = RESULTS_PATH / f"{cfg}_variants_predictions.csv"
        if not perf_file.exists() and not var_file.exists():
            print(f"Skipping {cfg}: no prediction files")
            continue
        recorded = read_cases(perf_file if perf_file.exists() else var_file)
        meta = {"model": models_in(recorded), "temperature": TEMPERATURE, "top_k": TOP_K if cfg != "baseline" else None}
        row = {"config": label, "model": meta["model"]}
        fair = dict(meta)

        if perf_file.exists():
            ev = evaluate(cases, read_cases(perf_file))
            (RESULTS_PATH / f"eval_{cfg}.json").write_text(json.dumps({**meta, **ev}, indent=2), encoding="utf-8")
            row.update(ev)
            fair["perf_set"] = fairness_from_stats(group_stats(cases, read_cases(perf_file)))
            row.update({k: v for k, v in fair["perf_set"].items() if k != "per_bucket"})
        if var_file.exists():
            vpreds = read_cases(var_file)
            fair["variants_set"] = fairness_from_stats(group_stats(vcases, vpreds))
            fair["counterfactual_consistency"] = counterfactual_consistency(vpreds)
            row["variants_urgency_dpr"] = fair["variants_set"]["urgency_dpr"]
            row.update(fair["counterfactual_consistency"])

        (RESULTS_PATH / f"fairness_{cfg}.json").write_text(json.dumps(fair, indent=2), encoding="utf-8")
        fairness_all[label] = fair
        rows.append(row)

    if not rows:
        print("No results found.")
        return

    out = RESULTS_PATH / "results_summary.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in SUMMARY_FIELDS})
    print(f"Wrote {out}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["config"] for r in rows]

    # Fig 1: performance bars
    metrics = [("urgency_exact_acc", "Urgency exact acc"), ("over_triage_rate", "Over-triage"),
               ("under_triage_rate", "Under-triage"), ("specialty_acc", "Specialty acc")]
    x = np.arange(len(metrics))
    width = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(rows):
        ax.bar(x + i * width, [r.get(k) or 0 for k, _ in metrics], width, label=r["config"])
    ax.set_xticks(x + width * (len(rows) - 1) / 2, [m for _, m in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Performance on evaluation set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_performance.png", dpi=200)
    plt.close(fig)

    # Fig 2: P(pred high acuity) per age x gender bucket (perf set)
    have = [l for l in labels if "perf_set" in fairness_all[l]]
    if have:
        fig, axes = plt.subplots(1, len(have), figsize=(3.6 * len(have), 3.4), squeeze=False)
        for ax, l in zip(axes[0], have):
            pb = fairness_all[l]["perf_set"]["per_bucket"]
            grid = np.array([[np.nan if pb[f"{a}_{g}"]["pred_high_rate"] is None else pb[f"{a}_{g}"]["pred_high_rate"]
                              for g in ("male", "female")]
                             for a in ("young", "old")], dtype=float)
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, "-" if np.isnan(grid[i, j]) else f"{grid[i, j]:.2f}",
                            ha="center", va="center", color="white")
            ax.set_xticks([0, 1], ["male", "female"])
            ax.set_yticks([0, 1], ["young <40", "old >=65"] if ax is axes[0][0] else ["", ""])
            dpr = fairness_all[l]["perf_set"]["urgency_dpr"]
            ax.set_title(f"{l}\nDPR={dpr:.2f}" if dpr is not None else l, fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="P(pred high acuity)")
        fig.savefig(FIG_DIR / "fig2_dpr_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Fig 3: counterfactual consistency
    have = [r for r in rows if r.get("urgency_consistency") is not None]
    if have:
        cm = [("urgency_consistency", "Urgency"), ("specialty_consistency", "Specialty"), ("joint_consistency", "Joint")]
        x = np.arange(len(cm))
        width = 0.8 / len(have)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, r in enumerate(have):
            ax.bar(x + i * width, [r[k] for k, _ in cm], width, label=r["config"])
        ax.set_xticks(x + width * (len(have) - 1) / 2, [m for _, m in cm])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of base cases with identical output across 4 variants")
        ax.set_title("Counterfactual consistency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_consistency.png", dpi=200)
        plt.close(fig)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
