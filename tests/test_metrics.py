"""Hand-checked tests for parsing, evaluation and fairness metrics. Run: python -m pytest tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from common import normalize_specialty, normalize_urgency, parse_model_output
from demographics import mask_demographics, swap_demographics
from evaluate import evaluate
from fairness_metrics import counterfactual_consistency, fairness_from_stats, group_stats


@pytest.mark.parametrize("raw,expected", [
    ("Red", "red"), ("orange (very urgent)", "orange"), ("2", "orange"), ("2 - very urgent emergency care", "orange"),
    ("very urgent", "orange"), ("acuity 4 (minors/standard emergency care)", "green"), ("urgent", "yellow"),
    ("immediate (category 1)", "red"), ("non-urgent", "blue"), ("banana", "INVALID"), ("", "INVALID"),
])
def test_normalize_urgency(raw, expected):
    assert normalize_urgency(raw) == expected


def test_normalize_specialty():
    assert normalize_specialty("Cardiac") == "cardiology"
    assert normalize_specialty(" Orthopedic Surgery ") == "orthopedics"
    assert normalize_specialty("['General Surgery', 'Urology']") == "general surgery"
    assert normalize_specialty("ENT") == "otolaryngology"


def test_parse_model_output():
    assert parse_model_output('```json\n{"urgency": "Orange", "specialty": "Cardiology", "rationale": "x"}\n```')[:2] == ("orange", "Cardiology")
    assert parse_model_output('<think>{maybe}</think>{"urgency": "3", "specialty": "Urology"}')[0] == "yellow"
    assert parse_model_output("not json")[0] == "INVALID"


def test_demographic_text():
    t = "Ms. ___ is ___ with pain. She says it did not wake her. Her husband called."
    male = swap_demographics(t, "male", 70)
    assert "Mr." in male and "He says" in male and "wake him" in male and "His husband" in male
    masked = mask_demographics(t)
    assert not any(w in masked.split() for w in ["Ms.", "She", "her", "Her"])


# ---- evaluation ----
def _case(cid, age, gender, urg, spec="Cardiology"):
    return {"case_id": cid, "age": str(age), "gender": gender, "ground_truth_urgency": urg,
            "ground_truth_specialty": spec, "ground_truth_specialty_all": spec}


def _pred(cid, urg, spec="Cardiology", vid=""):
    return {"case_id": cid, "variant_id": vid, "pred_urgency": urg, "pred_specialty": spec}


def test_evaluate():
    cases = {c["case_id"]: c for c in [_case("1", 30, "Male", "yellow"), _case("2", 30, "Male", "orange"),
                                       _case("3", 30, "Male", "yellow"), _case("4", 30, "Male", "red")]}
    preds = [_pred("1", "yellow"), _pred("2", "red", "Cardiac"), _pred("3", "green", "Urology"), _pred("4", "ERROR", "ERROR")]
    m = evaluate(cases, preds)
    assert m["n_error"] == 1
    assert m["urgency_exact_acc"] == pytest.approx(1 / 3)
    assert m["over_triage_rate"] == pytest.approx(1 / 3)
    assert m["under_triage_rate"] == pytest.approx(1 / 3)
    assert m["urgency_adjacent_acc"] == pytest.approx(1.0)
    assert m["specialty_acc"] == pytest.approx(2 / 3)


# ---- fairness ----
def test_dpr_and_eod():
    # young_male: 2 true-high (1 pred high), 2 true-low (1 pred high)  -> TPR .5, FPR .5, rate .5
    # old_female: 2 true-high (2 pred high), 2 true-low (0 pred high)  -> TPR 1,  FPR 0,  rate .5
    rows = [
        (_case("a", 30, "Male", "red"), "red"), (_case("b", 30, "Male", "orange"), "yellow"),
        (_case("c", 30, "Male", "yellow"), "orange"), (_case("d", 30, "Male", "green"), "green"),
        (_case("e", 70, "Female", "red"), "orange"), (_case("f", 70, "Female", "orange"), "red"),
        (_case("g", 70, "Female", "yellow"), "yellow"), (_case("h", 70, "Female", "green"), "blue"),
        (_case("i", 50, "Female", "red"), "blue"),  # middle age -> excluded
    ]
    cases = {c["case_id"]: c for c, _ in rows}
    preds = [_pred(c["case_id"], p) for c, p in rows]
    f = fairness_from_stats(group_stats(cases, preds))
    assert f["urgency_dpr"] == pytest.approx(0.0)
    assert f["urgency_tpr_gap"] == pytest.approx(0.5)
    assert f["urgency_fpr_gap"] == pytest.approx(0.5)
    assert f["urgency_eod"] == pytest.approx(0.5)
    assert f["per_bucket"]["young_male"]["n"] == 4


def test_counterfactual_consistency():
    def variants(cid, urgs, specs=("Cardiology",) * 4):
        names = ["young_male", "young_female", "old_male", "old_female"]
        return [_pred(cid, u, s, f"{cid}_{n}") for n, u, s in zip(names, urgs, specs)]
    preds = (variants("1", ["red"] * 4)
             + variants("2", ["red", "red", "orange", "orange"])                     # age flips only
             + variants("3", ["red"] * 4, ["Cardiology", "Cardiac", "Neurology", "Cardiology"])
             + variants("4", ["red"] * 3))                                           # incomplete -> skipped
    c = counterfactual_consistency(preds)
    assert c["n_base_cases"] == 3 and c["n_skipped_incomplete_or_error"] == 1
    assert c["urgency_consistency"] == pytest.approx(2 / 3)
    assert c["specialty_consistency"] == pytest.approx(2 / 3)
    assert c["joint_consistency"] == pytest.approx(1 / 3)
    assert c["urgency_gender_flip_rate"] == pytest.approx(0.0)
    assert c["urgency_age_flip_rate"] == pytest.approx(2 / 6)
