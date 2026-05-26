"""TDD tests for baselines/scoring.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, "."
)

from baselines import scoring as sc


# ------------------------------ parsers ------------------------------ #


@pytest.mark.parametrize("text,expected", [
    ("17.5", 17.5),
    ("the answer is 42", 42.0),
    ("approximately 8.3 Angstroms", 8.3),
    ("-3.14", -3.14),
    ("not a number", None),
    ("", None),
    ("1e3", 1000.0),
])
def test_parse_numeric(text, expected):
    assert sc.parse_numeric(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("no", False), ("True", True), ("FALSE", False),
    ("y", True), ("n", False), ("1", True), ("0", False),
    ("yes, residue 50 is buried", True),
    ("not relevant", None),
])
def test_parse_bool(text, expected):
    assert sc.parse_bool(text) == expected


@pytest.mark.parametrize("text", [
    "unreliable", "Unreliable", "UNRELIABLE", "uncertain", "abstain",
    "I don't know", "cannot determine", "low confidence",
])
def test_is_unreliable_response_positives(text):
    assert sc.is_unreliable_response(text)


@pytest.mark.parametrize("text", [
    "5.0", "yes", "True", "residue 50", "helix",
])
def test_is_unreliable_response_negatives(text):
    assert not sc.is_unreliable_response(text)


# ------------------------------ Bool / Int / Float ------------------- #


def test_bool_match():
    assert sc.score_question(True, "Bool", "yes")["correct"]
    assert sc.score_question(False, "Bool", "no")["correct"]
    assert not sc.score_question(True, "Bool", "no")["correct"]


def test_int_within_tolerance():
    assert sc.score_question(50, "Int", "51")["correct"]
    assert sc.score_question(50, "Int", "53")["correct"]   # within +/- 2
    assert not sc.score_question(50, "Int", "60")["correct"]


def test_float_within_tolerance():
    assert sc.score_question(8.3, "Float", "8.4")["correct"]   # 0.1 < 0.5
    assert sc.score_question(8.3, "Float", "8.7")["correct"]   # 0.4 < 0.5
    assert not sc.score_question(8.3, "Float", "12.0")["correct"]   # 3.7 > tol


def test_region_exact():
    assert sc.score_question([100, 130], "Region",
                                "residues 100 to 130")["correct"]
    assert sc.score_question([100, 130], "Region", [100, 130])["correct"]
    assert not sc.score_question([100, 130], "Region", [101, 130])["correct"]


def test_secstruct_match():
    assert sc.score_question("H", "SecStruct", "helix")["correct"]
    assert sc.score_question("E", "SecStruct", "strand")["correct"]
    assert sc.score_question("C", "SecStruct", "coil")["correct"]
    assert not sc.score_question("H", "SecStruct", "strand")["correct"]


# ------------------------------ Selective ---------------------------- #


def test_selective_correct_abstain():
    r = sc.score_question("Unreliable", "Float|Unreliable", "unreliable")
    assert r["correct"]
    assert r["abstained_correctly"]


def test_selective_over_confident():
    r = sc.score_question("Unreliable", "Float|Unreliable", "8.5")
    assert not r["correct"]
    assert r["abstained_correctly"] is False
    assert r["failure_mode"] == "over_confident"


def test_selective_over_abstention():
    r = sc.score_question(8.5, "Float|Unreliable", "unreliable")
    assert not r["correct"]
    assert r["failure_mode"] == "over_abstention"


def test_selective_correct_value():
    r = sc.score_question(8.5, "Float|Unreliable", "8.4")
    assert r["correct"]


# ------------------------------ aggregation -------------------------- #


def test_aggregate_basic():
    scored = [
        {"correct": True}, {"correct": True}, {"correct": False},
        {"correct": True},
    ]
    a = sc.aggregate(scored)
    assert a["n_total"] == 4
    assert a["accuracy_overall"] == 0.75


def test_aggregate_selective():
    scored = [
        {"correct": True, "gold_unreliable": True, "abstained_correctly": True},
        {"correct": False, "gold_unreliable": True, "abstained_correctly": False, "failure_mode": "over_confident"},
        {"correct": True, "gold_unreliable": False, "abstained_correctly": None},
        {"correct": False, "gold_unreliable": False, "abstained_correctly": None, "failure_mode": "over_abstention"},
    ]
    a = sc.aggregate(scored)
    assert a["abstention_recall"] == 0.5     # 1 of 2 gold-unreliable
    assert a["accuracy_when_value"] == 0.5    # 1 of 2 gold-value
    assert a["over_confident_rate"] == 0.25
    assert a["over_abstention_rate"] == 0.25
