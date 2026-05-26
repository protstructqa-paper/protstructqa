"""Tests for Method B: Template-matched few-shot exemplar sampling."""
from baselines.run_baseline_batched import pick_exemplars


def _q(qid, family, template, uniprot=None):
    return {"qid": qid, "family": family, "template": template,
              "uniprot": uniprot or qid.split("/")[1]}


def test_tmfs_prefers_same_template():
    train = [
        _q("human/P11111/A1/0", "A", "A1"),
        _q("human/P11112/A1/0", "A", "A1"),
        _q("human/P11113/A1/0", "A", "A1"),
        _q("human/P11114/A1/0", "A", "A1"),
        _q("human/P22221/A2/0", "A", "A2"),
        _q("human/P33331/B1/0", "B", "B1"),
        _q("human/P44441/D1/0", "D", "D1"),
    ]
    target = _q("human/Q00000/A1/99", "A", "A1")
    chosen = pick_exemplars(train, target, n=4, template_match=True)
    assert all(q["template"] == "A1" for q in chosen)
    # No leakage: target uniprot not selected
    assert all(q["uniprot"] != target["uniprot"] for q in chosen)
    assert len(chosen) == 4


def test_tmfs_falls_back_to_family():
    train = [
        # only 2 same-template entries, need 4
        _q("human/P1/A3/0", "A", "A3"),
        _q("human/P2/A3/0", "A", "A3"),
        _q("human/P3/A1/0", "A", "A1"),
        _q("human/P4/A2/0", "A", "A2"),
        _q("human/P5/A4/0", "A", "A4"),
        _q("human/P6/B1/0", "B", "B1"),
    ]
    target = _q("human/Q0/A3/99", "A", "A3")
    chosen = pick_exemplars(train, target, n=4, template_match=True)
    assert len(chosen) == 4
    # First 2 should be A3, then fall back to other A
    a3_count = sum(1 for q in chosen if q["template"] == "A3")
    assert a3_count == 2
    a_family = sum(1 for q in chosen if q["family"] == "A" and q["template"] != "A3")
    assert a_family == 2


def test_tmfs_disabled_matches_old_behavior():
    train = [
        _q("human/P1/A1/0", "A", "A1"),
        _q("human/P2/A1/0", "A", "A1"),
        _q("human/P3/A1/0", "A", "A1"),
        _q("human/P4/A1/0", "A", "A1"),
        _q("human/P5/A2/0", "A", "A2"),
        _q("human/P6/B1/0", "B", "B1"),
        _q("human/P7/D1/0", "D", "D1"),
    ]
    target = _q("human/Q0/A1/99", "A", "A1")
    chosen = pick_exemplars(train, target, n=4, template_match=False)
    # Old behavior: cross-family, never the same template
    assert all(q["template"] != "A1" for q in chosen)


def test_tmfs_deterministic():
    train = [_q(f"human/P{i}/A1/0", "A", "A1") for i in range(20)]
    target = _q("human/Q0/A1/99", "A", "A1")
    a = pick_exemplars(train, target, n=4, template_match=True, seed=42)
    b = pick_exemplars(train, target, n=4, template_match=True, seed=42)
    assert [q["uniprot"] for q in a] == [q["uniprot"] for q in b]


def test_tmfs_default_is_template_match():
    """Default arg must be template_match=True (Method B is the default)."""
    import inspect
    sig = inspect.signature(pick_exemplars)
    assert sig.parameters["template_match"].default is True
