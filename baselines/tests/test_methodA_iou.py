"""Tests for Method A: IoU-tolerant set scoring."""
from baselines.scoring import _set_correct, score_question, SET_IOU_THRESHOLD


def test_pairset_exact_match():
    g = [(1, 5), (2, 6), (3, 7)]
    p = [(1, 5), (2, 6), (3, 7)]
    assert _set_correct(g, p)


def test_pairset_off_by_one_passes_iou():
    """3/4 IoU = 0.6 should fail; 9/10 = 0.9 should pass."""
    g = [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13)]
    p = list(g) + [(10, 14)]   # 9 same + 1 extra: IoU = 9/10 = 0.9
    assert _set_correct(g, p)
    p2 = list(g)[:5]   # 5 of 9 = IoU 5/9 ~ 0.56
    assert not _set_correct(g, p2)


def test_residueset_exact():
    assert _set_correct([1, 2, 3], [1, 2, 3])
    assert not _set_correct([1, 2, 3], [4, 5, 6])


def test_residueset_iou():
    """8/9 IoU = 0.89 should fail (just below 0.9); 9/10 should pass."""
    g = list(range(1, 10))
    p1 = list(range(1, 10)) + [10]   # IoU 9/10 = 0.9 -> pass
    assert _set_correct(g, p1)
    p2 = list(range(1, 9))           # IoU 8/9 ~ 0.889 -> fail
    assert not _set_correct(g, p2)


def test_set_correct_empty():
    assert _set_correct([], [])
    assert not _set_correct([1, 2], [])
    assert not _set_correct([], [1, 2])


def test_score_question_pairset_uses_iou():
    g = [(1, 5)] * 0 + [(i, j) for i in range(20, 25) for j in range(50, 55)]
    p = list(g)
    p.append((100, 200))   # extra
    s = score_question(g, "PairSet", p)
    iou = len(set(g) & set(p)) / len(set(g) | set(p))
    if iou >= SET_IOU_THRESHOLD:
        assert s["correct"]
    else:
        assert not s["correct"]


def test_iou_threshold_constant():
    assert SET_IOU_THRESHOLD == 0.9
