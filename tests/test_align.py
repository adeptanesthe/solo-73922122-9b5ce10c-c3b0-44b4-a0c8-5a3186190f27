"""Tests for reference-subtitle forced alignment (app.align + state.align)."""

import itertools
import math

import numpy as np
import pytest

from app.align import forced_align
from app.decoder import NEG_INF
from app.state import MAX_ALIGN_CHARS, SessionError, SessionManager, advance, align

VOCAB = ["<blank>", "a", "b"]


def log_probs(frames):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(frames, dtype=np.float64))


def collapse(path, vocab):
    out = []
    previous = None
    for k in path:
        if k != 0 and k != previous:
            out.append(vocab[k])
        previous = k
    return "".join(out)


def states_of(path, text, vocab):
    """Extended-graph state sequence of a label path collapsing to ``text``."""
    emitted = 0
    states = []
    previous = None
    for k in path:
        if k != 0 and k != previous:
            emitted += 1
        states.append(2 * emitted if k == 0 else 2 * emitted - 1)
        previous = k
    assert emitted == len(text)
    return states


def brute_force_align(frames, text, vocab):
    """Exhaustive best path and per-char spans over all tied best paths."""
    v_size = len(vocab)
    best = NEG_INF
    best_spans = None
    for path in itertools.product(range(v_size), repeat=len(frames)):
        if collapse(path, vocab) != text:
            continue
        logp = 0.0
        for t, k in enumerate(path):
            p = frames[t][k]
            if p <= 0.0:
                logp = NEG_INF
                break
            logp += math.log(p)
        if logp == NEG_INF:
            continue
        states = states_of(path, text, vocab)
        spans = []
        for j in range(len(text)):
            frames_j = [t for t, s in enumerate(states) if s == 2 * j + 1]
            spans.append((min(frames_j), max(frames_j)))
        if best == NEG_INF or logp > best + 1e-9:
            best = logp
            best_spans = spans
        elif abs(logp - best) <= 1e-9:
            best_spans = [
                (min(a[0], b[0]), max(a[1], b[1]))
                for a, b in zip(best_spans, spans)
            ]
    return best, best_spans


def ids(vocab, text):
    return [vocab.index(ch) for ch in text]


# ------------------------------------------------------------------- core DP


def test_forced_align_matches_brute_force_random():
    rng = np.random.default_rng(11)
    vocab = ["<blank>", "a", "b"]
    for trial in range(30):
        frame_count = int(rng.integers(3, 7))
        probs = rng.random((frame_count, 3))
        probs /= probs.sum(axis=1, keepdims=True)
        if trial % 5 == 0:  # sprinkle hard zeros
            probs[rng.integers(0, frame_count), rng.integers(0, 3)] = 0.0
        text = "".join(rng.choice(["a", "b"], size=int(rng.integers(1, 4))))
        result = forced_align(log_probs(probs), ids(vocab, text))
        best, spans = brute_force_align(probs.tolist(), text, vocab)
        if best == NEG_INF:
            assert not result["aligned"], (trial, text)
            continue
        assert result["aligned"], (trial, text)
        assert abs(result["logp"] - best) < 1e-9, (trial, text)
        assert result["spans"] == spans, (trial, text)


def test_forced_align_deterministic_spans():
    frames = [
        [0.9, 0.1, 0.0],   # blank
        [0.0, 1.0, 0.0],   # "a" forced (blank impossible)
        [0.0, 1.0, 0.0],   # "a" sustained
        [0.9, 0.05, 0.05], # blank
        [0.0, 0.0, 1.0],   # "b" forced
    ]
    result = forced_align(log_probs(frames), ids(VOCAB, "ab"))
    assert result["aligned"]
    assert result["spans"] == [(1, 2), (4, 4)]
    expected = math.log(0.9) + math.log(1.0) + math.log(1.0) + math.log(0.9) + math.log(1.0)
    assert abs(result["logp"] - expected) < 1e-12


def test_forced_align_ties_span_full_range():
    # Every frame is [blank .5, a .5]: all 2^3 - 1 non-empty placements of "a"
    # tie at the same score, so the span covers frames 0..2.
    frames = [[0.5, 0.5]] * 3
    result = forced_align(log_probs(frames), [1])
    assert result["aligned"]
    assert result["spans"] == [(0, 2)]
    assert abs(result["logp"] - math.log(0.125)) < 1e-12


def test_forced_align_repeat_needs_blank():
    frames = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    result = forced_align(log_probs(frames), [1, 1])
    assert result["aligned"]
    assert result["spans"] == [(0, 0), (2, 2)]
    assert result["logp"] == 0.0

    # Without a blank frame the repeated character cannot be separated.
    frames = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    result = forced_align(log_probs(frames), [1, 1])
    assert not result["aligned"]


def test_forced_align_zero_probability_unreachable():
    frames = [[0.5, 0.5, 0.0]] * 4  # "b" has probability 0 everywhere
    result = forced_align(log_probs(frames), ids(VOCAB, "ab"))
    assert not result["aligned"]


def test_forced_align_too_few_frames():
    frames = [[0.5, 0.5]]  # one frame cannot hold two separated characters
    result = forced_align(log_probs(frames), [1, 1])
    assert not result["aligned"]


def test_forced_align_rejects_bad_input():
    with pytest.raises(Exception):
        forced_align(log_probs([[0.5, 0.5]]), [])
    with pytest.raises(Exception):
        forced_align(log_probs([[0.5, 0.5]]), [0])  # blank is not a reference char
    with pytest.raises(Exception):
        forced_align(log_probs([[0.5, 0.5]]), [7])


# ------------------------------------------------------------------ sessions


def make_session(frames, width=4, vocab=VOCAB):
    manager = SessionManager()
    return manager.create(vocab, width, frames)


def test_align_uses_only_consumed_frames():
    frames = [
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.9, 0.05, 0.05],
        [0.0, 0.0, 1.0],
    ]
    session = make_session(frames)
    result = align(session, "ab")
    assert not result["aligned"]  # nothing consumed yet
    assert result["frame"] == 0
    assert "尚未推进" in result["message"]

    advance(session, 2)
    result = align(session, "ab")
    assert not result["aligned"]  # "b" not reachable in the first two frames
    assert result["frame"] == 2

    advance(session, 4)
    result = align(session, "ab")
    assert result["aligned"]
    assert result["frame"] == 4
    assert [c["char"] for c in result["characters"]] == ["a", "b"]
    assert [(c["start"], c["end"]) for c in result["characters"]] == [(1, 1), (3, 3)]
    assert result["logp"] is not None


def test_align_validation():
    session = make_session([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    advance(session, 2)
    with pytest.raises(SessionError):
        align(session, "")
    with pytest.raises(SessionError):
        align(session, "a" * (MAX_ALIGN_CHARS + 1))
    with pytest.raises(SessionError):
        align(session, "z")
    with pytest.raises(SessionError):
        align(session, "<blank>")


def test_align_does_not_mutate_session():
    frames = [[0.5, 0.5, 0.0], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
    session = make_session(frames)
    advance(session, 3)
    before = (session.revision, session.next_frame, session.prefix,
              [(b.text, b.logp) for b in session.beams])
    align(session, "ab")
    align(session, "a")
    after = (session.revision, session.next_frame, session.prefix,
             [(b.text, b.logp) for b in session.beams])
    assert before == after


def test_align_result_is_json_safe():
    import json

    session = make_session([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    advance(session, 2)
    json.dumps(align(session, "ab"))
    json.dumps(align(session, "ba"))
