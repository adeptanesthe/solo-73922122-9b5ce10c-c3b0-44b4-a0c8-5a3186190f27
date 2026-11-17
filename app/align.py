"""Forced alignment of a reference text to the frames already consumed.

The aligner answers the annotator's question "which acoustic frames can this
character land on?" after a subtitle has been corrected.  It runs Viterbi
(max-plus) dynamic programming over the standard CTC alignment graph — the
extended label sequence ``blank, y1, blank, …, yL, blank`` — and therefore
finds the highest-probability frame path that collapses to the full reference
text.  The usual CTC rules apply:

* a run of identical labels collapses into one character, so two equal
  characters in a row must be separated by at least one blank frame;
* a frame with probability 0 (``log p = -inf``) makes every path through it
  unreachable, and if no positive-probability path remains the text cannot be
  aligned at all.

Besides the best path score, this module reports for every character the
earliest frame at which **any** tied best path enters the character's label
state and the latest frame at which any tied best path leaves it.  That slack
is exactly what an annotator needs to review timing boundaries at liaisons and
pauses.  Ties are detected with a forward/backward pass: a state sits on an
optimal path iff ``delta + beta`` equals the best total score (up to a small
floating-point tolerance; max-plus recurrences introduce only addition
rounding).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .decoder import BLANK_INDEX, NEG_INF, DecodeError

__all__ = ["forced_align"]


def _extended_labels(label_ids: Sequence[int]) -> np.ndarray:
    """``[y1, …, yL]`` → ``[blank, y1, blank, …, yL, blank]`` (length ``2L+1``)."""
    labels = np.zeros(2 * len(label_ids) + 1, dtype=np.int64)
    labels[1::2] = np.asarray(label_ids, dtype=np.int64)
    return labels


def forced_align(frames_logp: np.ndarray, label_ids: Sequence[int]) -> dict:
    """Align ``label_ids`` (vocab indices ≥ 1) to ``frames_logp`` ((T, V) logs).

    Returns ``{"aligned": False, ...}`` when no positive-probability path
    collapses to the text; otherwise ``{"aligned": True, "logp": best,
    "spans": [(start, end), ...]}`` with one ``(start, end)`` pair per
    character — 0-based inclusive frame indices of the character's label
    state, minimised / maximised over all tied best paths.
    """
    frames = np.asarray(frames_logp, dtype=np.float64)
    if frames.ndim != 2:
        raise DecodeError("帧数据必须是二维数组")
    if not label_ids:
        raise DecodeError("参考文本不能为空")
    ids = np.asarray(label_ids, dtype=np.int64)
    if np.any(ids <= BLANK_INDEX) or np.any(ids >= frames.shape[1]):
        raise DecodeError("参考文本标签索引越界")
    frame_count = frames.shape[0]
    if frame_count == 0:
        return {"aligned": False, "logp": None, "spans": None}

    labels = _extended_labels(label_ids)
    n_states = labels.size
    emit = frames[:, labels]  # (T, S) emission log prob of each state per frame

    # A transition s -> s+2 is legal only when the target is a non-blank label
    # different from the source label (a repeated character must go through
    # the blank state in between).  `allow_skip` is indexed by target state,
    # `allow_skip_from` by source state (same flag, shifted by two).
    allow_skip = np.zeros(n_states, dtype=bool)
    allow_skip[2:] = (labels[2:] != BLANK_INDEX) & (labels[2:] != labels[:-2])
    allow_skip_from = np.zeros(n_states, dtype=bool)
    allow_skip_from[:-2] = allow_skip[2:]

    # Forward pass: delta[t, s] = best score of a path covering frames 0..t
    # and sitting in state s at frame t.
    delta = np.full((frame_count, n_states), NEG_INF)
    delta[0, 0] = emit[0, 0]
    delta[0, 1] = emit[0, 1]  # the first label may already fire at frame 0
    for t in range(1, frame_count):
        prev = delta[t - 1]
        best = prev.copy()  # stay in s
        best[1:] = np.maximum(best[1:], prev[:-1])  # s-1 -> s
        skip = np.full(n_states, NEG_INF)
        skip[2:] = prev[:-2]  # s-2 -> s where allowed
        best = np.where(allow_skip & (skip > best), skip, best)
        delta[t] = best + emit[t]

    best_total = max(float(delta[frame_count - 1, n_states - 1]),
                     float(delta[frame_count - 1, n_states - 2]))
    if best_total == NEG_INF:
        return {"aligned": False, "logp": None, "spans": None}

    # Backward pass: beta[t, s] = best score of frames t+1..T-1 starting from
    # state s at frame t (emission at t itself is not included).
    beta = np.full((frame_count, n_states), NEG_INF)
    beta[frame_count - 1, n_states - 1] = 0.0
    beta[frame_count - 1, n_states - 2] = 0.0
    for t in range(frame_count - 2, -1, -1):
        nxt = beta[t + 1] + emit[t + 1]  # cost of moving to s' at t+1
        best = nxt.copy()  # stay
        best[:-1] = np.maximum(best[:-1], nxt[1:])  # s -> s+1
        skip = np.full(n_states, NEG_INF)
        skip[:-2] = nxt[2:]  # s -> s+2 where allowed
        best = np.where(allow_skip_from & (skip > best), skip, best)
        beta[t] = best

    # States through which an optimal path passes.  Max-plus arithmetic only
    # accumulates float-addition rounding, so a relative tolerance is safe.
    tol = 1e-9 * max(1.0, abs(best_total))
    total = delta + beta
    on_optimal = np.isfinite(total) & (np.abs(total - best_total) <= tol)

    spans = []
    for j in range(len(label_ids)):
        state = 2 * j + 1  # label state of character j
        enters: list[int] = []
        exits: list[int] = []
        for t in range(frame_count):
            if not on_optimal[t, state]:
                continue
            # First frame of the character in some optimal path: an optimal
            # edge arrives from a *different* state (or the path starts here).
            if t == 0:
                enters.append(t)
            else:
                for source in (state - 1, state - 2):
                    if source < 0:
                        continue
                    if source == state - 2 and not allow_skip[state]:
                        continue
                    if on_optimal[t - 1, source] and abs(
                        delta[t - 1, source] + emit[t, state] - delta[t, state]
                    ) <= tol:
                        enters.append(t)
                        break
            # Last frame: an optimal edge leaves for a different state (or the
            # path ends here).
            if t == frame_count - 1:
                exits.append(t)
            else:
                for target in (state + 1, state + 2):
                    if target >= n_states:
                        continue
                    if target == state + 2 and not allow_skip[target]:
                        continue
                    if on_optimal[t + 1, target] and abs(
                        delta[t, state] + emit[t + 1, target] - delta[t + 1, target]
                    ) <= tol:
                        exits.append(t)
                        break
        if not enters or not exits:  # unreachable for an aligned text
            return {"aligned": False, "logp": None, "spans": None}
        spans.append((min(enters), max(exits)))

    return {"aligned": True, "logp": best_total, "spans": spans}
