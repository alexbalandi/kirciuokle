# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Faithful replication of Anbinderis & Kasparaitis (2010) stressing rules.

"Automatic Stressing of Lithuanian Text Using Decision Trees", Information
Technology and Control 39(1):61-67, DOI 10.5755/j01.itc.39.1.12084.

The method: build a decision tree over letter sequences from a list of
stressed words; a *decision node* is the shallowest node whose subtree has a
single distinct stressing (stressed-letter index + accent mark) AND whose
matched letters already include the stressed letter. Each decision node
yields the shortest unambiguous rule (ÓRK covers órkaitė; okeãnas/okeãno
share OKEÃ). Words get a "#" terminator so a short word is never silently
shadowed by a longer one. Words no rule matches are LEFT UNSTRESSED —
abstention, not a majority guess. The paper's best pipeline applies ending
rules first, then beginning rules on the remainder (end-bgn, 95.53%
token-level accuracy on 1M running words incl. clitic handling).

Differences from the paper, by design: we train on our generated
dictionary's word types (their list came from 1M tokens of hand-reviewed
text; homographs resolved by frequency — we resolve by our default form),
and we skip the clitics list (the guesser tier only ever sees words the
dictionary does not cover; clitics are closed-class and always covered).
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import unicodedata

try:  # pragma: no cover
    from ._common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE
    from .train_guesser import apply_stress, load_training, load_vdu_eval, stress_of
except ImportError:  # pragma: no cover
    from _common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE
    from train_guesser import apply_stress, load_training, load_vdu_eval, stress_of

TERMINATOR = "#"


def extract_rules(items: list[tuple[str, int, str]]) -> dict[str, tuple[int, str]]:
    """items: (terminated sequence, stress index in sequence coords, mark).

    Returns {shortest unambiguous prefix: (index, mark)}. The rule set is
    prefix-free: descendants of a decision node are never emitted.
    """

    items = sorted(items)
    rules: dict[str, tuple[int, str]] = {}
    # explicit stack of (lo, hi, depth): items[lo:hi] share seq[:depth]
    stack = [(0, len(items), 0)]
    while stack:
        lo, hi, depth = stack.pop()
        first_seq, first_idx, first_mark = items[lo]
        if hi - lo == 1:
            # singleton branch: descend just past the stressed letter
            rules[first_seq[: max(depth, first_idx + 1)]] = (first_idx, first_mark)
            continue
        unique = all(
            idx == first_idx and mark == first_mark for _s, idx, mark in items[lo + 1 : hi]
        )
        if unique and first_idx < depth:
            rules[first_seq[:depth]] = (first_idx, first_mark)
            continue
        # split into children by the letter at `depth`; runs are contiguous
        # because items are sorted, and the terminator guarantees no
        # sequence is exhausted before a multi-item group splits
        run_start = lo
        while run_start < hi:
            letter = items[run_start][0][depth]
            run_end = run_start
            while run_end < hi and items[run_end][0][depth] == letter:
                run_end += 1
            stack.append((run_start, run_end, depth + 1))
            run_start = run_end
    return rules


class AnbinderisModel:
    """end-bgn: ending rules first, beginning rules on the remainder."""

    def __init__(self, pairs: list[tuple[str, int, str]]):
        # beginning tree: word + "#", stress index counted from the start
        self.begin = extract_rules(
            [(word + TERMINATOR, pos, mark) for word, pos, mark in pairs]
        )
        # ending tree: reversed word + "#", index counted from the end
        self.end = extract_rules(
            [(word[::-1] + TERMINATOR, len(word) - 1 - pos, mark) for word, pos, mark in pairs]
        )

    @staticmethod
    def _match(rules: dict[str, tuple[int, str]], seq: str) -> tuple[int, str] | None:
        for length in range(1, len(seq) + 1):
            hit = rules.get(seq[:length])
            if hit is not None:
                return hit
        return None

    def predict(self, word: str) -> tuple[int, str] | None:
        """Return (stressed letter index, mark) or None (abstain)."""
        hit = self._match(self.end, word[::-1] + TERMINATOR)
        if hit is not None:
            offset, mark = hit
            return len(word) - 1 - offset, mark
        hit = self._match(self.begin, word + TERMINATOR)
        if hit is not None:
            return hit
        return None

    def predict_form(self, word: str) -> str | None:
        hit = self.predict(word)
        return apply_stress(word, *hit) if hit else None


def score(predict, rows, label: str) -> dict[str, float]:
    answered = exact = position = 0
    for word, forms in rows:
        pred = predict(word)
        if pred is None:
            continue
        answered += 1
        norm = [unicodedata.normalize("NFC", f) for f in forms]
        if unicodedata.normalize("NFC", pred) in norm:
            exact += 1
        gold = {(stress_of(f) or (None,))[0] for f in norm}
        if (stress_of(pred) or (None,))[0] in gold:
            position += 1
    n = len(rows) or 1
    a = answered or 1
    print(
        f"{label}: n={len(rows):,} answered={answered / n:.1%} "
        f"exact={exact / a:.1%} position={position / a:.1%} (of answered)"
    )
    return {"n": len(rows), "answered": answered, "exact": exact, "position": position}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=float, default=0.02)
    args = parser.parse_args(argv)

    pairs = load_training(DEFAULT_GENERATED)
    rng = random.Random(20260705)
    rng.shuffle(pairs)
    cut = int(len(pairs) * args.holdout)
    held, train = pairs[:cut], pairs[cut:]
    model = AnbinderisModel(train)
    print(
        f"training words: {len(train):,}; rules: "
        f"end={len(model.end):,} begin={len(model.begin):,}"
    )

    held_rows = [(w, [apply_stress(w, p, m)]) for w, p, m in held]
    score(model.predict_form, held_rows, "in-domain held-out")

    db = sqlite3.connect(DEFAULT_GENERATED)
    covered = {w for (w,) in db.execute("SELECT word FROM words")}
    vdu_rows = load_vdu_eval(DEFAULT_VDU_SQLITE, covered)
    score(model.predict_form, vdu_rows, "VDU-uncovered slice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
