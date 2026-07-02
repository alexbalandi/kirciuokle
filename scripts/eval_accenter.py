# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "phonology-engine"]
# ///
"""Differential quality eval: current VDU+UDPipe pipeline vs an offline
candidate engine (LIEPA phonology_engine).

The current production pipeline (accent_text.py) is the baseline. Any
replacement engine must be measured against it before swapping — this script
is that regression gate.

Usage:
    uv run scripts/eval_accenter.py corpus.txt [--max-disagreements 40]
"""

import argparse
import asyncio
import difflib
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import accent_text  # noqa: E402  (the validated production pipeline)

STRESS_MARKS = ("̀", "́", "̃")  # grave, acute, tilde
MARK_BY_TYPE = {0: "̀", 1: "́", 2: "̃"}
WORD_RE = re.compile(r"[A-Za-zÀ-žĀ-ſ̀-ͯ]+")


def norm(s: str) -> str:
    """NFD + casefold: comparable stress signature."""
    return unicodedata.normalize("NFD", s).casefold()


def strip_marks(s: str) -> str:
    return "".join(c for c in norm(s) if c not in STRESS_MARKS)


def has_mark(s: str) -> bool:
    return any(m in norm(s) for m in STRESS_MARKS)


def _cp1257_ok(c: str) -> bool:
    try:
        c.encode("windows-1257")
        return True
    except UnicodeEncodeError:
        return False


def tokenize(text: str) -> list[str]:
    # NFD first: precomposed accented letters (ẽ, ỹ, ė́ …) become base letter
    # + combining mark, both covered by WORD_RE.
    return WORD_RE.findall(unicodedata.normalize("NFD", text))


def pe_option_forms(word: str, options: list) -> set[str]:
    """Derive every stress option's form from (letter_index, stress_type, ...)."""
    forms = set()
    for opt in options:
        idx, stress_type = opt[0], opt[1]
        mark = MARK_BY_TYPE.get(stress_type)
        if mark is None or idx >= len(word):
            continue
        forms.add(norm(word[: idx + 1] + mark + word[idx + 1 :]))
    return forms


def run_phonology_engine(text: str) -> list[dict]:
    from phonology_engine import PhonologyEngine

    pe = PhonologyEngine()

    def flatten(node):
        # pe.process() nests custom iterable wrappers; duck-type down to dicts
        if isinstance(node, dict):
            yield node
            return
        if isinstance(node, (str, bytes)):
            return
        try:
            children = iter(node)
        except TypeError:
            return
        for item in children:
            yield from flatten(item)

    words = []
    for w in flatten(pe.process(text)):
        words.append(
                {
                    "base": strip_marks(w["word"]),
                    "form": norm(w["utf8_stressed_word"]),
                    "options": pe_option_forms(w["word"], w["stress_options"]["options"])
                    | {norm(w["utf8_stressed_word"])},
                }
            )
    return words


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--max-disagreements", type=int, default=40)
    args = ap.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    # Fair input for both engines: drop pre-existing stress marks (e.g. cited
    # example words in Wikipedia articles) and anything the LIEPA engine's
    # windows-1257 encoding cannot represent.
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if c not in STRESS_MARKS)
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        c if _cp1257_ok(c) else " " for c in text
    )

    print("running baseline (VDU + UDPipe) ...", file=sys.stderr)
    stressed, unknown, resolved, unresolved = await accent_text.accent_text(text)
    unknown_bases = {strip_marks(w) for w in unknown}

    print("running phonology_engine ...", file=sys.stderr)
    pe_words = run_phonology_engine(text)

    vdu_tokens = [(strip_marks(t), norm(t)) for t in tokenize(stressed)]
    pe_bases = [w["base"] for w in pe_words]

    matcher = difflib.SequenceMatcher(
        a=[b for b, _ in vdu_tokens], b=pe_bases, autojunk=False
    )

    counts: Counter[str] = Counter()
    disagreements: Counter[tuple[str, str, str]] = Counter()
    pe_only_unknown = 0

    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            base, vdu_form = vdu_tokens[block.a + k]
            pe_word = pe_words[block.b + k]
            counts["aligned"] += 1

            vdu_marked, pe_marked = has_mark(vdu_form), has_mark(pe_word["form"])
            if not vdu_marked and not pe_marked:
                counts["both_bare"] += 1
            elif not vdu_marked:
                counts["pe_only"] += 1
                if base in unknown_bases:
                    pe_only_unknown += 1
            elif not pe_marked:
                counts["pe_miss"] += 1
                disagreements[(base, vdu_form, "(bare)")] += 1
            elif vdu_form == pe_word["form"]:
                counts["agree"] += 1
            else:
                counts["disagree"] += 1
                if vdu_form in pe_word["options"]:
                    counts["disagree_reachable"] += 1
                disagreements[(base, vdu_form, pe_word["form"])] += 1

    total_tokens = len(tokenize(text))
    compared = counts["agree"] + counts["disagree"] + counts["pe_miss"]
    agree_pct = 100 * counts["agree"] / compared if compared else 0
    reachable_pct = (
        100 * counts["disagree_reachable"] / counts["disagree"]
        if counts["disagree"]
        else 0
    )

    print(f"corpus tokens:            {total_tokens}")
    print(f"aligned tokens:           {counts['aligned']}")
    print(f"both unaccented:          {counts['both_bare']}")
    print(f"PE-only (VDU has none):   {counts['pe_only']}  "
          f"(of which VDU-unknown words: {pe_only_unknown})")
    print()
    print(f"comparable (VDU-stressed) tokens: {compared}")
    print(f"  agree:    {counts['agree']}  ({agree_pct:.2f}%)")
    print(f"  disagree: {counts['disagree']}  "
          f"(VDU's form among PE options: {counts['disagree_reachable']}, {reachable_pct:.0f}%)")
    print(f"  PE left bare: {counts['pe_miss']}")
    print()
    print(f"top disagreements (of {len(disagreements)} distinct):")
    for (base, vdu_form, pe_form), n in disagreements.most_common(args.max_disagreements):
        print(f"  {n:3}x  {base}: VDU={vdu_form}  PE={pe_form}")


if __name__ == "__main__":
    asyncio.run(main())
