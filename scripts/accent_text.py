# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fully stress (accentuate) Lithuanian text.

Pipeline:
  1. VDU kirciuokle (https://kalbu.vdu.lt) accents the whole text and flags
     ambiguous words (same spelling, different stress).
  2. UDPipe 2 (LINDAT, lithuanian-alksnis model) tags the same text in
     context: lemma, POS, case/gender/number/tense/person.
  3. For each ambiguous word the VDU per-word variants (each carries its own
     morphology label) are scored against the contextual tag; the best match
     wins. A small lemma table handles homographs whose variants share
     identical morphology (e.g. yra: buti -> yra`, irti -> y~ra).

Usage:
    uv run scripts/accent_text.py input.txt            # stressed text to stdout, report to stderr
    uv run scripts/accent_text.py input.txt -o out.txt
    echo "mano tekstas" | uv run scripts/accent_text.py -
    uv run scripts/accent_text.py input.txt --no-tagger   # skip UDPipe disambiguation
"""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field

import httpx

PAGE_URL = "https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/"
AJAX_URL = "https://kalbu.vdu.lt/ajax-call"
UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL = "lithuanian-alksnis"
CHUNK_LIMIT = 4500  # the VDU web UI caps input at 5000 chars

# ---------------------------------------------------------------- VDU client


async def get_nonce(client: httpx.AsyncClient) -> str:
    r = await client.get(PAGE_URL, timeout=30)
    r.raise_for_status()
    m = re.search(r'"NONCE":"([0-9a-f]+)"', r.text)
    if not m:
        raise RuntimeError("could not find nonce on kirciuoklis page")
    return m.group(1)


def chunk_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = max(text.rfind(c, 0, limit) for c in (".", "!", "?", "\n", " "))
        if cut <= 0:
            cut = limit
        chunks.append(text[: cut + 1])
        text = text[cut + 1 :]
    if text:
        chunks.append(text)
    return chunks


async def vdu_call(client: httpx.AsyncClient, nonce: str, data: dict) -> dict:
    r = await client.post(AJAX_URL, data={**data, "nonce": nonce}, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 200:
        raise RuntimeError(f"VDU API error: {payload}")
    return json.loads(payload["message"])


async def text_accents(client: httpx.AsyncClient, nonce: str, body: str) -> list[dict]:
    parts: list[dict] = []
    for chunk in chunk_text(body):
        msg = await vdu_call(client, nonce, {"action": "text_accents", "body": chunk})
        parts.extend(msg["textParts"])
    return parts


async def word_variants(client: httpx.AsyncClient, nonce: str, word: str) -> list[dict]:
    """-> [{form: 'yra`', mi: ['vksm., es. l., 3 asm.', ...]}]"""
    try:
        msg = await vdu_call(client, nonce, {"action": "word_accent", "word": word})
    except (RuntimeError, httpx.HTTPError):
        return []
    out = []
    for entry in msg.get("accentInfo", []):
        mi = [i.get("mi", "") for i in entry.get("information", []) if i.get("mi")]
        for form in entry.get("accented", []):
            out.append({"form": unicodedata.normalize("NFC", form), "mi": mi})
    return out


# ------------------------------------------------------------- UDPipe client


@dataclass
class Token:
    form: str
    lemma: str
    upos: str
    feats: dict = field(default_factory=dict)


async def udpipe_tag(client: httpx.AsyncClient, text: str) -> list[Token]:
    r = await client.post(
        UDPIPE_URL,
        data={"tokenizer": "", "tagger": "", "model": UDPIPE_MODEL, "data": text},
        timeout=120,
    )
    r.raise_for_status()
    conllu = r.json()["result"]
    tokens = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 6 or not cols[0].isdigit():  # skip ranges/empty nodes
            continue
        feats = {}
        if cols[5] != "_":
            feats = dict(f.split("=", 1) for f in cols[5].split("|"))
        tokens.append(Token(form=cols[1], lemma=cols[2], upos=cols[3], feats=feats))
    return tokens


# ------------------------------------------------- variant scoring machinery

# VDU 'mi' abbreviation -> normalized (slot, value) in UD terms
MI_TAGS = {
    "dkt.": ("pos", "NOUN"),
    "bdv.": ("pos", "ADJ"),
    "vksm.": ("pos", "VERB"),
    "dlv.": ("pos", "PART_VERB"),  # participle
    "psdlv.": ("pos", "PART_VERB"),
    "padlv.": ("pos", "PART_VERB"),
    "prv.": ("pos", "ADV"),
    "įv.": ("pos", "PRON"),
    "sktv.": ("pos", "NUM"),
    "jng.": ("pos", "CCONJ"),
    "prl.": ("pos", "ADP"),
    "dll.": ("pos", "PART"),
    "jst.": ("pos", "INTJ"),
    "vyr. g.": ("gender", "Masc"),
    "mot. g.": ("gender", "Fem"),
    "bev. g.": ("gender", "Neut"),
    "vns.": ("number", "Sing"),
    "dgs.": ("number", "Plur"),
    "vard.": ("case", "Nom"),
    "kilm.": ("case", "Gen"),
    "naud.": ("case", "Dat"),
    "gal.": ("case", "Acc"),
    "įnag.": ("case", "Ins"),
    "viet.": ("case", "Loc"),
    "šauksm.": ("case", "Voc"),
    "es. l.": ("tense", "Pres"),
    "būt. l.": ("tense", "Past"),
    "būt. k. l.": ("tense", "Past"),
    "būt. d. l.": ("tense", "PastIter"),
    "būs. l.": ("tense", "Fut"),
    "1 asm.": ("person", "1"),
    "2 asm.": ("person", "2"),
    "3 asm.": ("person", "3"),
    "veik. r.": ("voice", "Act"),
    "neveik. r.": ("voice", "Pass"),
    "aukšt. l.": ("degree", "Cmp"),
    "aukšč. l.": ("degree", "Sup"),
}


def parse_mi(mi: str) -> dict:
    """'bdv., vyr. g., vns. vard.' -> {pos: ADJ, gender: Masc, number: Sing, case: Nom}"""
    tags = {}
    s = mi.strip()
    # try longest-first so 'būt. k. l.' wins over 'būt. l.'
    for abbr in sorted(MI_TAGS, key=len, reverse=True):
        if abbr in s:
            slot, val = MI_TAGS[abbr]
            tags.setdefault(slot, val)
            s = s.replace(abbr, " ")
    return tags


def token_tags(tok: Token) -> dict:
    tags = {}
    up = tok.upos
    if up in ("VERB", "AUX"):
        tags["pos"] = "PART_VERB" if tok.feats.get("VerbForm") == "Part" else "VERB"
    elif up in ("NOUN", "PROPN"):
        tags["pos"] = "NOUN"
    elif up in ("CCONJ", "SCONJ"):
        tags["pos"] = "CCONJ"
    else:
        tags["pos"] = up
    for slot, feat in (
        ("gender", "Gender"),
        ("number", "Number"),
        ("case", "Case"),
        ("tense", "Tense"),
        ("person", "Person"),
        ("voice", "Voice"),
        ("degree", "Degree"),
    ):
        if feat in tok.feats:
            v = tok.feats[feat]
            if slot == "degree":
                v = {"Pos": None, "Cmp": "Cmp", "Sup": "Sup"}.get(v, v)
                if v is None:
                    continue
            tags[slot] = v
    return tags


def score(variant_tags: dict, ctx: dict) -> int:
    s = 0
    if "pos" in variant_tags and "pos" in ctx:
        s += 4 if variant_tags["pos"] == ctx["pos"] else -3
    for slot in ("case", "gender", "number", "tense", "person", "voice", "degree"):
        a, b = variant_tags.get(slot), ctx.get(slot)
        if a and b:
            s += 2 if a == b else -2
    return s


# same-morphology homographs: (word_lower, lemma) -> accented form
LEMMA_EXCEPTIONS = {
    ("yra", "būti"): "yrà",
    ("yra", "irti"): "ỹra",
}


def pick_variant(word: str, variants: list[dict], tok: Token | None):
    """-> (form, resolved_by) where resolved_by in {'lemma', 'context', None}"""
    if not variants:
        return None, None
    if tok:
        exc = LEMMA_EXCEPTIONS.get((word.lower(), tok.lemma))
        if exc:
            for v in variants:
                if v["form"] == unicodedata.normalize("NFC", exc):
                    return v["form"], "lemma"
        ctx = token_tags(tok)
        scored = [
            (max((score(parse_mi(mi), ctx) for mi in v["mi"]), default=0), i, v)
            for i, v in enumerate(variants)
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if len(scored) > 1 and scored[0][0] > scored[1][0]:
            return scored[0][2]["form"], "context"
    return variants[0]["form"], None


def match_case(accented: str, original: str) -> str:
    if original.isupper() and len(original) > 1:
        return accented.upper()
    if original[0].isupper():
        return accented[0].upper() + accented[1:]
    return accented


# ------------------------------------------------------------------ pipeline


def align(parts: list[dict], tokens: list[Token]) -> list[Token | None]:
    """For each WORD part, find the matching UDPipe token (in order)."""
    out: list[Token | None] = []
    ti = 0
    for p in parts:
        if p.get("type") not in ("WORD", "NON_LT"):
            continue
        found = None
        for j in range(ti, min(ti + 8, len(tokens))):
            if tokens[j].form.lower() == p["string"].lower():
                found = tokens[j]
                ti = j + 1
                break
        out.append(found)
    return out


async def accent_text(text: str, use_tagger: bool = True):
    async with httpx.AsyncClient(headers={"User-Agent": "accent-script/1.0"}) as client:
        nonce = await get_nonce(client)
        tagger_task = asyncio.create_task(udpipe_tag(client, text)) if use_tagger else None
        parts = await text_accents(client, nonce, text)

        tokens: list[Token] = []
        if tagger_task:
            try:
                tokens = await tagger_task
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                print(f"[tagger unavailable, falling back to defaults: {e}]", file=sys.stderr)

        word_parts = [p for p in parts if p.get("type") in ("WORD", "NON_LT")]
        aligned = align(parts, tokens) if tokens else [None] * len(word_parts)

        # fetch variants once per distinct ambiguous word
        ambiguous_words = {
            p["string"].lower()
            for p in word_parts
            if p.get("accentType") == "MULTIPLE_MEANING"
        }
        sem = asyncio.Semaphore(6)

        async def fetch(w):
            async with sem:
                return w, await word_variants(client, nonce, w)

        variants_by_word = dict(await asyncio.gather(*(fetch(w) for w in sorted(ambiguous_words))))

        out: list[str] = []
        unknown: set[str] = set()
        resolved: dict[str, str] = {}
        unresolved: dict[str, list[str]] = {}
        wi = 0
        for p in parts:
            if p.get("type") not in ("WORD", "NON_LT"):
                out.append(p["string"])
                continue
            tok = aligned[wi]
            wi += 1
            if p.get("type") == "NON_LT" or p.get("accentType") == "NONE":
                out.append(p["string"])
                unknown.add(p["string"])
            elif p.get("accentType") == "MULTIPLE_MEANING":
                variants = variants_by_word.get(p["string"].lower(), [])
                form, how = pick_variant(p["string"], variants, tok)
                if form is None:
                    form = p.get("accented", p["string"])
                form = match_case(form, p["string"])
                out.append(form)
                key = p["string"].lower()
                if how:
                    resolved[f"{key} -> {form}"] = how
                else:
                    unresolved[key] = [v["form"] for v in variants]
            else:
                out.append(p.get("accented", p["string"]))

    return unicodedata.normalize("NFC", "".join(out)), unknown, resolved, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input file or - for stdin")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument("--no-tagger", action="store_true", help="skip UDPipe disambiguation")
    args = ap.parse_args()

    if args.input == "-":
        text = sys.stdin.read()
    else:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()

    stressed, unknown, resolved, unresolved = asyncio.run(
        accent_text(text, use_tagger=not args.no_tagger)
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(stressed)
    else:
        sys.stdout.buffer.write(stressed.encode("utf-8"))

    if unknown:
        print(f"\n[no accent data, left as-is] {', '.join(sorted(unknown))}", file=sys.stderr)
    if resolved:
        print("[ambiguous, auto-resolved by context]", file=sys.stderr)
        for k, how in sorted(resolved.items()):
            print(f"  {k}  ({how})", file=sys.stderr)
    if unresolved:
        print("[ambiguous, could not resolve - VDU default used]", file=sys.stderr)
        for w, vs in sorted(unresolved.items()):
            print(f"  {w}: {' | '.join(vs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
