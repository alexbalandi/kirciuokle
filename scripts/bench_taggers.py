# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Benchmark Lithuanian taggers against UD_Lithuanian-ALKSNIS test data.

Examples:
    uv run scripts/bench_taggers.py --backends lindat --limit 20
    uv run --with stanza scripts/bench_taggers.py --backends stanza --limit 100
    uv run --python 3.11 --with trankit scripts/bench_taggers.py --backends trankit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


UD_TEST_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Lithuanian-ALKSNIS/master/lt_alksnis-ud-test.conllu"
)
LINDAT_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL = "lithuanian-alksnis"
SCAN_AHEAD = 8
SCORING_SLOTS = ("case", "gender", "number", "tense", "person", "voice", "degree")


@dataclass(frozen=True)
class Token:
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict[str, str]


@dataclass(frozen=True)
class Sentence:
    text: str
    tokens: list[Token]


@dataclass
class Counts:
    gold_tokens: int = 0
    aligned: int = 0
    upos_ok: int = 0
    lemma_ok: int = 0
    feats_ok: int = 0
    slots_ok: int = 0
    aux_verb_total: int = 0
    aux_verb_ok: int = 0


@dataclass
class Result:
    name: str
    sentences: int
    counts: Counts
    elapsed: float
    examples: list[str]


class Backend(Protocol):
    name: str

    def tag(self, texts: list[str]) -> list[Token]:
        ...


class BackendUnavailable(RuntimeError):
    pass


def parse_feats(raw: str | None) -> dict[str, str]:
    if not raw or raw == "_":
        return {}
    feats: dict[str, str] = {}
    for feature in raw.split("|"):
        separator = feature.find("=")
        if separator <= 0:
            continue
        feats[feature[:separator]] = feature[separator + 1 :]
    return feats


def feats_string(feats: dict[str, str]) -> str:
    if not feats:
        return "_"
    return "|".join(f"{key}={feats[key]}" for key in sorted(feats))


def parse_conllu_sentences(conllu: str) -> list[Sentence]:
    sentences: list[Sentence] = []
    text = ""
    tokens: list[Token] = []

    def flush() -> None:
        nonlocal text, tokens
        if tokens:
            sentence_text = text or " ".join(token.form for token in tokens)
            sentences.append(Sentence(text=sentence_text, tokens=tokens))
        text = ""
        tokens = []

    for raw_line in conllu.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            flush()
            continue
        if line.startswith("# text = "):
            text = line[len("# text = ") :]
            continue
        if line.startswith("#"):
            continue

        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue
        tokens.append(
            Token(
                form=columns[1],
                lemma=columns[2],
                upos=columns[3],
                xpos=columns[4],
                feats=parse_feats(columns[5]),
            )
        )

    flush()
    return sentences


def parse_conllu_tokens(conllu: str) -> list[Token]:
    tokens: list[Token] = []
    for sentence in parse_conllu_sentences(conllu):
        tokens.extend(sentence.tokens)
    return tokens


def has_letter(value: str) -> bool:
    return any(character.isalpha() for character in value)


def align_tokens(gold: list[Token], predicted: list[Token]) -> list[tuple[Token, Token | None]]:
    gold = [token for token in gold if has_letter(token.form)]
    predicted = [token for token in predicted if has_letter(token.form)]
    aligned: list[tuple[Token, Token | None]] = []
    predicted_index = 0

    for gold_token in gold:
        found: Token | None = None
        scan_end = min(predicted_index + SCAN_AHEAD, len(predicted))
        for index in range(predicted_index, scan_end):
            if predicted[index].form.lower() == gold_token.form.lower():
                found = predicted[index]
                predicted_index = index + 1
                break
        aligned.append((gold_token, found))

    return aligned


def token_tags(token: Token) -> dict[str, str]:
    """Faithful port of tokenTags in src/worker/disambiguation.ts."""
    tags: dict[str, str] = {}

    if token.upos in ("VERB", "AUX"):
        tags["pos"] = "PART_VERB" if token.feats.get("VerbForm") == "Part" else "VERB"
    elif token.upos in ("NOUN", "PROPN"):
        tags["pos"] = "NOUN"
    elif token.upos == "DET":
        # POS family follows VDU conventions: no DET in Lithuanian traditional grammar; see docs/SPEC13.md.
        tags["pos"] = "PRON"
    elif token.upos in ("CCONJ", "SCONJ"):
        tags["pos"] = "CCONJ"
    else:
        tags["pos"] = token.upos

    for slot, feature in (
        ("gender", "Gender"),
        ("number", "Number"),
        ("case", "Case"),
        ("tense", "Tense"),
        ("person", "Person"),
        ("voice", "Voice"),
    ):
        value = token.feats.get(feature)
        if value:
            tags[slot] = value

    degree = token.feats.get("Degree")
    if degree and degree != "Pos":
        tags["degree"] = degree

    return tags


def format_tags(tags: dict[str, str]) -> str:
    order = ("pos",) + SCORING_SLOTS
    parts = [f"{slot}={tags[slot]}" for slot in order if slot in tags]
    return "{" + ",".join(parts) + "}"


def evaluate(name: str, sentences: list[Sentence], predicted: list[Token], elapsed: float) -> Result:
    gold_tokens = [token for sentence in sentences for token in sentence.tokens]
    counts = Counts()
    examples: list[str] = []

    for gold, pred in align_tokens(gold_tokens, predicted):
        counts.gold_tokens += 1
        if pred is None:
            continue

        counts.aligned += 1
        if pred.upos == gold.upos:
            counts.upos_ok += 1
        if pred.lemma.casefold() == gold.lemma.casefold():
            counts.lemma_ok += 1
        if pred.feats == gold.feats:
            counts.feats_ok += 1

        gold_slots = token_tags(gold)
        pred_slots = token_tags(pred)
        if pred_slots == gold_slots:
            counts.slots_ok += 1
        elif len(examples) < 5:
            examples.append(
                f"{gold.form}: gold {format_tags(gold_slots)} pred {format_tags(pred_slots)}"
            )

        if gold.upos in ("AUX", "VERB"):
            counts.aux_verb_total += 1
            if pred.upos == gold.upos:
                counts.aux_verb_ok += 1

    return Result(
        name=name,
        sentences=len(sentences),
        counts=counts,
        elapsed=elapsed,
        examples=examples,
    )


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tagger-bench/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def load_gold(scratch_dir: Path, limit: int | None) -> list[Sentence]:
    conllu_path = scratch_dir / "lt_alksnis-ud-test.conllu"
    if not conllu_path.exists():
        print(f"downloading gold test set to {conllu_path}", file=sys.stderr)
        download(UD_TEST_URL, conllu_path)

    sentences = parse_conllu_sentences(conllu_path.read_text(encoding="utf-8"))
    if limit is not None:
        sentences = sentences[:limit]
    return sentences


class LindatBackend:
    name = "lindat"

    def __init__(self, url: str, batch_size: int) -> None:
        self.url = url
        self.batch_size = batch_size

    def tag(self, texts: list[str]) -> list[Token]:
        tokens: list[Token] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if start:
                time.sleep(0.2)
            conllu = self._post("\n".join(batch))
            tokens.extend(parse_conllu_tokens(conllu))
        return tokens

    def _post(self, text: str) -> str:
        payload = urllib.parse.urlencode(
            {
                "tokenizer": "",
                "tagger": "",
                "model": UDPIPE_MODEL,
                "data": text,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "tagger-bench/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
        payload_json = json.loads(body)
        result = payload_json.get("result")
        if not isinstance(result, str):
            raise RuntimeError(f"LINDAT response missing result: {payload_json!r}")
        return result


class StanzaBackend:
    name = "stanza"

    def __init__(self) -> None:
        try:
            import stanza  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable(
                "install with: uv run --with stanza scripts/bench_taggers.py ..."
            ) from exc

        try:
            stanza.download(
                "lt",
                processors="tokenize,pos,lemma",
                package="alksnis",
                verbose=False,
            )
            pipeline_kwargs: dict = {}
            resources_dir = os.getenv("STANZA_RESOURCES_DIR")
            if resources_dir:
                pipeline_kwargs["dir"] = resources_dir
            self.pipeline = stanza.Pipeline(
                "lt",
                processors="tokenize,pos,lemma",
                package="alksnis",
                use_gpu=False,
                verbose=False,
                **pipeline_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on optional model download
            raise BackendUnavailable(f"could not initialize stanza: {exc}") from exc

    def tag(self, texts: list[str]) -> list[Token]:
        doc = self.pipeline("\n".join(texts))
        tokens: list[Token] = []
        for sentence in doc.sentences:
            for word in sentence.words:
                tokens.append(
                    Token(
                        form=word.text or "_",
                        lemma=word.lemma or "_",
                        upos=word.upos or "_",
                        xpos=word.xpos or "_",
                        feats=parse_feats(word.feats or "_"),
                    )
                )
        return tokens


class TrankitBackend:
    name = "trankit"

    def __init__(self) -> None:
        try:
            import trankit  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable(
                "install with: uv run --python 3.11 --with trankit "
                "scripts/bench_taggers.py ..."
            ) from exc

        try:
            self.pipeline = trankit.Pipeline("lithuanian")
        except Exception as exc:  # pragma: no cover - depends on optional package/model
            raise BackendUnavailable(f"could not initialize trankit: {exc}") from exc

    def tag(self, texts: list[str]) -> list[Token]:
        document = self.pipeline("\n".join(texts))
        tokens: list[Token] = []
        for sentence in document.get("sentences", []):
            for token in sentence.get("tokens", []):
                words = token.get("expanded") or token.get("words") or [token]
                for word in words:
                    tokens.append(
                        Token(
                            form=word.get("text") or word.get("form") or "_",
                            lemma=word.get("lemma") or "_",
                            upos=word.get("upos") or "_",
                            xpos=word.get("xpos") or "_",
                            feats=parse_feats(word.get("feats") or "_"),
                        )
                    )
        return tokens


def build_backend(name: str, args: argparse.Namespace) -> Backend:
    if name == "lindat":
        return LindatBackend(args.lindat_url, args.lindat_batch_size)
    if name == "stanza":
        return StanzaBackend()
    if name == "trankit":
        return TrankitBackend()
    raise ValueError(f"unknown backend: {name}")


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:5.1f}%"


def rate(tokens: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "n/a"
    return f"{tokens / elapsed:7.1f}"


def print_results(results: list[Result], unavailable: list[tuple[str, str]]) -> None:
    print(
        "backend  sents  aligned  upos    lemma   feats   slots   aux/v   tok/s"
    )
    print(
        "-------  -----  -------  ------  ------  ------  ------  ------  -------"
    )
    for result in results:
        counts = result.counts
        print(
            f"{result.name:<7}  "
            f"{result.sentences:>5}  "
            f"{percent(counts.aligned, counts.gold_tokens):>7}  "
            f"{percent(counts.upos_ok, counts.aligned):>6}  "
            f"{percent(counts.lemma_ok, counts.aligned):>6}  "
            f"{percent(counts.feats_ok, counts.aligned):>6}  "
            f"{percent(counts.slots_ok, counts.aligned):>6}  "
            f"{percent(counts.aux_verb_ok, counts.aux_verb_total):>6}  "
            f"{rate(counts.gold_tokens, result.elapsed):>7}"
        )

    if unavailable:
        print()
        for name, reason in unavailable:
            print(f"{name}: backend unavailable: {reason}")

    print()
    for result in results:
        examples = "; ".join(result.examples) if result.examples else "(none)"
        print(f"{result.name} slot mismatches: {examples}")


def parse_backend_list(value: str) -> list[str]:
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one backend is required")
    unknown = sorted(set(names) - {"lindat", "stanza", "trankit"})
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown backend(s): {', '.join(unknown)}")
    return names


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        type=parse_backend_list,
        default=parse_backend_list("lindat,stanza,trankit"),
        help="comma-separated backends: lindat,stanza,trankit",
    )
    parser.add_argument("--limit", type=int, default=400, help="sentence limit")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(".scratch") / "tagger-bench",
        help="download/cache directory",
    )
    parser.add_argument("--lindat-url", default=LINDAT_URL)
    parser.add_argument(
        "--lindat-batch-size",
        type=int,
        default=40,
        help="sentences per LINDAT REST request",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.lindat_batch_size < 1:
        parser.error("--lindat-batch-size must be positive")

    sentences = load_gold(args.scratch_dir, args.limit)
    texts = [sentence.text for sentence in sentences]

    results: list[Result] = []
    unavailable: list[tuple[str, str]] = []

    for backend_name in args.backends:
        print(f"running {backend_name} on {len(sentences)} sentences ...", file=sys.stderr)
        try:
            backend = build_backend(backend_name, args)
        except BackendUnavailable as exc:
            unavailable.append((backend_name, str(exc)))
            continue

        started = time.perf_counter()
        predicted = backend.tag(texts)
        elapsed = time.perf_counter() - started
        results.append(evaluate(backend.name, sentences, predicted, elapsed))

    print_results(results, unavailable)
    return 0 if results or unavailable else 1


if __name__ == "__main__":
    raise SystemExit(main())
