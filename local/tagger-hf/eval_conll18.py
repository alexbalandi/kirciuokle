# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Run official CoNLL-18 UD metrics for the Lithuanian tagger."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Iterable

from coverage_diff import SLOT_FEATS_KEYS
from metrics import canonicalize_feats, parse_feats


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"
DEFAULT_GOLD = DEFAULT_RAW_DIR / "lt_alksnis-ud-test.conllu"
OFFICIAL_EVAL_URL = "https://universaldependencies.org/conll18/conll18_ud_eval.py"
OFFICIAL_EVAL_FILE = "conll18_ud_eval.py"
REPORT_METRICS = ("Tokens", "Words", "UPOS", "UFeats", "Lemmas")
VDU_UPOS_NORMALIZATION = {"DET": "PRON", "AUX": "VERB"}


def ensure_final_blank_line(text: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text


def download_official_eval(raw_dir: Path, force: bool) -> Path:
    destination = raw_dir / OFFICIAL_EVAL_FILE
    if destination.exists() and not force:
        return destination

    raw_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OFFICIAL_EVAL_URL,
        headers={"User-Agent": "kirciuokle-conll18-eval/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()

    with tempfile.NamedTemporaryFile(delete=False, dir=raw_dir, suffix=".tmp") as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    shutil.move(str(tmp_path), destination)
    return destination


def load_official_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("conll18_ud_eval_cached", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_conllu_text(module: ModuleType, text: str) -> object:
    return module.load_conllu(io.StringIO(ensure_final_blank_line(text)))


def evaluate_texts(module: ModuleType, gold_text: str, system_text: str) -> dict:
    gold_ud = load_conllu_text(module, gold_text)
    system_ud = load_conllu_text(module, system_text)
    return module.evaluate(gold_ud, system_ud)


def filtered_official_table(module: ModuleType, evaluation: dict) -> str:
    full_table = module.build_evaluation_table(evaluation, verbose=True, counts=False)
    lines = full_table.splitlines()
    selected = lines[:2]
    keep = set(REPORT_METRICS)
    for line in lines[2:]:
        metric = line.split("|", 1)[0].strip()
        if metric in keep:
            selected.append(line)
    return "\n".join(selected)


def project_feats(raw_feats: str) -> str:
    feats = parse_feats(raw_feats)
    restricted = {key: value for key, value in feats.items() if key in SLOT_FEATS_KEYS}
    return canonicalize_feats(
        "|".join(f"{key}={value}" for key, value in restricted.items())
    )


def project_conllu_vdu(text: str) -> str:
    lines: list[str] = []
    for raw_line in ensure_final_blank_line(text).splitlines():
        if not raw_line or raw_line.startswith("#"):
            lines.append(raw_line)
            continue
        columns = raw_line.split("\t")
        if len(columns) != 10:
            lines.append(raw_line)
            continue
        if columns[0].isdigit():
            columns[3] = VDU_UPOS_NORMALIZATION.get(columns[3], columns[3])
            columns[5] = project_feats(columns[5])
        lines.append("\t".join(columns))
    return ensure_final_blank_line("\n".join(lines))


def system_syntax_for_scorer(text: str) -> str:
    """Make non-scored dependency columns valid for morphology-only outputs."""
    output: list[str] = []
    sentence: list[list[str] | str] = []

    def flush() -> None:
        nonlocal sentence
        numeric_indices = [
            index
            for index, item in enumerate(sentence)
            if isinstance(item, list) and item[0].isdigit()
        ]
        if numeric_indices:
            root_id = sentence[numeric_indices[0]][0]  # type: ignore[index]
            for offset, index in enumerate(numeric_indices):
                columns = sentence[index]
                assert isinstance(columns, list)
                if offset == 0:
                    columns[6] = "0"
                    columns[7] = "root"
                else:
                    columns[6] = str(root_id)
                    columns[7] = "dep"
        for item in sentence:
            output.append("\t".join(item) if isinstance(item, list) else item)
        sentence = []

    for raw_line in ensure_final_blank_line(text).splitlines():
        if not raw_line:
            flush()
            output.append("")
            continue
        if raw_line.startswith("#"):
            sentence.append(raw_line)
            continue
        columns = raw_line.split("\t")
        sentence.append(columns if len(columns) == 10 else raw_line)
    flush()
    return ensure_final_blank_line("\n".join(output))


def read_gold_texts(gold_text: str) -> list[str]:
    texts: list[str] = []
    current_tokens: list[str] = []
    fallback_texts: list[str] = []

    def flush_tokens() -> None:
        nonlocal current_tokens
        if current_tokens:
            fallback_texts.append(" ".join(current_tokens))
            current_tokens = []

    for raw_line in gold_text.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            flush_tokens()
            continue
        if line.startswith("# text = "):
            texts.append(line[len("# text = ") :])
            continue
        if line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) == 10 and columns[0].isdigit():
            current_tokens.append(columns[1])
    flush_tokens()
    return texts or fallback_texts


def endpoint_for(url: str) -> str:
    stripped = url.rstrip("/")
    if stripped.endswith("/process") or stripped.endswith("/api/process"):
        return stripped
    return f"{stripped}/process"


def tag_with_udpipe_url(
    url: str,
    text: str,
    model: str,
    timeout: float,
) -> str:
    payload = urllib.parse.urlencode(
        {
            "data": text,
            "tokenizer": "",
            "tagger": "",
            "model": model,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint_for(url),
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "kirciuokle-conll18-eval/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_text = response.read().decode("utf-8")
    try:
        decoded = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("tagger response is not JSON") from exc
    if "result" not in decoded:
        raise RuntimeError(f"tagger response did not include 'result': {decoded!r}")
    return str(decoded["result"])


def read_system(args: argparse.Namespace, gold_text: str) -> tuple[str, str]:
    if args.system is not None:
        return args.system.read_text(encoding="utf-8"), str(args.system)

    sentence_texts = read_gold_texts(gold_text)
    if not sentence_texts:
        raise RuntimeError(f"no sentence texts found in {args.gold}")
    raw_text = "\n".join(sentence_texts)
    system_text = tag_with_udpipe_url(
        args.tagger_url,
        raw_text,
        model=args.model,
        timeout=args.timeout,
    )
    return system_text, endpoint_for(args.tagger_url)


def render_report(
    *,
    name: str,
    gold_path: Path,
    system_source: str,
    official_path: Path,
    official_table: str,
    projected_table: str,
    tagger_url: str | None,
) -> str:
    note = ""
    if tagger_url is not None:
        note = (
            "\nWhen `--tagger-url` is used, the gold `# text =` sentence text is "
            "sent through the tagger. Tokenization differences therefore count "
            "against Words, UPOS, and UFeats exactly as in the shared task.\n"
        )

    return "\n".join(
        [
            f"# CoNLL-18 UD Evaluation: {name}",
            "",
            f"- Gold: `{gold_path}`",
            f"- System: `{system_source}`",
            f"- Official scorer: `{official_path}`",
            note.rstrip(),
            "",
            "## Official CoNLL-18 UD Table",
            "",
            "Unmodified gold and system morphology; no VDU convention projection.",
            "",
            "```text",
            official_table,
            "```",
            "",
            "## VDU-convention projection (this project's metric)",
            "",
            "Gold and system are both projected with DET->PRON, AUX->VERB, and "
            "FEATS restricted to the project scoring slots before scoring.",
            "",
            "```text",
            projected_table,
            "```",
            "",
        ]
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Lithuanian tagger with the official CoNLL-18 UD scorer "
            "and a separate VDU-convention projection."
        )
    )
    parser.add_argument("--name", required=True, help="report name")
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="gold ALKSNIS test CoNLL-U file",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--system", type=Path, help="system CoNLL-U file")
    source_group.add_argument(
        "--tagger-url",
        help="UDPipe REST endpoint or base URL; receives gold # text sentences",
    )
    parser.add_argument(
        "--model",
        default="lithuanian-alksnis",
        help="UDPipe model form value when --tagger-url is used",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--force-download-official",
        action="store_true",
        help="refresh cached conll18_ud_eval.py",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.gold.exists():
        raise FileNotFoundError(
            f"missing gold file {args.gold}; run local/tagger-hf/fetch_corpora.py first"
        )

    official_path = download_official_eval(args.raw_dir, args.force_download_official)
    official = load_official_module(official_path)
    gold_text = args.gold.read_text(encoding="utf-8")
    system_text, system_source = read_system(args, gold_text)

    scorer_system = system_syntax_for_scorer(system_text)
    official_evaluation = evaluate_texts(official, gold_text, scorer_system)
    official_table = filtered_official_table(official, official_evaluation)

    projected_gold = project_conllu_vdu(gold_text)
    projected_system = project_conllu_vdu(scorer_system)
    projected_evaluation = evaluate_texts(official, projected_gold, projected_system)
    projected_table = filtered_official_table(official, projected_evaluation)

    report = render_report(
        name=args.name,
        gold_path=args.gold,
        system_source=system_source,
        official_path=official_path,
        official_table=official_table,
        projected_table=projected_table,
        tagger_url=args.tagger_url,
    )
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.runs_dir / f"conll18-{args.name}.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
