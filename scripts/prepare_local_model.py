# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "onnx",
#   "onnxscript",
#   "onnxruntime",
#   "sentencepiece",
#   "torch",
#   "transformers<5",
# ]
# ///
"""Build the local browser bundle and report quantization size/parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "local-model"
RUNTIME_DIR = MODEL_DIR / "runtime"
ACCENTUATOR_DIR = REPO_ROOT / "local" / "accentuator"
APP_DIR = REPO_ROOT / "local" / "app"
RELEASE_DIR = REPO_ROOT / "local" / "accentuator" / "joint" / "hf_release"
JOINT_DIR = REPO_ROOT / "local" / "accentuator" / "joint"
CORPUS = REPO_ROOT / "local" / "accentuator" / "data" / "eval" / "lrt-smoke.txt"
GENERATED_DB = ACCENTUATOR_DIR / "data" / "generated.sqlite"
I18N_SOURCE = REPO_ROOT / "src" / "client" / "i18n.ts"
I18N_MODULE = MODEL_DIR / "i18n.js"
BASELINE_ONNX = RELEASE_DIR / "joint.int8.onnx"
FP32_ONNX = RELEASE_DIR / "joint.onnx"
META_JSON = RELEASE_DIR / "joint.meta.json"
FULL_INT8_NAME = "joint.full-int8.onnx"
BASELINE_NAME = "joint.int8.onnx"
LABEL_BRIDGE_NAME = "label_bridge.json"
PARITY_GATE = 0.97
ORT_WEB_VERSION = "1.22.0"
TRANSFORMERS_JS_VERSION = "3.7.1"
ORT_WEB_TARBALL = (
    "https://registry.npmjs.org/onnxruntime-web/-/"
    f"onnxruntime-web-{ORT_WEB_VERSION}.tgz"
)
TRANSFORMERS_JS_TARBALL = (
    "https://registry.npmjs.org/@huggingface/transformers/-/"
    f"transformers-{TRANSFORMERS_JS_VERSION}.tgz"
)
RUNTIME_PACKAGES = (
    {
        "package": "onnxruntime-web",
        "version": ORT_WEB_VERSION,
        "url": ORT_WEB_TARBALL,
        "files": (
            ("dist/ort.min.mjs", "ort.min.mjs"),
            ("dist/ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.mjs"),
            ("dist/ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.wasm"),
            ("dist/ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.mjs"),
            ("dist/ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.jsep.wasm"),
        ),
    },
    {
        "package": "@huggingface/transformers",
        "version": TRANSFORMERS_JS_VERSION,
        "url": TRANSFORMERS_JS_TARBALL,
        "files": (("dist/transformers.min.js", "transformers.min.js"),),
    },
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)

for import_path in (JOINT_DIR, ACCENTUATOR_DIR, APP_DIR):
    sys.path.insert(0, str(import_path))
from export_joint_onnx import (  # noqa: E402
    JointExportWrapper,
    agreement,
    load_joint,
    load_parity_rows,
    make_collator,
    make_onnx_session,
    onnx_decisions,
    quantize_int8,
    torch_decisions,
)
from eval_nodict_pipeline import load_generated as load_label_vocabulary  # noqa: E402
from kirciuokle import disambiguate as disamb  # noqa: E402


@dataclass
class Parity:
    pos_matches: int
    stress_matches: int
    tokens: int
    pos_rate: float
    stress_rate: float


@dataclass
class TableRow:
    name: str
    file_name: str | None
    bytes: int | None
    pos_rate: float | None
    stress_rate: float | None
    shipped: bool
    note: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--parity-sentences", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    # Source overrides: ship a different int8 model + its matching tokenizer.
    # Defaults reproduce the hf_release baseline (joint_v2_literary). To ship
    # the pruned joint_v3 bundle:
    #   --model-onnx local/accentuator/joint/pruned/onnx/joint.int8.partial.onnx
    #   --model-name joint.int8.partial.onnx
    #   --meta-json  local/accentuator/joint/pruned/onnx/joint.meta.json
    #   --tokenizer-dir local/accentuator/joint/pruned/tokenizer
    parser.add_argument("--model-onnx", type=Path, default=BASELINE_ONNX)
    parser.add_argument("--model-name", type=str, default=BASELINE_NAME)
    parser.add_argument("--meta-json", type=Path, default=META_JSON)
    parser.add_argument("--tokenizer-dir", type=Path, default=RELEASE_DIR)
    return parser


def copy_if_changed(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, target)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_member(tar: tarfile.TarFile, member_name: str, target: Path) -> None:
    member = tar.getmember(member_name)
    if not member.isfile():
        raise ValueError(f"{member_name} is not a file in {tar.name}")
    source = tar.extractfile(member)
    if source is None:
        raise ValueError(f"could not extract {member_name} from {tar.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def download_tgz(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    print(f"downloading runtime package: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def vendor_runtime(output_dir: Path) -> dict[str, object]:
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "path": "runtime/",
        "packages": {},
        "files": {},
    }

    with tempfile.TemporaryDirectory(prefix="bundled-weights-runtime-") as temp_name:
        temp_dir = Path(temp_name)
        for package in RUNTIME_PACKAGES:
            package_name = str(package["package"])
            version = str(package["version"])
            tgz_name = f"{package_name.replace('@', '').replace('/', '-')}-{version}.tgz"
            tgz_path = temp_dir / tgz_name
            download_tgz(str(package["url"]), tgz_path)
            package_sha = sha256_file(tgz_path)
            print(
                "runtime package: "
                f"{package_name}@{version} sha256={package_sha[:12]}..."
            )
            package_files: list[str] = []
            with tarfile.open(tgz_path, "r:gz") as archive:
                for source_name, target_name in package["files"]:
                    archive_name = f"package/{source_name}"
                    target = runtime_dir / str(target_name)
                    safe_extract_member(archive, archive_name, target)
                    file_sha = sha256_file(target)
                    package_files.append(str(target_name))
                    manifest["files"][str(target_name)] = {
                        "bytes": target.stat().st_size,
                        "package": package_name,
                        "sha256": file_sha,
                        "source": source_name,
                    }
                    print(
                        "runtime file: "
                        f"{target.relative_to(REPO_ROOT).as_posix()} "
                        f"size={format_bytes(target.stat().st_size)} "
                        f"sha256={file_sha[:12]}..."
                    )
            manifest["packages"][package_name] = {
                "version": version,
                "tarball": str(package["url"]),
                "tarball_sha256": package_sha,
                "files": package_files,
            }

    return manifest


def copy_static_model_files(
    output_dir: Path,
    model_onnx: Path = BASELINE_ONNX,
    model_name: str = BASELINE_NAME,
    meta_json: Path = META_JSON,
    tokenizer_dir: Path = RELEASE_DIR,
) -> None:
    copy_if_changed(model_onnx, output_dir / model_name)
    copy_if_changed(meta_json, output_dir / "joint.meta.json")
    # labels are prune-invariant (label ids unchanged) — always from hf_release
    labels = RELEASE_DIR / "labels.json"
    if labels.exists():
        copy_if_changed(labels, output_dir / "labels.json")
    for name in TOKENIZER_FILES:
        source = tokenizer_dir / name
        if source.exists():
            copy_if_changed(source, output_dir / name)


def parse_model_label(label: str) -> tuple[str, dict[str, str]]:
    upos, separator, raw_feats = str(label).partition("|")
    feats: dict[str, str] = {}
    if not separator or raw_feats == "_":
        return upos, feats
    for item in raw_feats.split("|"):
        key, value_separator, value = item.partition("=")
        if value_separator and key and value:
            feats[key] = value
    return upos, feats


def slots_for_model_label(label: str) -> dict[str, str]:
    upos, feats = parse_model_label(label)
    token = disamb.Token(form="", lemma="", upos=upos, xpos="_", feats=feats)
    return {str(slot): str(value) for slot, value in disamb.token_tags(token).items()}


def spurious_slots(variant_slots: dict[str, str], context_slots: dict[str, str]) -> int:
    return sum(1 for slot in variant_slots if slot not in context_slots)


def best_mi_label(
    context_slots: dict[str, str],
    candidates: Iterable[object],
) -> str:
    best_label = ""
    best_score: int | None = None
    best_spurious: int | None = None
    for candidate in candidates:
        label = str(getattr(candidate, "label"))
        slots = dict(getattr(candidate, "slots"))
        score = disamb.score_tags(slots, context_slots)
        spurious = spurious_slots(slots, context_slots)
        if (
            best_score is None
            or score > best_score
            or (score == best_score and (best_spurious is None or spurious < best_spurious))
            or (
                score == best_score
                and spurious == best_spurious
                and len(label) < len(best_label)
            )
        ):
            best_label = label
            best_score = score
            best_spurious = spurious
    return best_label


def write_label_bridge(output_dir: Path) -> None:
    if not GENERATED_DB.exists():
        raise FileNotFoundError(GENERATED_DB)

    meta = json.loads((output_dir / "joint.meta.json").read_text(encoding="utf-8"))
    model_labels = [str(label) for label in meta["labels"]]
    candidates, _entries, _slot_cache = load_label_vocabulary(GENERATED_DB, set())
    bridge = {
        "source": {
            "dictionary": str(GENERATED_DB.relative_to(REPO_ROOT)).replace("\\", "/"),
            "model_meta": "joint.meta.json",
            "parser": "local/app/kirciuokle/disambiguate.py",
            "label_vocabulary_builder": "local/accentuator/eval_nodict_pipeline.py",
        },
        "score_tags": {
            "pos_match": 4,
            "pos_mismatch": -3,
            "slot_match": 2,
            "slot_mismatch": -2,
            "one_sided_slots": "skip",
            "tie_break": "fewest_spurious_slots",
        },
        "mi_vocab": [
            {
                "label": str(candidate.label),
                "slots": {str(slot): str(value) for slot, value in candidate.slots.items()},
            }
            for candidate in candidates
        ],
        "model_labels": {
            label: slots_for_model_label(label)
            for label in model_labels
        },
    }
    target = output_dir / LABEL_BRIDGE_NAME
    target.write_text(
        json.dumps(bridge, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "label bridge: "
        f"mi_vocab={len(bridge['mi_vocab']):,}; "
        f"model_labels={len(bridge['model_labels']):,}; "
        f"size={format_bytes(target.stat().st_size)}"
    )
    print("sample bridge mappings:")
    preferred_samples = [
        "NOUN|Case=Gen|Gender=Fem|Number=Sing",
        "NOUN|Case=Nom|Gender=Masc|Number=Sing",
        "VERB|Mood=Ind|Number=Plur|Person=2|Tense=Pres|VerbForm=Fin",
        "PRON|Case=Acc|Gender=Masc|Number=Sing|Person=3",
        "ADP|Case=Gen",
    ]
    sample_labels = [label for label in preferred_samples if label in model_labels]
    if len(sample_labels) < 5:
        sample_labels.extend(
            label
            for label in model_labels
            if label not in sample_labels
            and slots_for_model_label(label).get("pos")
            in {"NOUN", "VERB", "PRON", "ADP", "ADV"}
        )
    sample_labels = sample_labels[:5]
    for label in sample_labels:
        print(f"  {label} -> {best_mi_label(slots_for_model_label(label), candidates)}")


def find_balanced(text: str, start: int, opener: str, closer: str) -> str:
    depth = 0
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    for index in range(start, len(text)):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                block_comment = False
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char == "/" and nxt == "/":
            line_comment = True
            continue
        if char == "/" and nxt == "*":
            block_comment = True
            continue
        if char in ("'", '"', "`"):
            quote = char
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"could not find balanced {opener}{closer} block")


def extract_object_literal(source: str, export_name: str) -> str:
    marker = f"export const {export_name}"
    marker_index = source.index(marker)
    equals_index = source.index("=", marker_index)
    open_index = source.index("{", equals_index)
    return find_balanced(source, open_index, "{", "}")


def pilot_ui() -> dict[str, dict[str, str]]:
    return {
        "lt": {
            "title": "Kirčiuoklė",
            "eyebrow": "bundled weights pilot",
            "subtitle": "veikia tik jūsų naršyklėje — tekstas nesiunčiamas į serverį",
            "tokenBudgetLabel": "Tokenų biudžetas",
            "modelLoading": "Modelis kraunamas...",
            "modelMetadata": "Skaitoma modelio metainformacija...",
            "modelLoadFailed": "Modelio įkelti nepavyko",
            "modelLabel": "Modelis",
            "cacheLabel": "cache",
            "cachePresent": "yra",
            "cacheWillFill": "bus pildoma",
            "cacheStored": "įrašyta",
            "cacheFailed": "nepavyko",
            "cacheUnavailable": "nepasiekiama",
            "cacheMiss": "nėra",
            "wasmThreads": "WASM gijos",
            "creatingSession": "Kuriama ONNX sesija",
            "ready": "Paruošta",
            "readingCache": "Skaitoma iš Cache API",
            "downloading": "Atsisiunčiama",
            "unknownSize": "nežinomas dydis",
            "sentences": "Sakiniai",
            "batches": "partijos",
            "running": "vykdoma...",
            "batch": "partija",
            "tokensPerSecond": "tokenų/s",
            "done": "Baigta",
            "sentenceShort": "sak.",
            "tokens": "tokenų",
            "secondsShort": "s",
            "errorPrefix": "Klaida",
            "memoryLabel": "Atmintis",
            "wasmMemoryLabel": "WASM",
            "jsHeapMemoryLabel": "JS heap",
            "memoryLimitReached": "Pasiekta naršyklės atminties riba — įkelkite puslapį iš naujo.",
            "charsSuffix": "ženkl.",
            "popoverEmpty": "Morfologijos eilučių virš slenksčio nėra.",
            "noStressTitle": "Nekirčiuota / svetimžodis",
            "footer": "ONNX Runtime Web WASM · transformers.js tokenizer · lokali Cache API talpykla",
        },
        "en": {
            "title": "Lithuanian Stress Marker",
            "eyebrow": "bundled weights pilot",
            "subtitle": "runs fully in your browser — nothing is sent to a server",
            "tokenBudgetLabel": "Token budget",
            "modelLoading": "Loading model...",
            "modelMetadata": "Reading model metadata...",
            "modelLoadFailed": "Could not load the model",
            "modelLabel": "Model",
            "cacheLabel": "cache",
            "cachePresent": "ready",
            "cacheWillFill": "will fill",
            "cacheStored": "stored",
            "cacheFailed": "failed",
            "cacheUnavailable": "unavailable",
            "cacheMiss": "missing",
            "wasmThreads": "WASM threads",
            "creatingSession": "Creating ONNX session",
            "ready": "Ready",
            "readingCache": "Reading from Cache API",
            "downloading": "Downloading",
            "unknownSize": "unknown size",
            "sentences": "Sentences",
            "batches": "batches",
            "running": "running...",
            "batch": "batch",
            "tokensPerSecond": "tokens/s",
            "done": "Done",
            "sentenceShort": "sent.",
            "tokens": "tokens",
            "secondsShort": "s",
            "errorPrefix": "Error",
            "memoryLabel": "Memory",
            "wasmMemoryLabel": "WASM",
            "jsHeapMemoryLabel": "JS heap",
            "memoryLimitReached": "Memory limit reached — reload the page.",
            "charsSuffix": "chars",
            "popoverEmpty": "No morphology rows above the threshold.",
            "noStressTitle": "Not accented / foreign",
            "footer": "ONNX Runtime Web WASM · transformers.js tokenizer · local Cache API storage",
        },
        "ru": {
            "title": "Литовская акцентуация",
            "eyebrow": "bundled weights pilot",
            "subtitle": "работает полностью в браузере — текст не отправляется на сервер",
            "tokenBudgetLabel": "Бюджет токенов",
            "modelLoading": "Модель загружается...",
            "modelMetadata": "Чтение метаданных модели...",
            "modelLoadFailed": "Не удалось загрузить модель",
            "modelLabel": "Модель",
            "cacheLabel": "кэш",
            "cachePresent": "есть",
            "cacheWillFill": "будет заполнен",
            "cacheStored": "сохранён",
            "cacheFailed": "не удалось",
            "cacheUnavailable": "недоступен",
            "cacheMiss": "нет",
            "wasmThreads": "потоки WASM",
            "creatingSession": "Создание ONNX-сессии",
            "ready": "Готово",
            "readingCache": "Чтение из Cache API",
            "downloading": "Загрузка",
            "unknownSize": "размер неизвестен",
            "sentences": "Предложения",
            "batches": "пакеты",
            "running": "выполняется...",
            "batch": "пакет",
            "tokensPerSecond": "токенов/с",
            "done": "Готово",
            "sentenceShort": "предл.",
            "tokens": "токенов",
            "secondsShort": "с",
            "errorPrefix": "Ошибка",
            "memoryLabel": "Память",
            "wasmMemoryLabel": "WASM",
            "jsHeapMemoryLabel": "JS heap",
            "memoryLimitReached": "Достигнут предел памяти — перезагрузите страницу.",
            "charsSuffix": "зн.",
            "popoverEmpty": "Нет морфологических строк выше порога.",
            "noStressTitle": "Без ударения / иностранное",
            "footer": "ONNX Runtime Web WASM · токенизатор transformers.js · локальное хранилище Cache API",
        },
    }


def write_i18n_module() -> None:
    source = I18N_SOURCE.read_text(encoding="utf-8")
    ui_object = extract_object_literal(source, "UI")
    gloss_object = extract_object_literal(source, "MORPH_GLOSSES")
    module = f"""// Generated by scripts/prepare_local_model.py.
// Source of truth for production strings and morphology glosses: src/client/i18n.ts.
// Do not edit this file by hand.

export const LANGS = ["lt", "en", "ru"];

export const UI = {ui_object};

const G = (lt, en, ru) => ({{ lt, en, ru }});

export const MORPH_GLOSSES = {gloss_object};

export const PILOT_UI = {json.dumps(pilot_ui(), ensure_ascii=False, indent=2, sort_keys=True)};

const MORPH_KEYS = Object.keys(MORPH_GLOSSES).sort((a, b) => b.length - a.length);

function walkMorphPiece(piece) {{
  let rest = piece.trim();
  const tokens = [];

  while (rest.length > 0) {{
    const key = MORPH_KEYS.find(
      (candidate) =>
        rest.startsWith(candidate) &&
        (rest.length === candidate.length || rest[candidate.length] === " "),
    );

    if (key) {{
      tokens.push({{ text: key, gloss: MORPH_GLOSSES[key] }});
      rest = rest.slice(key.length).trimStart();
    }} else {{
      const space = rest.indexOf(" ");
      if (space === -1) {{
        tokens.push({{ text: rest }});
        rest = "";
      }} else {{
        tokens.push({{ text: rest.slice(0, space) }});
        rest = rest.slice(space + 1);
      }}
    }}
  }}

  return tokens;
}}

function segmentForToken(token, lang) {{
  if (!token.gloss) {{
    return {{ text: token.text }};
  }}

  const text = token.gloss[lang];
  return lang === "lt" ? {{ text }} : {{ text, lt: token.gloss.lt }};
}}

function readingSegments(reading, lang) {{
  const pieces = reading.split(", ");
  const segments = [];

  pieces.forEach((piece, pieceIndex) => {{
    if (pieceIndex > 0) {{
      segments.push({{ text: ", " }});
    }}

    walkMorphPiece(piece).forEach((token, tokenIndex) => {{
      if (tokenIndex > 0) {{
        segments.push({{ text: " " }});
      }}
      segments.push(segmentForToken(token, lang));
    }});
  }});

  return segments;
}}

export function morphologySegments(info, lang) {{
  if (!info) {{
    return [];
  }}

  const segments = [];

  info.split("; ").forEach((segment, segmentIndex) => {{
    if (segmentIndex > 0) {{
      segments.push({{ text: "; " }});
    }}

    const dash = segment.indexOf(" - ");
    const mi = dash === -1 ? segment : segment.slice(0, dash);
    const meaning = dash === -1 ? "" : segment.slice(dash);
    segments.push(...readingSegments(mi, lang));

    if (meaning) {{
      segments.push({{ text: meaning }});
    }}
  }});

  return segments;
}}

export function translateMorphology(info, lang) {{
  return morphologySegments(info, lang)
    .map((segment) => segment.text)
    .join("");
}}

export function detectLang() {{
  const stored = localStorage.getItem("lang");
  if (stored === "lt" || stored === "en" || stored === "ru") {{
    return stored;
  }}

  const nav = navigator.language?.toLowerCase() ?? "";
  if (nav.startsWith("lt")) {{
    return "lt";
  }}
  if (nav.startsWith("ru")) {{
    return "ru";
  }}
  return "en";
}}
"""
    I18N_MODULE.write_text(module, encoding="utf-8", newline="\n")
    print(
        "i18n module: "
        f"{I18N_MODULE.relative_to(REPO_ROOT).as_posix()} "
        f"size={format_bytes(I18N_MODULE.stat().st_size)}"
    )


def run_parity(
    loaded: object,
    onnx_path: Path,
    sentence_count: int,
    batch_size: int,
) -> Parity:
    rows = load_parity_rows(CORPUS, sentence_count)
    collator = make_collator(loaded)
    wrapper = JointExportWrapper(loaded.model).eval()
    session = make_onnx_session(onnx_path)
    torch_pos: list[int] = []
    torch_stress: list[int] = []
    onnx_pos: list[int] = []
    onnx_stress: list[int] = []

    for start in range(0, len(rows), batch_size):
        batch = collator(rows[start : start + batch_size])
        pos, stress = torch_decisions(wrapper, batch)
        torch_pos.extend(pos)
        torch_stress.extend(stress)
        pos, stress = onnx_decisions(session, batch)
        onnx_pos.extend(pos)
        onnx_stress.extend(stress)

    pos_matches, total, pos_rate = agreement(torch_pos, onnx_pos)
    stress_matches, stress_total, stress_rate = agreement(torch_stress, onnx_stress)
    if total != stress_total:
        raise RuntimeError("POS/stress parity token counts differ")
    return Parity(
        pos_matches=pos_matches,
        stress_matches=stress_matches,
        tokens=total,
        pos_rate=pos_rate,
        stress_rate=stress_rate,
    )


def maybe_build_full_int8(output_dir: Path, force: bool) -> tuple[Path | None, str]:
    full_path = output_dir / FULL_INT8_NAME
    if full_path.exists() and not force:
        return full_path, "reused existing full dynamic int8"
    try:
        quantize_int8(FP32_ONNX, full_path, per_channel=False, encoder_layers=-1)
        return full_path, "full dynamic int8"
    except Exception as exc:  # noqa: BLE001
        if full_path.exists():
            full_path.unlink()
        return None, f"full dynamic int8 failed: {exc}"


def make_row(
    name: str,
    path: Path | None,
    parity: Parity | None,
    shipped: bool,
    note: str,
    existing_models: dict[str, object] | None = None,
) -> TableRow:
    existing = (
        existing_models.get(path.name, {})
        if existing_models and path and path.name in existing_models
        else {}
    )
    return TableRow(
        name=name,
        file_name=path.name if path else None,
        bytes=path.stat().st_size if path and path.exists() else None,
        pos_rate=parity.pos_rate if parity else existing.get("pos_parity"),
        stress_rate=parity.stress_rate if parity else existing.get("stress_parity"),
        shipped=shipped,
        note=note,
    )


def print_table(rows: Iterable[TableRow]) -> None:
    print()
    print("| artifact | file | size | POS parity | stress parity | default | note |")
    print("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row.name,
                    row.file_name or "-",
                    format_bytes(row.bytes),
                    format_rate(row.pos_rate),
                    format_rate(row.stress_rate),
                    "yes" if row.shipped else "no",
                    row.note,
                ]
            )
            + " |"
        )
    print()


def write_manifest(
    output_dir: Path,
    rows: list[TableRow],
    default_model: str,
    runtime: dict[str, object],
) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "default_model": default_model,
        "parity_gate": PARITY_GATE,
        "models": {
            row.file_name: {
                "artifact": row.name,
                "bytes": row.bytes,
                "pos_parity": row.pos_rate,
                "stress_parity": row.stress_rate,
                "default": row.shipped,
                "note": row.note,
            }
            for row in rows
            if row.file_name
        },
        "runtime": runtime,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    mib = size / (1024 * 1024)
    return f"{size:,} B / {mib:.1f} MiB"


def format_rate(rate: float | None) -> str:
    return "-" if rate is None else f"{rate:.2%}"


def main(argv: Iterable[str] | None = None) -> int:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_dir / "manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.exists()
        else {}
    )
    existing_models = existing_manifest.get("models", {})

    print(f"copying model bundle files to {output_dir}")
    copy_static_model_files(
        output_dir,
        model_onnx=args.model_onnx,
        model_name=args.model_name,
        meta_json=args.meta_json,
        tokenizer_dir=args.tokenizer_dir,
    )
    write_label_bridge(output_dir)
    write_i18n_module()
    runtime_manifest = vendor_runtime(output_dir)

    loaded = None
    baseline_parity = None
    full_parity = None
    if not args.skip_parity:
        print(f"loading Torch checkpoint on CPU for {args.parity_sentences} sentence parity")
        loaded = load_joint(JOINT_DIR / "checkpoints" / "joint_v2_literary.best.pt")
        print("measuring partial-int8 baseline parity")
        baseline_parity = run_parity(
            loaded,
            output_dir / BASELINE_NAME,
            args.parity_sentences,
            args.batch_size,
        )

    full_path = None
    full_note = "skipped"
    if not args.skip_full:
        full_path, full_note = maybe_build_full_int8(output_dir, args.force_full)
        if full_path and not args.skip_parity:
            if loaded is None:
                loaded = load_joint(JOINT_DIR / "checkpoints" / "joint_v2_literary.best.pt")
            print("measuring full dynamic-int8 parity")
            full_parity = run_parity(
                loaded,
                full_path,
                args.parity_sentences,
                args.batch_size,
            )

    default_model = args.model_name
    full_ships = bool(
        full_path
        and full_parity
        and full_parity.pos_rate >= PARITY_GATE
        and full_parity.stress_rate >= PARITY_GATE
    )
    if full_ships:
        default_model = FULL_INT8_NAME

    rows = [
        make_row(
            "shipped int8 model",
            output_dir / args.model_name,
            baseline_parity,
            default_model == args.model_name,
            f"source: {args.model_onnx.name}",
            existing_models,
        )
    ]
    if full_path:
        rows.append(
            make_row(
                "full dynamic int8",
                full_path,
                full_parity,
                full_ships,
                "ships by >=97% parity gate" if full_ships else "below >=97% parity gate",
                existing_models,
            )
        )
    elif not args.skip_full:
        rows.append(make_row("full dynamic int8", None, None, False, full_note, existing_models))

    write_manifest(output_dir, rows, default_model, runtime_manifest)
    print_table(rows)
    print(f"default model: {default_model}")
    print(f"manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
