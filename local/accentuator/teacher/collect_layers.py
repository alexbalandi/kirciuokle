from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
JOINT_DIR = ACCENTUATOR_DIR / "joint"
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"
APP_DIR = LOCAL_DIR / "app"

for _path in (SCRIPT_DIR, ACCENTUATOR_DIR, JOINT_DIR, TAGGER_DIR, APP_DIR):
    sys.path.insert(0, str(_path))

from _common import DEFAULT_GENERATED, normalize_lt, safe_relative, strip_accents  # noqa: E402
import eval_chrestomatija as chrest  # noqa: E402
import eval_nodict_pipeline as nodict  # noqa: E402
import eval_joint as joint_eval  # noqa: E402
from kirciuokle import disambiguate as disamb  # noqa: E402
from metrics import combined_label, feats_string  # noqa: E402


DEFAULT_CHECKPOINT = JOINT_DIR / "checkpoints" / "joint_v1_polish.best.pt"
DEFAULT_OUTPUT_DIR = ACCENTUATOR_DIR / "data" / "teacher"
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class SilverLayerToken:
    word: str
    accented: str
    mi: str | None
    ambiguous: bool
    ud: dict[str, str] | None = None


def resolve_input_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    for base in (ACCENTUATOR_DIR, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return ACCENTUATOR_DIR / path
    return path


def default_layers_path(corpus_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{corpus_path.stem}.layers.jsonl"


def word_key(text: str | None) -> str:
    return strip_accents(normalize_lt(text or "")).casefold()


def norm_form(text: str | None) -> str:
    return normalize_lt(text or "").casefold()


def answer_form(word: str, predicted: str | None) -> str | None:
    if predicted is None:
        return None
    key = word_key(word)
    form = key if predicted == "" else norm_form(predicted)
    return form if word_key(form) == key else None


def layer_form(layer: dict[str, Any] | None) -> str | None:
    if not isinstance(layer, dict):
        return None
    form = layer.get("form")
    return str(form) if form else None


def layer_pos(layer: dict[str, Any] | None) -> str | None:
    if not isinstance(layer, dict):
        return None
    label = layer.get("pos_label")
    return str(label) if label else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_silver_layer_tokens(path: Path) -> list[SilverLayerToken]:
    tokens: list[SilverLayerToken] = []
    skipped_bad = 0
    total_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            total_rows += 1
            word = word_key(raw.get("word"))
            accented = norm_form(raw.get("accented"))
            if not word or not accented:
                # exotic-Unicode tokenizer debris (bare combining marks from
                # Wikipedia etymology glosses etc.) — skip, but refuse a
                # silver file that is broken wholesale
                skipped_bad += 1
                if skipped_bad > max(200, total_rows // 100):
                    raise ValueError(
                        f"too many bad silver rows ({skipped_bad}) by {path}:{line_number}"
                    )
                continue

            raw_ud = raw.get("ud") if "ud" in raw else None
            ud: dict[str, str] | None = None
            if isinstance(raw_ud, dict) and raw_ud.get("upos"):
                ud = {
                    "upos": str(raw_ud.get("upos") or "X"),
                    "feats": str(raw_ud.get("feats") or "_"),
                }
            tokens.append(
                SilverLayerToken(
                    word=word,
                    accented=accented,
                    mi=raw.get("mi") or None,
                    ambiguous=bool(raw.get("ambiguous")),
                    ud=ud,
                )
            )
    if skipped_bad:
        print(f"silver loader: skipped {skipped_bad} bad rows in {path}")
    return tokens


def load_layer_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def write_jsonl_append(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def valid_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {path}:{line_number}: {exc}") from exc
            count += 1
    return count


def rows_from_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = normalize_lt(line.strip())
            if not text:
                continue
            words = chrest.tokenized_words(text)
            if not words:
                continue
            rows.append(
                {
                    "id": f"{corpus_path.stem}-{line_number}",
                    "source": corpus_path.stem,
                    "text": text,
                    "tokens": [{"word": word} for word in words],
                }
            )
    add_token_metadata(rows)
    return rows


def rows_from_pos_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_jsonl(path):
        tokens: list[dict[str, Any]] = []
        for index, word in enumerate(raw.get("tokens") or []):
            text = str(word)
            if not nodict.has_letter(text):
                continue
            tokens.append({"word": text, "source_token_index": index})
        if not tokens:
            continue
        rows.append(
            {
                "id": str(raw.get("id") or f"{path.stem}-{len(rows) + 1}"),
                "source": path.stem,
                "text": str(raw.get("text") or " ".join(str(t["word"]) for t in tokens)),
                "tokens": tokens,
            }
        )
    add_token_metadata(rows)
    return rows


def add_token_metadata(rows: list[dict[str, Any]]) -> None:
    global_index = 0
    for sentence_index, row in enumerate(rows):
        row.setdefault("sent_id", sentence_index)
        for token_index, token in enumerate(row.get("tokens") or []):
            token["token_index"] = token_index
            token["global_index"] = global_index
            token["key"] = word_key(str(token.get("word") or ""))
            global_index += 1


def flatten_tokens(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [token for row in rows for token in row.get("tokens", [])]


def align_by_key(
    source_keys: list[str],
    items: list[Any],
    item_key: Callable[[Any], str],
    window: int = 8,
) -> tuple[list[Any | None], int, int]:
    aligned: list[Any | None] = [None] * len(source_keys)
    item_keys = [item_key(item) for item in items]
    skipped_source = 0
    skipped_items = 0
    source_index = 0
    item_index = 0

    while source_index < len(source_keys):
        target = source_keys[source_index]
        if item_index < len(items) and item_keys[item_index] == target:
            aligned[source_index] = items[item_index]
            source_index += 1
            item_index += 1
            continue

        found_item = None
        for lookahead in range(item_index + 1, min(len(items), item_index + window + 1)):
            if item_keys[lookahead] == target:
                found_item = lookahead
                break

        found_source = None
        if item_index < len(items):
            current = item_keys[item_index]
            for lookahead in range(
                source_index + 1,
                min(len(source_keys), source_index + window + 1),
            ):
                if source_keys[lookahead] == current:
                    found_source = lookahead
                    break

        if found_item is not None and (
            found_source is None or found_item - item_index <= found_source - source_index
        ):
            skipped_items += found_item - item_index
            item_index = found_item
            continue
        if found_source is not None:
            skipped_source += found_source - source_index
            source_index = found_source
            continue

        skipped_source += 1
        source_index += 1

    skipped_items += max(0, len(items) - item_index)
    return aligned, skipped_source, skipped_items


def load_vdu_alignment(
    rows: list[dict[str, Any]],
    silver_path: Path | None,
) -> tuple[dict[int, SilverLayerToken], tuple[int, int]]:
    if silver_path is None:
        return {}, (0, 0)
    silver = load_silver_layer_tokens(silver_path)
    tokens = flatten_tokens(rows)
    aligned, skipped_tokens, skipped_silver = align_by_key(
        [str(token["key"]) for token in tokens],
        silver,
        lambda item: str(item.word),
    )
    by_global: dict[int, SilverLayerToken] = {}
    for token, silver_token in zip(tokens, aligned):
        if silver_token is not None:
            by_global[int(token["global_index"])] = silver_token
    return by_global, (skipped_tokens, skipped_silver)


def udpipe_slots(ud: dict[str, str] | None, form: str = "") -> dict[str, str] | None:
    if not ud:
        return None
    upos = str(ud.get("upos") or "")
    if not upos:
        return None
    feats = disamb.parse_feats(str(ud.get("feats") or "_"))
    token = disamb.Token(form=form, lemma=form.casefold(), upos=upos, xpos="_", feats=feats)
    return {str(key): str(value) for key, value in disamb.token_tags(token).items()}


def nvidia_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def predict_joint(
    rows: list[dict[str, Any]],
    checkpoint: Path,
    batch_size: int,
    force_cpu: bool,
    cuda_memory_threshold_mib: int,
) -> list[list[dict[str, Any] | None]]:
    import torch
    from torch.utils.data import DataLoader

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if nvidia_available():
        print("nvidia-smi: available")
    else:
        print("nvidia-smi: unavailable; falling back to torch device detection")
    device = chrest.choose_device(force_cpu, cuda_memory_threshold_mib)
    model, tokenizer, checkpoint_payload = joint_eval.instantiate_from_checkpoint(
        checkpoint,
        device=device,
    )
    char_vocab = checkpoint_payload["char_vocab"]
    joint_rows = [
        {
            "id": row.get("id"),
            "text": row.get("text") or "",
            "tokens": [
                {"word": str(token.get("word") or ""), "pos_label": "X|_", "stress": None}
                for token in row.get("tokens", [])
            ],
        }
        for row in rows
    ]
    collator = joint_eval.JointCollator(tokenizer, model.labels, char_vocab)
    loader = DataLoader(
        joint_eval.JointDataset(joint_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    started = time.perf_counter()
    predictions = joint_eval.predict_batches(model, loader, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    token_count = sum(len(row.get("tokens", [])) for row in rows)
    print(
        f"joint predicted {token_count:,} tokens in "
        f"{time.perf_counter() - started:.1f}s on {device}"
    )

    aligned_rows: list[list[dict[str, Any] | None]] = []
    skipped_source = skipped_joint = 0
    for row, predicted_row in zip(rows, predictions):
        source_keys = [str(token["key"]) for token in row.get("tokens", [])]
        pred_tokens = list(predicted_row.get("tokens") or [])
        aligned, skipped_row_source, skipped_row_joint = align_by_key(
            source_keys,
            pred_tokens,
            lambda item: word_key(str(item.get("word") or "")),
        )
        skipped_source += skipped_row_source
        skipped_joint += skipped_row_joint
        aligned_rows.append(aligned)
    if skipped_source or skipped_joint:
        print(
            f"joint alignment: skipped_source={skipped_source:,} "
            f"skipped_joint={skipped_joint:,}"
        )
    return aligned_rows


def tagger_label(token: Any) -> str:
    return combined_label(str(token.upos or "X"), feats_string(dict(token.feats or {})))


def predict_tagger(
    rows: list[dict[str, Any]],
    tagger_url_override: str | None,
    tagger_timeout: float,
    request_timeout: float,
) -> list[list[str | None]]:
    output: list[list[str | None]] = []
    skipped_source = skipped_tagger = 0
    with nodict.tagger_url(tagger_url_override, tagger_timeout) as url:
        for row_index, row in enumerate(rows, start=1):
            conllu = nodict.post_tagger(url, str(row.get("text") or ""), timeout=request_timeout)
            parsed = nodict.parse_conllu(conllu)
            aligned, skipped_row_source, skipped_row_tagger = align_by_key(
                [str(token["key"]) for token in row.get("tokens", [])],
                parsed,
                lambda item: word_key(str(item.form)),
            )
            skipped_source += skipped_row_source
            skipped_tagger += skipped_row_tagger
            output.append([tagger_label(token) if token is not None else None for token in aligned])
            if row_index % 100 == 0:
                print(f"tagger annotated {row_index:,}/{len(rows):,} sentences")
    if skipped_source or skipped_tagger:
        print(
            f"tagger alignment: skipped_source={skipped_source:,} "
            f"skipped_tagger={skipped_tagger:,}"
        )
    return output


def predict_liepa(tokens: list[dict[str, Any]]) -> dict[int, str]:
    from phonology_engine import PhonologyEngine
    from guess_uncovered import engine_accent

    pe = PhonologyEngine()
    cache: dict[str, str | None] = {}
    for key in sorted({str(token["key"]) for token in tokens if token.get("key")}):
        form = engine_accent(pe, key)
        cache[key] = answer_form(key, form) if form else None
    return {
        int(token["global_index"]): str(cache[str(token["key"])])
        for token in tokens
        if cache.get(str(token["key"]))
    }


def load_dictionary_layers(
    tokens: list[dict[str, Any]],
    joint_rows: list[list[dict[str, Any] | None]],
    rows: list[dict[str, Any]],
    generated: Path,
) -> dict[int, dict[str, str]]:
    target_words = {str(token["key"]) for token in tokens if token.get("key")}
    print(f"loading generated.sqlite target rows: {len(target_words):,} word keys")
    _candidates, entries, slot_cache = nodict.load_generated(generated, target_words)
    print(f"dictionary rows loaded: {len(entries):,}")

    out: dict[int, dict[str, str]] = {}
    for row, row_joint in zip(rows, joint_rows):
        for token, joint_token in zip(row.get("tokens", []), row_joint):
            key = str(token["key"])
            entry = entries.get(key)
            if entry is None:
                continue
            joint_label = str(joint_token.get("pos") or "") if joint_token else ""
            picked = nodict.pick_dict_form(entry, joint_label, slot_cache)
            default = nodict.default_form(entry)
            payload: dict[str, str] = {}
            picked_form = answer_form(key, picked) if picked else None
            default_answer = answer_form(key, default) if default else None
            if picked_form:
                payload["form"] = picked_form
            if default_answer:
                payload["default_form"] = default_answer
            if payload:
                out[int(token["global_index"])] = payload
    return out


def layer_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "vdu": 0,
        "joint_accent": 0,
        "joint_pos": 0,
        "liepa": 0,
        "dict": 0,
        "dict_default": 0,
        "tagger_pos": 0,
        "udpipe_slots": 0,
    }
    for row in rows:
        for token in row.get("tokens", []):
            layers = token.get("layers") or {}
            counts["vdu"] += int(bool(layer_form(layers.get("vdu"))))
            counts["joint_accent"] += int(bool(layer_form(layers.get("joint"))))
            counts["joint_pos"] += int(bool(layer_pos(layers.get("joint"))))
            counts["liepa"] += int(bool(layer_form(layers.get("liepa"))))
            dict_layer = layers.get("dict") if isinstance(layers.get("dict"), dict) else {}
            counts["dict"] += int(bool(layer_form(dict_layer)))
            counts["dict_default"] += int(bool(dict_layer.get("default_form")))
            counts["tagger_pos"] += int(bool(layer_pos(layers.get("tagger"))))
            counts["udpipe_slots"] += int(bool(token.get("udpipe_slots")))
    return counts


def print_layer_counts(rows: list[dict[str, Any]]) -> None:
    total = sum(len(row.get("tokens", [])) for row in rows)
    print(f"sentences: {len(rows):,}; word tokens: {total:,}")
    for name, count in layer_counts(rows).items():
        print(f"{name}: {count:,}/{total:,} ({count / (total or 1):.2%})")


def collect_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    vdu_silver: Path | None = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    generated: Path = DEFAULT_GENERATED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_cpu: bool = False,
    cuda_memory_threshold_mib: int = 6144,
    tagger_url_override: str | None = None,
    tagger_timeout: float = 180.0,
    request_timeout: float = 180.0,
) -> Path:
    add_token_metadata(rows)
    output_path = resolve_output_path(output_path)
    existing = valid_jsonl_count(output_path)
    if existing > len(rows):
        raise RuntimeError(
            f"existing output has {existing:,} rows, but corpus has only {len(rows):,}: "
            f"{safe_relative(output_path)}"
        )
    if existing == len(rows):
        print(f"layers already complete: {safe_relative(output_path)}")
        print_layer_counts(load_layer_rows(output_path))
        return output_path

    pending = rows[existing:]
    print(
        f"collecting layers for {len(pending):,}/{len(rows):,} sentences "
        f"-> {safe_relative(output_path)}"
    )

    silver_by_global, (skipped_vdu_tokens, skipped_vdu_silver) = load_vdu_alignment(
        rows,
        vdu_silver,
    )
    if vdu_silver is not None:
        print(
            f"vdu alignment: matched={len(silver_by_global):,} "
            f"skipped_tokens={skipped_vdu_tokens:,} skipped_silver={skipped_vdu_silver:,}"
        )

    pending_tokens = flatten_tokens(pending)
    joint_rows = predict_joint(
        pending,
        checkpoint=checkpoint,
        batch_size=batch_size,
        force_cpu=force_cpu,
        cuda_memory_threshold_mib=cuda_memory_threshold_mib,
    )
    tagger_rows = predict_tagger(
        pending,
        tagger_url_override=tagger_url_override,
        tagger_timeout=tagger_timeout,
        request_timeout=request_timeout,
    )
    print("running LIEPA per distinct word")
    liepa_by_global = predict_liepa(pending_tokens)
    dict_by_global = load_dictionary_layers(pending_tokens, joint_rows, pending, generated)

    output_rows: list[dict[str, Any]] = []
    for row, row_joint, row_tagger in zip(pending, joint_rows, tagger_rows):
        out_tokens: list[dict[str, Any]] = []
        for token, joint_token, tagger_label_value in zip(
            row.get("tokens", []),
            row_joint,
            row_tagger,
        ):
            global_index = int(token["global_index"])
            key = str(token["key"])
            layers: dict[str, dict[str, Any]] = {}

            silver_token = silver_by_global.get(global_index)
            if silver_token is not None:
                layers["vdu"] = {
                    "form": answer_form(key, str(silver_token.accented)),
                    "mi": silver_token.mi,
                    "ambiguous": bool(silver_token.ambiguous),
                }

            if joint_token is not None:
                joint_layer: dict[str, Any] = {
                    "pos_label": str(joint_token.get("pos") or ""),
                }
                form = answer_form(key, joint_token.get("stress"))
                if form:
                    joint_layer["form"] = form
                layers["joint"] = joint_layer

            liepa_form = liepa_by_global.get(global_index)
            if liepa_form:
                layers["liepa"] = {"form": liepa_form}

            dict_layer = dict_by_global.get(global_index)
            if dict_layer:
                layers["dict"] = dict_layer

            if tagger_label_value:
                layers["tagger"] = {"pos_label": tagger_label_value}

            out_token: dict[str, Any] = {
                "word": str(token.get("word") or ""),
                "key": key,
                "token_index": int(token.get("token_index", 0)),
                "global_index": global_index,
                "layers": layers,
            }
            slots = udpipe_slots(silver_token.ud, str(token.get("word") or "")) if silver_token else None
            if slots:
                out_token["udpipe_slots"] = slots
            if "source_token_index" in token:
                out_token["source_token_index"] = int(token["source_token_index"])
            out_tokens.append(out_token)

        output_rows.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "sent_id": int(row.get("sent_id", 0)),
                "text": row.get("text") or "",
                "tokens": out_tokens,
            }
        )

    write_jsonl_append(output_path, output_rows)
    all_rows = load_layer_rows(output_path)
    print(f"layers written: {safe_relative(output_path)}")
    print_layer_counts(all_rows)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--vdu-silver", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-memory-threshold-mib", type=int, default=6144)
    parser.add_argument("--tagger-url")
    parser.add_argument("--tagger-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    corpus = resolve_input_path(args.corpus)
    vdu_silver = resolve_input_path(args.vdu_silver) if args.vdu_silver else None
    checkpoint = resolve_input_path(args.checkpoint)
    generated = resolve_input_path(args.generated)
    output = resolve_output_path(args.out) if args.out else default_layers_path(corpus)

    for path, label in ((corpus, "corpus"), (checkpoint, "joint checkpoint"), (generated, "generated DB")):
        if not path.exists():
            parser.error(f"missing {label}: {path}")
    if vdu_silver is not None and not vdu_silver.exists():
        parser.error(f"missing VDU silver JSONL: {vdu_silver}")

    rows = rows_from_corpus(corpus)
    if not rows:
        parser.error(f"no word-token sentences in corpus: {corpus}")
    collect_rows(
        rows,
        output,
        vdu_silver=vdu_silver,
        checkpoint=checkpoint,
        generated=generated,
        batch_size=args.batch_size,
        force_cpu=args.cpu,
        cuda_memory_threshold_mib=args.cuda_memory_threshold_mib,
        tagger_url_override=args.tagger_url,
        tagger_timeout=args.tagger_timeout,
        request_timeout=args.request_timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
