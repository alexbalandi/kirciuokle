# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Assemble a Hugging Face release folder for a trained Lithuanian tagger."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = BASE_DIR / "model_card_template.md"
DEFAULT_RELEASE_DIR = BASE_DIR / "hf_release"


CC_BY_SA_4_LICENSE = """Creative Commons Attribution-ShareAlike 4.0 International

This model is licensed under the Creative Commons Attribution-ShareAlike 4.0
International Public License (CC BY-SA 4.0).

You are free to:

- Share: copy and redistribute the material in any medium or format.
- Adapt: remix, transform, and build upon the material for any purpose.

Under the following terms:

- Attribution: give appropriate credit, provide a link to the license, and
  indicate if changes were made.
- ShareAlike: if you remix, transform, or build upon the material, distribute
  your contributions under the same license as the original.
- No additional restrictions: do not apply legal terms or technological
  measures that legally restrict others from doing anything the license permits.

Full legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode
SPDX-License-Identifier: CC-BY-SA-4.0
"""


def load_json(path: Path | None, *, required: bool = True) -> dict:
    if path is None:
        return {}
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label} directory: {path}")


def copy_tree_contents(source: Path, destination: Path) -> None:
    require_dir(source, "source")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def clean_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def format_value(key: str, value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        lowered = key.lower()
        if 0.0 <= value <= 1.0 and any(
            marker in lowered for marker in ("accuracy", "score", "f1", "rate")
        ):
            return f"{value:.2%}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def markdown_table(rows: list[Mapping[str, object]]) -> str:
    if not rows:
        return "_TBD_"
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(key, row.get(key, "")) for key in headers)
            + " |"
        )
    return "\n".join(lines)


def mapping_table(payload: Mapping[str, object]) -> str:
    if not payload:
        return "_TBD_"
    return markdown_table(
        [{"field": key, "value": value} for key, value in sorted(payload.items())]
    )


def json_to_markdown(payload: object) -> str:
    if isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, Mapping)]
        return markdown_table(rows) if rows else format_value("value", payload)
    if isinstance(payload, Mapping):
        if "rows" in payload:
            return json_to_markdown(payload["rows"])
        return mapping_table(payload)
    if payload is None:
        return "_TBD_"
    return str(payload)


def metric_rows(final_json: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("best_dev", "test"):
        payload = final_json.get(split)
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if key in {"head", "pooling", "epoch", "step"}:
                continue
            rows.append({"split": split, "metric": key, "value": value})
    return rows


def extract_fenced_section(markdown: str, heading_fragment: str) -> str:
    if not markdown:
        return "_TBD_"
    lines = markdown.splitlines()
    heading_index = -1
    target = heading_fragment.casefold()
    for index, line in enumerate(lines):
        if line.startswith("#") and target in line.casefold():
            heading_index = index
            break
    if heading_index == -1:
        return "_TBD_"

    in_fence = False
    collected: list[str] = []
    for line in lines[heading_index + 1 :]:
        if line.startswith("#") and not in_fence:
            break
        if line.startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if in_fence:
            collected.append(line)
    return "\n".join(collected).strip() or "_TBD_"


def read_conll18_markdown(run_dir: Path, requested: Path | None, bench: Mapping[str, object]) -> str:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested)
    value = bench.get("conll18_md") or bench.get("conll18_markdown_path")
    if isinstance(value, str) and value:
        candidates.append(Path(value))
    candidates.append(run_dir.parent / f"conll18-{run_dir.name}.md")

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    embedded = bench.get("conll18_markdown")
    return str(embedded) if embedded else ""


def known_fact_keys() -> set[str]:
    return {
        "bench_slots",
        "comparison_rows",
        "comparison_table",
        "conll18_md",
        "conll18_markdown",
        "conll18_markdown_path",
        "model_id",
        "official_conll18_table",
        "official_table",
        "speed",
        "speed_table",
        "vdu_conll18_table",
        "vdu_table",
    }


def context_from_inputs(
    *,
    run_dir: Path,
    bench: Mapping[str, object],
    facts: Mapping[str, object],
    conll18_markdown: str,
) -> dict[str, str]:
    final_json = load_json(run_dir / "final.json")
    head_config = load_json(run_dir / "best" / "head_config.json", required=False)
    merged_facts = {**bench, **facts}

    official_table = (
        merged_facts.get("official_conll18_table")
        or merged_facts.get("official_table")
        or extract_fenced_section(conll18_markdown, "Official CoNLL-18")
    )
    vdu_table = (
        merged_facts.get("vdu_conll18_table")
        or merged_facts.get("vdu_table")
        or extract_fenced_section(conll18_markdown, "VDU-convention")
    )
    speed_payload = merged_facts.get("speed_table") or merged_facts.get("speed")
    comparison_payload = (
        merged_facts.get("comparison_table") or merged_facts.get("comparison_rows")
    )

    free_form = {
        key: value
        for key, value in merged_facts.items()
        if key not in known_fact_keys()
    }

    run_name = str(final_json.get("run_name") or run_dir.name)
    base_model = str(
        final_json.get("requested_model")
        or head_config.get("base_model")
        or "EMBEDDIA/litlat-bert"
    )
    return {
        "bench_json_table": json_to_markdown(bench),
        "bench_slots": format_value("bench_slots", merged_facts.get("bench_slots", "TBD")),
        "base_model": base_model,
        "comparison_table": json_to_markdown(comparison_payload),
        "export_date": dt.date.today().isoformat(),
        "facts_table": mapping_table(free_form),
        "final_metrics": markdown_table(metric_rows(final_json)),
        "head": str(final_json.get("head") or head_config.get("head") or "TBD"),
        "max_length": str(head_config.get("max_length", "TBD")),
        "model_id": str(merged_facts.get("model_id") or "YOUR_ORG/lithuanian-tagger"),
        "official_conll18_table": str(official_table),
        "pooling": str(final_json.get("pooling") or head_config.get("pooling") or "TBD"),
        "run_name": run_name,
        "speed_table": json_to_markdown(speed_payload),
        "vdu_conll18_table": str(vdu_table),
    }


def render_template(template_path: Path, context: Mapping[str, str]) -> str:
    text = template_path.read_text(encoding="utf-8")
    for key, value in context.items():
        text = text.replace(f"{{{{{key}}}}}", value)
    return text


def write_release(
    *,
    run_dir: Path,
    out_dir: Path,
    onnx_dir: Path | None,
    readme: str,
    clean: bool,
) -> None:
    best_dir = run_dir / "best"
    require_dir(best_dir, "best model")
    if clean:
        clean_output(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    copy_tree_contents(best_dir, out_dir)
    if onnx_dir is not None:
        require_dir(onnx_dir, "ONNX")
        copy_tree_contents(onnx_dir, out_dir / "onnx")

    (out_dir / "LICENSE").write_text(CC_BY_SA_4_LICENSE, encoding="utf-8")
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def push_to_hub(out_dir: Path, repo_id: str, private: bool) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--push requires HF_TOKEN in the environment")
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "--push requires huggingface_hub; install it in the export environment"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(out_dir))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a trained tagger run as a Hugging Face model folder."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="runs/<name>")
    parser.add_argument(
        "--bench-json",
        type=Path,
        required=True,
        help="structured benchmark JSON used to fill the model card",
    )
    parser.add_argument("--out", type=Path, required=True, help="release output folder")
    parser.add_argument("--onnx-dir", type=Path, help="optional ONNX artifact directory")
    parser.add_argument("--facts-json", type=Path, help="free-form model-card facts")
    parser.add_argument(
        "--conll18-md",
        type=Path,
        help="CoNLL-18 markdown report; defaults to runs/conll18-<run>.md",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="model card template",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove --out before assembling the release",
    )
    parser.add_argument(
        "--push",
        metavar="REPO_ID",
        help="optional Hugging Face Hub model repo id to upload after assembly",
    )
    parser.add_argument("--private", action="store_true", help="create private HF repo")
    args = parser.parse_args(list(argv) if argv is not None else None)

    require_dir(args.run_dir, "run")
    require_dir(args.run_dir / "best", "best model")
    bench = load_json(args.bench_json)
    facts = load_json(args.facts_json, required=False)
    conll18_markdown = read_conll18_markdown(args.run_dir, args.conll18_md, bench)
    context = context_from_inputs(
        run_dir=args.run_dir,
        bench=bench,
        facts=facts,
        conll18_markdown=conll18_markdown,
    )
    readme = render_template(args.template, context)
    write_release(
        run_dir=args.run_dir,
        out_dir=args.out,
        onnx_dir=args.onnx_dir,
        readme=readme,
        clean=args.clean,
    )

    if args.push:
        push_to_hub(args.out, args.push, private=args.private)

    print(f"HF release ready at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
