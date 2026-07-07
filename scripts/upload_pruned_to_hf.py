# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub"]
# ///
"""Add the vocabulary-pruned browser variant to the HF accentuator repo.

Non-destructive: uploads the pruned int8 ONNX + its matching pruned
tokenizer + meta under a `pruned/` prefix and appends a "Pruned variant"
section to the card. The existing joint_v3 root files are left untouched.
"""

from __future__ import annotations

import io
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "alexbalandi/litlat-bert-lithuanian-accentuator"
ROOT = Path(__file__).resolve().parents[1]
PRUNED = ROOT / "local" / "accentuator" / "joint" / "pruned"

UPLOADS = {
    "pruned/joint.int8.partial.onnx": PRUNED / "onnx" / "joint.int8.partial.onnx",
    "pruned/joint.int8.full.onnx": PRUNED / "onnx" / "joint.int8.full.onnx",
    "pruned/joint.meta.json": PRUNED / "onnx" / "joint.meta.json",
    "pruned/tokenizer.json": PRUNED / "tokenizer" / "tokenizer.json",
    "pruned/tokenizer_config.json": PRUNED / "tokenizer" / "tokenizer_config.json",
    "pruned/special_tokens_map.json": PRUNED / "tokenizer" / "special_tokens_map.json",
    "pruned/config.json": PRUNED / "tokenizer" / "config.json",
}

SECTION = """
## Pruned variant (for in-browser / edge inference)

`pruned/` holds a vocabulary-pruned int8 build of this exact model,
meant for shipping to a browser (onnxruntime-web) or any size-sensitive
deployment.

**How it was pruned.** litlat-bert is trilingual (Lithuanian, Latvian,
English), so most of its ~84k SentencePiece vocabulary never fires on
Lithuanian text. We tokenized the whole open dictionary (~575k words)
plus every corpus used in the project and kept only the pieces that were
ever produced — **62,273 of 84,201 embedding rows** — plus, as a safety
floor, every special token, every ≤2-character piece, and the BPE
merge-path pieces the retained ones depend on. The ~22k dropped rows are
overwhelmingly Latvian/English-specific subwords. Removing an unused
embedding row is mathematically lossless for any input that doesn't
tokenize through it; a held-out segmentation census matched the full
tokenizer on **99.99%** of tokens.

The result is **quality-lossless**: chrestomatija-gold, audited-LRT and
ALKSNIS scores all land within ~0.02pp of the full model, and
foreign-word abstention is within ~0.4pp. Only the embedding matrix
shrank (it is ~43% of the encoder).

| file | what | size |
|---|---|---|
| `pruned/joint.int8.partial.onnx` | **recommended** int8 (parity-safe scope), 99.2% token agreement vs torch | ~470 MB |
| `pruned/joint.int8.full.onnx` | fully dynamic int8 — smaller/faster, 96.8% agreement | ~140 MB |
| `pruned/tokenizer.json` (+ configs) | the **matching pruned tokenizer** — required with the pruned weights | ~2 MB |
| `pruned/joint.meta.json` | char vocab + label metadata for decoding | — |

Use `pruned/tokenizer.json` with the pruned weights — the root-level
`tokenizer.json` is the full-vocabulary tokenizer and its ids do **not**
line up with the pruned embedding. Everything else (the I/O contract,
decoding, `label_bridge.json`) is identical to the full model.
"""


def main() -> int:
    for path in UPLOADS.values():
        if not path.exists():
            raise FileNotFoundError(path)

    token = None
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN"):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
    api = HfApi(token=token)

    # append the section to the card (idempotent)
    readme_path = hf_hub_download(REPO, "README.md")
    card = Path(readme_path).read_text(encoding="utf-8")
    if "## Pruned variant" not in card:
        anchor = "## Model I/O (exact contract)"
        if anchor in card:
            card = card.replace(anchor, SECTION.strip() + "\n\n" + anchor, 1)
        else:
            card = card.rstrip() + "\n\n" + SECTION.strip() + "\n"
        api.upload_file(
            path_or_fileobj=io.BytesIO(card.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=REPO,
            commit_message="Document the pruned browser variant",
        )
        print("card updated")
    else:
        print("card already documents the pruned variant; leaving as-is")

    for repo_path, local_path in UPLOADS.items():
        size = local_path.stat().st_size
        print(f"uploading {repo_path} ({size/1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=REPO,
            commit_message=f"Add {repo_path}",
        )
    print("done: https://huggingface.co/alexbalandi/litlat-bert-lithuanian-accentuator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
