# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "httpx"]
# ///
"""Upload local-model/ to the remote R2 bucket, multipart-capable.

`wrangler r2 object put` caps uploads at 300 MiB, which the heavy model tier
(~450 MB) exceeds. This uses R2's S3-compatible API instead, deriving S3
credentials from the CLOUDFLARE_API_TOKEN in .env (documented R2 mechanism:
access_key_id = the token's ID, secret_access_key = hex(sha256(token))).

Upload order matters: manifest.json goes LAST so clients never see a manifest
pointing at objects that aren't in place yet.

Usage: uv run scripts/upload_local_model_r2_multipart.py
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path

import boto3
import httpx
from boto3.s3.transfer import TransferConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "local-model"
BUCKET = "kirciuokle-models"

CONTENT_TYPES = {
    ".onnx": "application/octet-stream",
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".json": "application/json",
}


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = load_env()
    token = env.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_TOKEN")
    account = env.get("CLOUDFLARE_ACCOUNT_ID") or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        raise SystemExit("CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID missing")

    verify = httpx.get(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    ).json()
    if not verify.get("success"):
        raise SystemExit(f"token verify failed: {verify}")
    token_id = verify["result"]["id"]

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=token_id,
        aws_secret_access_key=hashlib.sha256(token.encode()).hexdigest(),
        region_name="auto",
    )
    transfer = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
    )

    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    # manifest last: it's the pointer that makes the rest live.
    files.sort(key=lambda p: (p.name == "manifest.json", str(p)))
    for path in files:
        key = path.relative_to(SRC).as_posix()
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        size_mb = path.stat().st_size / 1048576
        print(f"put {key} ({size_mb:.1f} MB, {content_type})", flush=True)
        s3.upload_file(
            str(path),
            BUCKET,
            key,
            ExtraArgs={"ContentType": content_type},
            Config=transfer,
        )
    print(f"uploaded {len(files)} files to r2://{BUCKET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
