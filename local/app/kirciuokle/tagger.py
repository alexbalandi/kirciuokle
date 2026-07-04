from __future__ import annotations

import os

import httpx

from .disambiguate import Token, parse_conllu


UDPIPE_MODEL = "lithuanian-alksnis"
UDPIPE_TIMEOUT_SECONDS = 10.0


async def tag_text(text: str) -> list[Token]:
    tagger_url = os.getenv("TAGGER_URL", "http://tagger:8001").rstrip("/")
    async with httpx.AsyncClient(timeout=UDPIPE_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{tagger_url}/process",
            data={
                "tokenizer": "",
                "tagger": "",
                "model": UDPIPE_MODEL,
                "data": text,
            },
            headers={
                "accept": "application/json",
                "content-type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        payload = response.json()

    result = payload.get("result")
    if not isinstance(result, str):
        raise ValueError("UDPipe response did not include CoNLL-U.")

    return parse_conllu(result)
