from __future__ import annotations

import os
from functools import lru_cache

import stanza
from fastapi import FastAPI, Form


app = FastAPI(title="Stanza UDPipe-compatible Lithuanian tagger")


@lru_cache(maxsize=1)
def pipeline() -> stanza.Pipeline:
    return stanza.Pipeline(
        "lt",
        processors="tokenize,pos,lemma",
        package="alksnis",
        use_gpu=False,
        dir=os.getenv("STANZA_RESOURCES_DIR"),
    )


@app.post("/process")
async def process(
    data: str = Form(...),
    tokenizer: str = Form(""),
    tagger: str = Form(""),
    model: str = Form("lithuanian-alksnis"),
) -> dict[str, str]:
    del tokenizer, tagger, model
    doc = pipeline()(data)
    return {"result": to_conllu(doc)}


def to_conllu(doc: stanza.Document) -> str:
    lines: list[str] = []
    for sentence_index, sentence in enumerate(doc.sentences, start=1):
        lines.append(f"# sent_id = {sentence_index}")
        for index, word in enumerate(sentence.words, start=1):
            columns = [
                str(index),
                word.text or "_",
                word.lemma or "_",
                word.upos or "_",
                word.xpos or "_",
                word.feats or "_",
                "_",
                "_",
                "_",
                "_",
            ]
            lines.append("\t".join(columns))
        lines.append("")
    return "\n".join(lines)
