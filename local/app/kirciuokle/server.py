from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .accent import accent_text_local_first
from .dictionary import FallbackMode, WordDictionary, lookup_word_variants
from .disambiguate import to_public_variants
from .vdu import WORD_CACHE_SECONDS, UpstreamError, accent_text


MAX_TEXT_LENGTH = 20_000
ALL_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

logger = logging.getLogger(__name__)


def create_app(
    *,
    dictionary: WordDictionary | None = None,
    fallback: FallbackMode | None = None,
    static_dir: str | Path | None = None,
    dict_path: str | Path | None = None,
    migrations_dir: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        get_dictionary(app)
        try:
            yield
        finally:
            if app.state.owns_dictionary and app.state.dictionary is not None:
                app.state.dictionary.close()
                app.state.dictionary = None

    app = FastAPI(title="Kirčiuoklė local replica", lifespan=lifespan)
    app.state.dictionary = dictionary
    app.state.owns_dictionary = dictionary is None
    app.state.dict_path = Path(dict_path or os.getenv("DICT_PATH", "/data/words.sqlite"))
    app.state.migrations_dir = Path(
        migrations_dir or os.getenv("MIGRATIONS_DIR", "")
    ) if migrations_dir or os.getenv("MIGRATIONS_DIR") else None
    app.state.fallback = normalize_fallback(fallback or os.getenv("FALLBACK", "vdu"))
    app.state.accent_source = normalize_source(os.getenv("ACCENT_SOURCE", "local"))

    @app.api_route("/api/accent", methods=ALL_METHODS)
    async def handle_accent(
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        try:
            if request.method != "POST":
                return json({"error": "Metodas nepalaikomas."}, 405)

            payload = await read_json(request)
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or len(text.strip()) == 0:
                return json({"error": "Įveskite tekstą."}, 400)

            if len(text) > MAX_TEXT_LENGTH:
                return json({"error": "Tekstas per ilgas."}, 413)

            dictionary = get_dictionary(app)
            source = get_accent_source(request, app)
            if source == "vdu":
                if app.state.fallback == "none":
                    raise UpstreamError()
                response = await accent_text(
                    text,
                    lookup_variants=lambda word: lookup_word_variants(
                        dictionary,
                        word,
                        fallback="vdu",
                    ),
                )
            else:
                response = await accent_text_local_first(
                    text,
                    dictionary,
                    fallback=app.state.fallback,
                    background_tasks=background_tasks,
                )

            return json(response)
        except UpstreamError as error:
            return json({"error": str(error)}, 502)
        except Exception:
            logger.exception("Unexpected /api/accent failure")
            return json({"error": "Įvyko netikėta klaida."}, 500)

    @app.api_route("/api/word", methods=ALL_METHODS)
    async def handle_word(request: Request) -> JSONResponse:
        try:
            if request.method != "GET":
                return json({"error": "Metodas nepalaikomas."}, 405)

            word = (request.query_params.get("w") or "").strip()
            if not word:
                return json({"error": "Trūksta žodžio."}, 400)

            dictionary = get_dictionary(app)
            variants = await lookup_word_variants(
                dictionary,
                word,
                fallback=app.state.fallback,
            )
            return json(
                {"variants": to_public_variants(variants)},
                headers={"cache-control": f"public, max-age={WORD_CACHE_SECONDS}"},
            )
        except UpstreamError as error:
            return json({"error": str(error)}, 502)
        except Exception:
            logger.exception("Unexpected /api/word failure")
            return json({"error": "Įvyko netikėta klaida."}, 500)

    @app.api_route("/api/{_:path}", methods=ALL_METHODS)
    async def api_not_found() -> JSONResponse:
        return json({"error": "API maršrutas nerastas."}, 404)

    static_path = Path(static_dir or os.getenv("STATIC_DIR", "./static"))
    if static_path.exists():
        app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

    return app


def get_dictionary(app: FastAPI) -> WordDictionary:
    dictionary = app.state.dictionary
    if dictionary is None:
        dictionary = WordDictionary(app.state.dict_path, app.state.migrations_dir)
        app.state.dictionary = dictionary
    return dictionary


async def read_json(request: Request) -> Any:
    try:
        return await request.json()
    except Exception:
        return None


def json(
    body: Any,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        body,
        status_code=status,
        headers={
            "content-type": "application/json; charset=utf-8",
            "x-content-type-options": "nosniff",
            **(headers or {}),
        },
    )


def normalize_fallback(value: str) -> FallbackMode:
    return "none" if value == "none" else "vdu"


AccentSource = Literal["local", "vdu"]


def normalize_source(value: str) -> AccentSource:
    return "vdu" if value == "vdu" else "local"


def get_accent_source(request: Request, app: FastAPI) -> AccentSource:
    requested = request.query_params.get("source")
    if requested in ("local", "vdu"):
        return requested
    return app.state.accent_source


app = create_app()
