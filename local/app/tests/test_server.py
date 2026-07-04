from __future__ import annotations

from fastapi.testclient import TestClient

from kirciuokle.server import create_app

from .helpers import make_dictionary


def test_api_accent_error_mapping(tmp_path) -> None:
    app = create_app(dictionary=make_dictionary(tmp_path), fallback="none")

    with TestClient(app) as client:
        assert client.get("/api/accent").json() == {"error": "Metodas nepalaikomas."}

        empty = client.post("/api/accent", json={"text": "   "})
        assert empty.status_code == 400
        assert empty.json() == {"error": "Įveskite tekstą."}

        too_long = client.post("/api/accent", json={"text": "a" * 20_001})
        assert too_long.status_code == 413
        assert too_long.json() == {"error": "Tekstas per ilgas."}

        upstream = client.post("/api/accent?source=vdu", json={"text": "Čia"})
        assert upstream.status_code == 502
        assert upstream.json() == {"error": "VDU kirčiuoklė laikinai nepasiekiama."}


def test_api_word_and_unknown_route_error_mapping(tmp_path) -> None:
    app = create_app(dictionary=make_dictionary(tmp_path), fallback="none")

    with TestClient(app) as client:
        missing_word = client.get("/api/word")
        assert missing_word.status_code == 400
        assert missing_word.json() == {"error": "Trūksta žodžio."}

        wrong_method = client.post("/api/word?w=yra")
        assert wrong_method.status_code == 405
        assert wrong_method.json() == {"error": "Metodas nepalaikomas."}

        unknown_route = client.get("/api/nope")
        assert unknown_route.status_code == 404
        assert unknown_route.json() == {"error": "API maršrutas nerastas."}
