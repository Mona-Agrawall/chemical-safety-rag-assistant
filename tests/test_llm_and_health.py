import json

import httpx
import pytest

from rag.errors import LLMError, LLMStreamError, LLMTimeoutError, LLMUnavailableError, ModelNotFoundError
from rag.health import HealthState, check_ollama, model_available
from rag.llm import OllamaClient

URL = "http://ollama.test:11434"


def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def client(handler) -> OllamaClient:
    return OllamaClient(URL, "mistral", transport=httpx.MockTransport(handler), read_timeout_s=2)


def test_streams_tokens_and_sends_options():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        body = ndjson({"message": {"content": "Hello "}}, {"message": {"content": "[S1]"}}, {"done": True})
        return httpx.Response(200, content=body)

    out = list(client(handler).stream_chat([{"role": "user", "content": "hi"}]))
    assert out == ["Hello ", "[S1]"]
    assert seen["model"] == "mistral" and seen["stream"] is True
    assert seen["options"]["temperature"] == 0.0


def test_model_not_found_maps_to_specific_error():
    def handler(req):
        return httpx.Response(404, json={"error": "model 'mistral' not found"})

    with pytest.raises(ModelNotFoundError, match="ollama pull mistral"):
        list(client(handler).stream_chat([]))


def test_server_error_maps_to_llm_error():
    with pytest.raises(LLMError, match="HTTP 500"):
        list(client(lambda r: httpx.Response(500, json={"error": "boom"})).stream_chat([]))


def test_connection_refused_maps_to_unavailable():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(LLMUnavailableError, match="Cannot connect"):
        list(client(handler).stream_chat([]))


def test_read_timeout_maps_to_timeout():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(LLMTimeoutError):
        list(client(handler).stream_chat([]))


def test_error_inside_stream():
    body = ndjson({"message": {"content": "partial"}}, {"error": "out of memory"})
    gen = client(lambda r: httpx.Response(200, content=body)).stream_chat([])
    assert next(gen) == "partial"
    with pytest.raises(LLMStreamError, match="out of memory"):
        next(gen)


def test_stream_without_done_is_an_error():
    body = ndjson({"message": {"content": "partial"}})
    with pytest.raises(LLMStreamError, match="ended unexpectedly"):
        list(client(lambda r: httpx.Response(200, content=body)).stream_chat([]))


def test_malformed_line():
    with pytest.raises(LLMStreamError, match="Malformed"):
        list(client(lambda r: httpx.Response(200, content=b"not json\n")).stream_chat([]))


# ── health ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("model", "installed", "ok"),
    [
        ("mistral", ["mistral:latest"], True),
        ("mistral:latest", ["mistral:latest"], True),
        ("mistral:7b", ["mistral:latest"], False),
        ("llama3", ["mistral:latest"], False),
    ],
)
def test_model_available(model, installed, ok):
    assert model_available(model, installed) is ok


def tags(*names):
    return httpx.MockTransport(lambda r: httpx.Response(200, json={"models": [{"name": n} for n in names]}))


def test_health_ok():
    h = check_ollama(URL, "mistral", transport=tags("mistral:latest"))
    assert h.ok and h.state is HealthState.OK


def test_health_model_missing():
    h = check_ollama(URL, "mistral", transport=tags("llama3:latest"))
    assert h.state is HealthState.MODEL_MISSING and "ollama pull mistral" in h.message
    assert h.models == ["llama3:latest"]


def test_health_server_down():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    h = check_ollama(URL, "mistral", transport=httpx.MockTransport(handler))
    assert h.state is HealthState.SERVER_DOWN and "ollama serve" in h.message


def test_health_timeout():
    def handler(req):
        raise httpx.ConnectTimeout("slow", request=req)

    assert check_ollama(URL, "mistral", transport=httpx.MockTransport(handler)).state is HealthState.TIMEOUT


def test_health_bad_response():
    t = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>"))
    assert check_ollama(URL, "mistral", transport=t).state is HealthState.ERROR
    t = httpx.MockTransport(lambda r: httpx.Response(503))
    assert check_ollama(URL, "mistral", transport=t).state is HealthState.ERROR
