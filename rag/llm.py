"""Minimal streaming client for Ollama's ``/api/chat`` endpoint."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx

from rag.errors import (
    LLMError,
    LLMStreamError,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotFoundError,
)

log = logging.getLogger(__name__)
Message = dict[str, str]


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 180.0,
        keep_alive: str = "30m",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.options: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx}
        self.keep_alive = keep_alive
        self.timeout = httpx.Timeout(read_timeout_s, connect=connect_timeout_s)
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self._transport)

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Yield response text fragments as Ollama produces them.

        Raises a subclass of :class:`LLMError` describing what went wrong.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }
        produced = False
        try:
            with self._client() as client, client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise _http_error(resp, self.model)
                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMStreamError(f"Malformed response from Ollama: {line[:120]!r}") from exc
                    if "error" in data:
                        raise LLMStreamError(f"Ollama reported an error: {data['error']}")
                    piece = data.get("message", {}).get("content", "")
                    if piece:
                        produced = True
                        yield piece
                    if data.get("done"):
                        return
                raise LLMStreamError("The response stream ended unexpectedly.")
        except LLMError:
            raise
        except httpx.ConnectTimeout as exc:
            raise LLMUnavailableError(f"Timed out connecting to Ollama at {self.base_url}.") from exc
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(f"Cannot connect to Ollama at {self.base_url}.") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"The model did not respond within {self.timeout.read:.0f} s."
            ) from exc
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            if produced:
                raise LLMStreamError("The connection to Ollama dropped mid-answer.") from exc
            raise LLMUnavailableError(f"Connection to Ollama failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

    def chat(self, messages: list[Message]) -> str:
        return "".join(self.stream_chat(messages))


def _http_error(resp: httpx.Response, model: str) -> LLMError:
    try:
        detail = resp.json().get("error", "")
    except Exception:
        detail = resp.text[:200]
    if resp.status_code == 404 or "not found" in str(detail).lower():
        return ModelNotFoundError(
            f"Model '{model}' is not available in Ollama. Run: ollama pull {model}"
        )
    return LLMError(f"Ollama returned HTTP {resp.status_code}: {detail}")
