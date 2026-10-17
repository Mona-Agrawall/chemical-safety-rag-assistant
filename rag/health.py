"""Lightweight Ollama health check (no text generation)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import httpx


class HealthState(str, Enum):
    OK = "ok"
    SERVER_DOWN = "server_down"
    MODEL_MISSING = "model_missing"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class HealthStatus:
    state: HealthState
    message: str
    models: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    checked_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return self.state is HealthState.OK


def model_available(model: str, installed: list[str]) -> bool:
    """Match Ollama naming: "mistral" == "mistral:latest"."""
    want = model if ":" in model else f"{model}:latest"
    return any(name in (model, want) for name in installed)


def check_ollama(
    base_url: str,
    model: str,
    timeout_s: float = 3.0,
    transport: httpx.BaseTransport | None = None,
) -> HealthStatus:
    """GET /api/tags: distinguishes server down, model missing and OK."""
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s, transport=transport) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
        latency = (time.perf_counter() - t0) * 1000
    except httpx.TimeoutException:
        return HealthStatus(
            HealthState.TIMEOUT,
            f"Ollama at {base_url} did not answer within {timeout_s:.0f} s. It may still be starting up.",
        )
    except httpx.HTTPError:
        return HealthStatus(
            HealthState.SERVER_DOWN,
            f"Cannot reach Ollama at {base_url}. Start it with `ollama serve`.",
        )
    if resp.status_code != 200:
        return HealthStatus(HealthState.ERROR, f"Ollama returned HTTP {resp.status_code}.", latency_ms=latency)
    try:
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except ValueError:
        return HealthStatus(HealthState.ERROR, "Ollama returned an unreadable response.", latency_ms=latency)
    if not model_available(model, names):
        return HealthStatus(
            HealthState.MODEL_MISSING,
            f"Ollama is running but model '{model}' is not installed. Run `ollama pull {model}`.",
            models=names,
            latency_ms=latency,
        )
    return HealthStatus(HealthState.OK, f"{model} ready", models=names, latency_ms=latency)
