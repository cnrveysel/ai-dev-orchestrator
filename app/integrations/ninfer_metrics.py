"""NInfer (llama.cpp-style) Prometheus metrics -> live decoding speed.

tok/s is computed from DELTAS between two scrapes, never from lifetime totals:
    delta(tokens_predicted_total) / delta(tokens_predicted_seconds_total)
"""
from __future__ import annotations

import asyncio
import time
import urllib.request
from dataclasses import dataclass

TOKENS_TOTAL = "llamacpp:tokens_predicted_total"
TOKENS_SECONDS = "llamacpp:tokens_predicted_seconds_total"


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            name_part, value = line.rsplit(None, 1)
            name = name_part.split("{", 1)[0]
            values[name] = values.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return values


@dataclass
class MetricsSample:
    ok: bool
    tokens_per_second: float | None = None  # None = no decoding during interval
    tokens_total: float | None = None
    error: str = ""
    at: float = 0.0


class DecodeRateTracker:
    def __init__(self) -> None:
        self._prev: tuple[float, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def update(self, tokens_total: float, seconds_total: float) -> float | None:
        prev, self._prev = self._prev, (tokens_total, seconds_total)
        if prev is None:
            return None
        d_tok = tokens_total - prev[0]
        d_sec = seconds_total - prev[1]
        if d_tok < 0 or d_sec < 0:  # server restarted: counters reset
            return None
        if d_tok == 0 or d_sec <= 0:
            return None
        return d_tok / d_sec


class NInferMetricsClient:
    def __init__(self, url: str, timeout: float = 3.0):
        self.url = url
        self.timeout = timeout
        self.tracker = DecodeRateTracker()

    def _fetch(self) -> str:
        req = urllib.request.Request(self.url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 (user-configured URL)
            return resp.read().decode("utf-8", errors="replace")

    async def sample(self) -> MetricsSample:
        now = time.time()
        try:
            text = await asyncio.to_thread(self._fetch)
        except Exception as exc:
            self.tracker.reset()
            return MetricsSample(False, error=str(exc), at=now)
        vals = parse_prometheus(text)
        if TOKENS_TOTAL not in vals or TOKENS_SECONDS not in vals:
            return MetricsSample(False, error="decode counters not exposed by endpoint", at=now)
        rate = self.tracker.update(vals[TOKENS_TOTAL], vals[TOKENS_SECONDS])
        return MetricsSample(True, rate, vals[TOKENS_TOTAL], at=now)


def probe_models(base_url: str, timeout: float = 3.0) -> list[str]:
    """Blocking: list model ids from an OpenAI-compatible /v1/models endpoint."""
    import json
    url = base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        data = json.loads(resp.read().decode("utf-8"))
    return [str(m.get("id")) for m in data.get("data", []) if isinstance(m, dict)]
