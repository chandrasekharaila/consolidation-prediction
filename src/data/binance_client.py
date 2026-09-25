from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

import requests
from requests.adapters import HTTPAdapter

LOG = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"

# Requests' default pool is 10 connections; concurrent scans need more or urllib3
# discards and rebuilds connections on every call.
DEFAULT_POOL_SIZE = 32

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ignore",
]


class BinanceClient:
    """Thin public-API client with a rough weight budget and retries.

    Spot IP limit is 6000 weight/min; we target a conservative budget so a wide
    multi-symbol scan does not trip a 429/418 ban.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        weight_budget: int = 4000,
        timeout: int = 15,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.weight_budget = weight_budget
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        adapter = HTTPAdapter(
            pool_connections=DEFAULT_POOL_SIZE, pool_maxsize=DEFAULT_POOL_SIZE
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._weight_used = 0
        self._window_start = time.time()
        self._lock = threading.Lock()

    def _throttle(self, weight: int) -> None:
        # Guarded so concurrent callers (the live app fetches symbols in parallel)
        # cannot overshoot the shared weight budget.
        with self._lock:
            now = time.time()
            if now - self._window_start >= 60:
                self._window_start = now
                self._weight_used = 0
            if self._weight_used + weight > self.weight_budget:
                sleep_for = 60 - (now - self._window_start)
                if sleep_for > 0:
                    LOG.debug("rate limit: sleeping %.1fs", sleep_for)
                    time.sleep(sleep_for)
                self._window_start = time.time()
                self._weight_used = 0
            self._weight_used += weight

    def _get(self, path: str, params: dict[str, Any], weight: int) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle(weight)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network hiccup
                last_error = exc
                time.sleep(min(2**attempt, 20))
                continue

            self._note_used_weight(resp)
            if resp.status_code in (429, 418):
                retry_after = int(resp.headers.get("Retry-After", 5))
                LOG.warning("rate limited (%s), backing off %ss", resp.status_code, retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                last_error = requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
                time.sleep(min(2**attempt, 20))
                continue
            if resp.status_code >= 400:
                raise requests.HTTPError(f"{resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise RuntimeError(f"GET {path} failed after {self.max_retries} attempts: {last_error}")

    def _note_used_weight(self, resp) -> None:
        """Adopt the server's own usage counter.

        A purely local window can be offset from Binance's, so our count under-reports
        at the boundary and we overshoot into a 429. The header is authoritative.
        """
        used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
        if not used:
            return
        try:
            used_weight = int(used)
        except ValueError:
            return
        with self._lock:
            self._weight_used = max(self._weight_used, used_weight)

    def get_exchange_info(self) -> dict:
        return self._get("/api/v3/exchangeInfo", {}, weight=20)

    def get_24h_tickers(self) -> list[dict]:
        return self._get("/api/v3/ticker/24hr", {}, weight=80)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        weight = 2 if limit <= 1000 else 5
        return self._get("/api/v3/klines", params, weight=weight)

    def iter_klines(
        self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000
    ) -> Iterator[list]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval: {interval}")
        step = INTERVAL_MS[interval]
        cursor = int(start_ms)
        while cursor < end_ms:
            batch = self.get_klines(symbol, interval, cursor, end_ms, limit=limit)
            if not batch:
                break
            for kline in batch:
                yield kline
            last_open = int(batch[-1][0])
            next_cursor = last_open + step
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
