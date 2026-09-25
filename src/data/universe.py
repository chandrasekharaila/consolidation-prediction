from __future__ import annotations

from .binance_client import BinanceClient

STABLES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USTC", "AEUR",
    "EUR", "GBP", "TRY", "BRL", "RUB", "ZAR", "UAH", "IDRT", "USD1", "USDE",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def build_universe(
    client: BinanceClient,
    top_n: int = 200,
    quote_asset: str = "USDT",
    exclude_stables: bool = True,
    exclude_leveraged: bool = True,
) -> list[str]:
    info = client.get_exchange_info()
    tickers = client.get_24h_tickers()
    quote_volume = {t["symbol"]: float(t.get("quoteVolume", 0.0)) for t in tickers}

    symbols: list[str] = []
    for sym in info.get("symbols", []):
        if sym.get("status") != "TRADING":
            continue
        if sym.get("quoteAsset") != quote_asset:
            continue
        if sym.get("isSpotTradingAllowed") is False:
            continue
        name = sym["symbol"]
        base = sym.get("baseAsset", "")
        if exclude_stables and base in STABLES:
            continue
        if exclude_leveraged and name.endswith(LEVERAGED_SUFFIXES):
            continue
        if quote_volume.get(name, 0.0) <= 0:
            continue
        symbols.append(name)

    symbols.sort(key=lambda s: quote_volume.get(s, 0.0), reverse=True)
    return symbols[:top_n]
