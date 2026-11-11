"""Live financial market snapshot fetcher with dual-provider resilience and Gram Gold synthesis."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx

from news_bot.models import MarketQuote

logger = logging.getLogger(__name__)

TROY_OUNCE_IN_GRAMS: float = 31.1034768

TRADINGVIEW_SCAN_URL = "https://scanner.tradingview.com/global/scan"
YAHOO_CHART_HOSTS: tuple[str, ...] = (
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class InstrumentSpec:
    """Configuration for a tracked macro/banking market indicator."""

    symbol_id: str
    icon: str
    label: str
    unit_suffix: str
    decimals: int
    tv_tickers: tuple[str, ...]
    yahoo_tickers: tuple[str, ...]


INSTRUMENT_SPECS: tuple[InstrumentSpec, ...] = (
    InstrumentSpec(
        symbol_id="usd_try",
        icon="💵",
        label="Dolar/TL",
        unit_suffix=" TL",
        decimals=2,
        tv_tickers=("FX_IDC:USDTRY", "OANDA:USDTRY"),
        yahoo_tickers=("USDTRY=X", "TRY=X"),
    ),
    InstrumentSpec(
        symbol_id="eur_try",
        icon="💶",
        label="Euro/TL",
        unit_suffix=" TL",
        decimals=2,
        tv_tickers=("FX_IDC:EURTRY", "OANDA:EURTRY"),
        yahoo_tickers=("EURTRY=X",),
    ),
    InstrumentSpec(
        symbol_id="eur_usd",
        icon="💱",
        label="Euro/Dolar",
        unit_suffix="",
        decimals=4,
        tv_tickers=("FX_IDC:EURUSD", "OANDA:EURUSD"),
        yahoo_tickers=("EURUSD=X",),
    ),
    InstrumentSpec(
        symbol_id="gram_gold_try",
        icon="🪙",
        label="Gram Altın",
        unit_suffix=" TL",
        decimals=2,
        tv_tickers=("FX_IDC:XAUTRYG",),
        yahoo_tickers=(),
    ),
    InstrumentSpec(
        symbol_id="ons_gold_usd",
        icon="🥇",
        label="Ons Altın",
        unit_suffix=" $",
        decimals=2,
        tv_tickers=("TVC:GOLD", "OANDA:XAUUSD"),
        yahoo_tickers=("GC=F",),
    ),
    InstrumentSpec(
        symbol_id="brent_oil_usd",
        icon="🛢",
        label="Brent Petrol",
        unit_suffix=" $",
        decimals=2,
        tv_tickers=("TVC:UKOIL",),
        yahoo_tickers=("BZ=F",),
    ),
    InstrumentSpec(
        symbol_id="bist_100",
        icon="📈",
        label="BIST 100",
        unit_suffix="",
        decimals=2,
        tv_tickers=("BIST:XU100",),
        yahoo_tickers=("XU100.IS",),
    ),
    InstrumentSpec(
        symbol_id="bist_bank",
        icon="🏦",
        label="BIST Bankacılık",
        unit_suffix="",
        decimals=2,
        tv_tickers=("BIST:XBANK",),
        yahoo_tickers=("XBANK.IS",),
    ),
)


def _is_valid_number(value: object) -> bool:
    """Return True if value is a finite int/float (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def compute_gram_gold_try(
    *,
    ons_gold: MarketQuote,
    usd_try: MarketQuote,
) -> MarketQuote:
    """
    Synthesize live Turkish Spot Gram Gold (TL) and exact 24-hour compounded % change
    from Spot Ounce Gold (USD) and USD/TRY quotes.
    """
    gram_price = (ons_gold.price * usd_try.price) / TROY_OUNCE_IN_GRAMS
    ons_factor = 1.0 + (ons_gold.change_pct_24h / 100.0)
    usd_factor = 1.0 + (usd_try.change_pct_24h / 100.0)

    if ons_factor > 0.0 and usd_factor > 0.0:
        prev_ons = ons_gold.price / ons_factor
        prev_usd = usd_try.price / usd_factor
        prev_gram = (prev_ons * prev_usd) / TROY_OUNCE_IN_GRAMS
        gram_change_pct = ((gram_price - prev_gram) / prev_gram) * 100.0
    else:
        gram_change_pct = 0.0

    return MarketQuote(
        symbol_id="gram_gold_try",
        icon="🪙",
        label="Gram Altın",
        price=gram_price,
        change_pct_24h=gram_change_pct,
        unit_suffix=" TL",
        decimals=2,
    )


def _extract_yahoo_price_and_change(payload: dict[str, Any]) -> tuple[float, float] | None:
    """Extract current price and 24-hour % change from Yahoo Finance v8 chart response."""
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    results = chart.get("result")
    if not isinstance(results, list) or not results:
        return None

    first = results[0]
    if not isinstance(first, dict):
        return None

    meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
    timestamps = first.get("timestamp") if isinstance(first.get("timestamp"), list) else []
    indicators = first.get("indicators") if isinstance(first.get("indicators"), dict) else {}
    quote_list = indicators.get("quote") if isinstance(indicators.get("quote"), list) else []
    raw_closes = (
        quote_list[0].get("close")
        if quote_list and isinstance(quote_list[0], dict) and isinstance(quote_list[0].get("close"), list)
        else []
    )

    valid_bars: list[tuple[int, float]] = []
    for idx, close_val in enumerate(raw_closes):
        if _is_valid_number(close_val) and float(close_val) > 0.0:
            ts_val = (
                int(timestamps[idx])
                if idx < len(timestamps) and _is_valid_number(timestamps[idx])
                else 0
            )
            valid_bars.append((ts_val, float(close_val)))

    reg_price = meta.get("regularMarketPrice")
    if _is_valid_number(reg_price) and float(reg_price) > 0.0:
        current_price = float(reg_price)
    elif valid_bars:
        current_price = valid_bars[-1][1]
    else:
        return None

    prev_close: float | None = None
    reg_time = meta.get("regularMarketTime")
    if len(valid_bars) >= 2:
        last_bar_ts = valid_bars[-1][0]
        if (
            _is_valid_number(reg_time)
            and last_bar_ts > 0
            and (int(reg_time) - last_bar_ts) > 20 * 3600
        ):
            prev_close = valid_bars[-1][1]
        else:
            prev_close = valid_bars[-2][1]
    elif len(valid_bars) == 1:
        for fallback_key in ("previousClose", "chartPreviousClose"):
            candidate = meta.get(fallback_key)
            if _is_valid_number(candidate) and float(candidate) > 0.0:
                prev_close = float(candidate)
                break
    else:
        for fallback_key in ("previousClose", "chartPreviousClose"):
            candidate = meta.get(fallback_key)
            if _is_valid_number(candidate) and float(candidate) > 0.0:
                prev_close = float(candidate)
                break

    if prev_close is None or prev_close <= 0.0:
        return (current_price, 0.0)

    change_pct = ((current_price - prev_close) / prev_close) * 100.0
    return (current_price, change_pct)


class MarketDataFetcher:
    """Fetches live FX, Gold, Brent Oil, and BIST indices with 24h % changes."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._transport = transport

    def fetch_snapshot(self) -> list[MarketQuote]:
        """
        Fetch all configured market instruments in display order.
        Never raises on external network failure; returns available quotes gracefully.
        """
        quotes_by_id: dict[str, MarketQuote] = {}

        with httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            transport=self._transport,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            tv_raw = self._fetch_tradingview_batch(client)
            for spec in INSTRUMENT_SPECS:
                for tv_ticker in spec.tv_tickers:
                    if tv_ticker in tv_raw:
                        price, change_pct = tv_raw[tv_ticker]
                        quotes_by_id[spec.symbol_id] = MarketQuote(
                            symbol_id=spec.symbol_id,
                            icon=spec.icon,
                            label=spec.label,
                            price=price,
                            change_pct_24h=change_pct,
                            unit_suffix=spec.unit_suffix,
                            decimals=spec.decimals,
                        )
                        break

            for spec in INSTRUMENT_SPECS:
                if spec.symbol_id in quotes_by_id or not spec.yahoo_tickers:
                    continue
                yahoo_pair = self._fetch_yahoo_fallback(client, spec.yahoo_tickers)
                if yahoo_pair is not None:
                    price, change_pct = yahoo_pair
                    quotes_by_id[spec.symbol_id] = MarketQuote(
                        symbol_id=spec.symbol_id,
                        icon=spec.icon,
                        label=spec.label,
                        price=price,
                        change_pct_24h=change_pct,
                        unit_suffix=spec.unit_suffix,
                        decimals=spec.decimals,
                    )

        if (
            "gram_gold_try" not in quotes_by_id
            and "ons_gold_usd" in quotes_by_id
            and "usd_try" in quotes_by_id
        ):
            quotes_by_id["gram_gold_try"] = compute_gram_gold_try(
                ons_gold=quotes_by_id["ons_gold_usd"],
                usd_try=quotes_by_id["usd_try"],
            )

        ordered_quotes = [
            quotes_by_id[spec.symbol_id]
            for spec in INSTRUMENT_SPECS
            if spec.symbol_id in quotes_by_id
        ]
        logger.info(
            "Piyasa özeti çekildi: %d/%d gösterge hazır.",
            len(ordered_quotes),
            len(INSTRUMENT_SPECS),
        )
        return ordered_quotes

    def _fetch_tradingview_batch(
        self, client: httpx.Client
    ) -> dict[str, tuple[float, float]]:
        """Query TradingView Scanner in a single batch HTTP POST request."""
        all_tickers: list[str] = []
        for spec in INSTRUMENT_SPECS:
            for ticker in spec.tv_tickers:
                if ticker not in all_tickers:
                    all_tickers.append(ticker)

        payload = {
            "symbols": {
                "tickers": all_tickers,
                "query": {"types": []},
            },
            "columns": ["close", "change"],
        }
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://www.tradingview.com",
            "Referer": "https://www.tradingview.com/",
        }

        try:
            response = client.post(TRADINGVIEW_SCAN_URL, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "TradingView piyasa verisi alınamadı, yedek kaynağa geçiliyor: %s",
                exc,
            )
            return {}

        rows = body.get("data") if isinstance(body, dict) else None
        if not isinstance(rows, list):
            return {}

        parsed: dict[str, tuple[float, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = row.get("s")
            values = row.get("d")
            if (
                isinstance(ticker, str)
                and isinstance(values, list)
                and len(values) >= 2
                and _is_valid_number(values[0])
                and float(values[0]) > 0.0
            ):
                price = float(values[0])
                change_pct = float(values[1]) if _is_valid_number(values[1]) else 0.0
                parsed[ticker] = (price, change_pct)

        return parsed

    def _fetch_yahoo_fallback(
        self,
        client: httpx.Client,
        yahoo_tickers: tuple[str, ...],
    ) -> tuple[float, float] | None:
        """Query Yahoo Finance v8 Chart endpoint for the first succeeding symbol."""
        for symbol in yahoo_tickers:
            for host in YAHOO_CHART_HOSTS:
                url = f"{host}/v8/finance/chart/{symbol}"
                try:
                    response = client.get(
                        url,
                        params={"range": "5d", "interval": "1d"},
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.debug(
                        "Yahoo Finance yedek sorgusu başarısız (%s @ %s): %s",
                        symbol,
                        host,
                        exc,
                    )
                    continue

                extracted = _extract_yahoo_price_and_change(payload)
                if extracted is not None:
                    return extracted

        logger.warning(
            "Piyasa göstergesi yedek kaynaklardan da alınamadı: %s",
            ", ".join(yahoo_tickers),
        )
        return None
