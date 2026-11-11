"""Unit tests for live market snapshot fetching, Gram Gold synthesis, and Turkish formatting."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from news_bot.market import (
    TROY_OUNCE_IN_GRAMS,
    MarketDataFetcher,
    compute_gram_gold_try,
)
from news_bot.models import MarketQuote
from news_bot.notifier import (
    TelegramNotifier,
    format_change_badge,
    format_market_snapshot,
    format_turkish_number,
)


def test_compute_gram_gold_try_calculates_exact_spot_and_compounded_24h_change() -> None:
    ons = MarketQuote(
        symbol_id="ons_gold_usd",
        icon="🥇",
        label="Ons Altın",
        price=2650.0,
        change_pct_24h=1.0,  # +1.00%
        unit_suffix=" $",
        decimals=2,
    )
    usd_try = MarketQuote(
        symbol_id="usd_try",
        icon="💵",
        label="Dolar/TL",
        price=34.85,
        change_pct_24h=0.5,  # +0.50%
        unit_suffix=" TL",
        decimals=2,
    )

    gram = compute_gram_gold_try(ons_gold=ons, usd_try=usd_try)
    expected_price = (2650.0 * 34.85) / TROY_OUNCE_IN_GRAMS
    # (1.01 * 1.005 - 1) * 100 = 1.505%
    assert gram.symbol_id == "gram_gold_try"
    assert gram.price == pytest.approx(expected_price, rel=1e-6)
    assert gram.change_pct_24h == pytest.approx(1.505, rel=1e-6)


def test_tradingview_batch_fetch_returns_all_eight_ordered_quotes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "scanner.tradingview.com"
        return httpx.Response(
            200,
            json={
                "totalCount": 8,
                "data": [
                    {"s": "BIST:XBANK", "d": [12840.6, 2.15]},
                    {"s": "FX_IDC:USDTRY", "d": [34.852, 0.18]},
                    {"s": "FX_IDC:EURTRY", "d": [37.419, -0.12]},
                    {"s": "FX_IDC:EURUSD", "d": [1.0738, -0.30]},
                    {"s": "FX_IDC:XAUTRYG", "d": [2968.45, 0.95]},
                    {"s": "TVC:GOLD", "d": [2650.2, 0.77]},
                    {"s": "TVC:UKOIL", "d": [73.45, -1.15]},
                    {"s": "BIST:XU100", "d": [9450.25, 1.42]},
                ],
            },
        )

    fetcher = MarketDataFetcher(transport=httpx.MockTransport(handler))
    quotes = fetcher.fetch_snapshot()

    assert [q.symbol_id for q in quotes] == [
        "usd_try",
        "eur_try",
        "eur_usd",
        "gram_gold_try",
        "ons_gold_usd",
        "brent_oil_usd",
        "bist_100",
        "bist_bank",
    ]
    assert quotes[0].price == pytest.approx(34.852)
    assert quotes[0].change_pct_24h == pytest.approx(0.18)
    assert quotes[3].price == pytest.approx(2968.45)
    assert quotes[7].label == "BIST Bankacılık"


def test_yahoo_fallback_and_gram_gold_synthesis_when_tradingview_fails() -> None:
    yahoo_mock_prices: dict[str, tuple[float, float]] = {
        "USDTRY=X": (34.00, 34.34),   # +1.00%
        "EURTRY=X": (37.00, 36.63),   # -1.00%
        "EURUSD=X": (1.0800, 1.0746), # -0.50%
        "GC=F": (2600.0, 2652.0),     # +2.00%
        "BZ=F": (75.00, 74.25),       # -1.00%
        "XU100.IS": (9300.0, 9486.0), # +2.00%
        "XBANK.IS": (12500.0, 12875.0),  # +3.00%
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "scanner.tradingview.com":
            return httpx.Response(503, text="Service Unavailable")

        symbol = request.url.path.rsplit("/", 1)[-1]
        if symbol not in yahoo_mock_prices:
            return httpx.Response(404, json={"chart": {"result": None}})

        prev_close, curr_close = yahoo_mock_prices[symbol]
        return httpx.Response(
            200,
            json={
                "chart": {
                    "result": [
                        {
                            "meta": {
                                "symbol": symbol,
                                "regularMarketPrice": curr_close,
                                "regularMarketTime": 1731312000,
                            },
                            "timestamp": [1731225600, 1731312000],
                            "indicators": {
                                "quote": [{"close": [prev_close, curr_close]}]
                            },
                        }
                    ]
                }
            },
        )

    fetcher = MarketDataFetcher(transport=httpx.MockTransport(handler))
    quotes = fetcher.fetch_snapshot()

    by_id = {q.symbol_id: q for q in quotes}
    assert len(quotes) == 8
    assert by_id["usd_try"].change_pct_24h == pytest.approx(1.0)
    assert by_id["ons_gold_usd"].change_pct_24h == pytest.approx(2.0)
    # Synthesized Gram Altın: (1.01 * 1.02 - 1) * 100 = 3.02%
    assert by_id["gram_gold_try"].change_pct_24h == pytest.approx(3.02)
    assert by_id["brent_oil_usd"].change_pct_24h == pytest.approx(-1.0)
    assert by_id["bist_100"].change_pct_24h == pytest.approx(2.0)


def test_market_fetcher_degrades_gracefully_on_complete_outage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network unreachable")

    fetcher = MarketDataFetcher(transport=httpx.MockTransport(handler))
    assert fetcher.fetch_snapshot() == []


def test_turkish_formatting_and_market_snapshot_card() -> None:
    assert format_turkish_number(2968.456, 2) == "2.968,46"
    assert format_turkish_number(1.0738, 4) == "1,0738"
    assert format_change_badge(1.424) == "🟢 +%1,42"
    assert format_change_badge(-0.85) == "🔴 -%0,85"
    assert format_change_badge(-0.002) == "⚪ %0,00"

    sample_quotes = [
        MarketQuote(
            symbol_id="usd_try",
            icon="💵",
            label="Dolar/TL",
            price=34.85,
            change_pct_24h=0.18,
            unit_suffix=" TL",
            decimals=2,
        ),
        MarketQuote(
            symbol_id="gram_gold_try",
            icon="🪙",
            label="Gram Altın",
            price=2968.45,
            change_pct_24h=0.95,
            unit_suffix=" TL",
            decimals=2,
        ),
        MarketQuote(
            symbol_id="brent_oil_usd",
            icon="🛢",
            label="Brent Petrol",
            price=73.45,
            change_pct_24h=-1.15,
            unit_suffix=" $",
            decimals=2,
        ),
    ]

    card = format_market_snapshot(
        sample_quotes,
        now=datetime(2026, 11, 11, 5, 0, tzinfo=timezone.utc),
    )
    assert "📊 <b>Piyasa Özeti (24 Saatlik Değişim)</b>" in card
    assert "11.11.2026 | 08:00 TSİ" in card
    assert "💵 <b>Dolar/TL:</b> 34,85 TL (🟢 +%0,18)" in card
    assert "🪙 <b>Gram Altın:</b> 2.968,45 TL (🟢 +%0,95)" in card
    assert "🛢 <b>Brent Petrol:</b> 73,45 $ (🔴 -%1,15)" in card

    notifier = TelegramNotifier(bot_token="dummy", chat_id="123", dry_run=True)
    assert notifier.send_digest([], market_quotes=sample_quotes) is True
