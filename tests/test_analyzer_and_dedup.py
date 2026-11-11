"""Unit tests for authority-aware deduplication and Gemini structured triage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from news_bot.analyzer import GeminiNewsAnalyzer
from news_bot.deduplicator import deduplicate_candidates
from news_bot.models import NewsCategory, RawArticle
from news_bot.prefilter import canonicalize_url, compute_article_id
from news_bot.storage import ArticleStorage


def _build_raw(
    url: str,
    title: str,
    *,
    source_name: str = "Kaynak",
    tier: int = 2,
    language: str = "tr",
) -> RawArticle:
    canonical = canonicalize_url(url)
    return RawArticle(
        article_id=compute_article_id(canonical),
        source_id=source_name.lower().replace(" ", "_"),
        source_name=source_name,
        authority_tier=tier,
        language=language,
        title=title,
        url=url,
        canonical_url=canonical,
        summary=title,
        published_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 11, 11, 10, 5, tzinfo=timezone.utc),
    )


def test_deduplicator_prefers_tier1_authority_and_keeps_distinct_reports(
    tmp_path: Path,
) -> None:
    storage = ArticleStorage(tmp_path / "state.db")

    media_copy = _build_raw(
        "https://portal.example.com/tcmb-faiz-1",
        "Merkez Bankası (TCMB) politika faizini yüzde 45 seviyesinde sabit bıraktı",
        source_name="Ekonomi Portalı",
        tier=2,
    )
    official_tcmb = _build_raw(
        "https://tcmb.gov.tr/duyuru-faiz-1",
        "TCMB politika faizini yüzde 45 seviyesinde sabit tuttu",
        source_name="TCMB Basın Duyuruları",
        tier=1,
    )
    distinct_tcmb_report = _build_raw(
        "https://tcmb.gov.tr/duyuru-odeme-dengesi",
        "TCMB Eylül ayı ödemeler dengesi istatistiklerini açıkladı",
        source_name="TCMB Basın Duyuruları",
        tier=1,
    )

    accepted, dropped = deduplicate_candidates(
        [media_copy, official_tcmb, distinct_tcmb_report],
        storage,
        fuzzy_threshold=80.0,
    )

    assert dropped == 1
    assert len(accepted) == 2
    accepted_sources = [a.source_name for a in accepted]
    assert "TCMB Basın Duyuruları" in accepted_sources
    assert "Ekonomi Portalı" not in accepted_sources


def test_gemini_analyzer_parses_structured_output_and_suppresses_cross_lang_duplicate() -> None:
    fed_official = _build_raw(
        "https://federalreserve.gov/newsevents/pressreleases/monetary20261111a.htm",
        "Federal Reserve issues FOMC statement keeping rates unchanged",
        source_name="Federal Reserve (Fed)",
        tier=1,
        language="en",
    )
    tr_copy = _build_raw(
        "https://bloomberght.com/fed-faiz-kararini-acikladi-1",
        "Fed faiz kararını açıkladı: Faizler sabit bırakıldı",
        source_name="Bloomberg HT",
        tier=2,
        language="tr",
    )

    mock_payload = {
        "evaluations": [
            {
                "article_id": fed_official.article_id,
                "is_relevant": True,
                "category": "WORLD_ECONOMY",
                "importance_score": 10,
                "duplicate_of_id": "",
                "turkish_headline": "Fed Politika Faizini Sabit Tuttu",
                "summary_bullets": [
                    "FOMC politika faiz aralığını değiştirmedi.",
                    "Enflasyon hedefe yaklaşana kadar temkinli duruş korunacak.",
                ],
                "impact_note": "Gelişmekte olan ülke sermaye akımları ve küresel dolar likiditesi için belirleyici.",
            },
            {
                "article_id": tr_copy.article_id,
                "is_relevant": True,
                "category": "WORLD_ECONOMY",
                "importance_score": 9,
                "duplicate_of_id": fed_official.article_id,
                "turkish_headline": "Fed Faizi Değiştirmedi",
                "summary_bullets": ["Fed faizi sabit bıraktı."],
                "impact_note": "",
            },
        ]
    }

    captured_kwargs: dict[str, object] = {}

    class DummyInteractions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(output_text=json.dumps(mock_payload, ensure_ascii=False))

    dummy_client = SimpleNamespace(interactions=DummyInteractions())
    analyzer = GeminiNewsAnalyzer(
        api_key="fake-key",
        model="gemini-3.8-flash",
        batch_size=15,
        client=dummy_client,  # type: ignore[arg-type]
    )

    results = analyzer.evaluate_articles([fed_official, tr_copy])

    assert captured_kwargs["model"] == "gemini-3.8-flash"
    assert captured_kwargs["store"] is False
    assert results[fed_official.article_id].is_relevant is True
    assert results[fed_official.article_id].category == NewsCategory.WORLD_ECONOMY
    assert results[fed_official.article_id].importance_score == 10
    assert results[fed_official.article_id].duplicate_of_id is None

    # Cross-language duplicate automatically suppressed so user isn't pinged twice!
    assert results[tr_copy.article_id].duplicate_of_id == fed_official.article_id
    assert results[tr_copy.article_id].is_relevant is False

