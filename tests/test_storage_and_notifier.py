"""Unit tests for SQLite persistence, digest window scheduling, and Telegram formatting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from news_bot.models import (
    ArticleEvaluation,
    EnrichedArticle,
    NewsCategory,
    RawArticle,
)
from news_bot.notifier import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    format_critical_alert,
    format_digest_messages,
    parse_chat_ids,
)
from news_bot.pipeline import determine_due_digest_slot
from news_bot.storage import ArticleStorage


def _sample_enriched(
    idx: int,
    *,
    category: NewsCategory,
    score: int,
    queued_for_digest: bool,
) -> EnrichedArticle:
    now = datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc)
    raw = RawArticle(
        article_id=f"id_{idx:04d}",
        source_id="bddk",
        source_name="BDDK & Kurum <Test>",
        authority_tier=1,
        title=f"Ham Başlık {idx}",
        url=f"https://www.bddk.org.tr/duyuru/{idx}?a=1&b=2",
        canonical_url=f"https://www.bddk.org.tr/duyuru/{idx}?a=1&b=2",
        summary="Özet",
        published_at=now,
        fetched_at=now,
    )
    ev = ArticleEvaluation(
        article_id=raw.article_id,
        is_relevant=True,
        category=category,
        importance_score=score,
        duplicate_of_id=None,
        turkish_headline=f"BDDK <Kredi & Mevduat> Düzenlemesi #{idx}",
        summary_bullets=[
            "Ticari kredilerde büyüme sınırı %2'den %1,5'e çekildi.",
            "TL mevduat payı hedefleri güncellendi.",
        ],
        impact_note="Bankacılık net faiz marjı üzerinde doğrudan etkili.",
    )
    return EnrichedArticle(
        raw=raw,
        evaluation=ev,
        instant_sent=False,
        queued_for_digest=queued_for_digest,
        digest_sent=False,
    )


def test_storage_digest_lifecycle_and_slot_tracking(tmp_path: Path) -> None:
    storage = ArticleStorage(tmp_path / "bot.db")

    critical_item = _sample_enriched(
        1, category=NewsCategory.TR_BANKING, score=9, queued_for_digest=False
    )
    digest_item = _sample_enriched(
        2, category=NewsCategory.TR_ECONOMY, score=8, queued_for_digest=True
    )

    storage.save_articles([critical_item, digest_item])
    storage.mark_instant_sent([critical_item.raw.article_id])

    pending = storage.get_pending_digest_articles()
    assert len(pending) == 1
    assert pending[0].raw.article_id == "id_0002"

    storage.mark_digest_sent(["id_0002"])
    assert storage.get_pending_digest_articles() == []

    slot = "2026-11-11-morning"
    assert storage.has_digest_slot_run(slot) is False
    storage.record_digest_slot_run(slot)
    assert storage.has_digest_slot_run(slot) is True


def test_determine_due_digest_slot_respects_turkey_timezone() -> None:
    # 06:00 UTC == 09:00 TSİ -> Morning digest window
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 6, 0, tzinfo=timezone.utc))
        == "2026-11-11-morning"
    )
    # 16:00 UTC == 19:00 TSİ -> Evening digest window
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc))
        == "2026-11-11-evening"
    )
    # 11:00 UTC == 14:00 TSİ -> Outside digest windows
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 11, 0, tzinfo=timezone.utc))
        is None
    )


def test_telegram_formatters_escape_html_and_chunk_long_digests() -> None:
    item = _sample_enriched(
        1, category=NewsCategory.TR_BANKING, score=10, queued_for_digest=False
    )
    alert_html = format_critical_alert(item)
    assert "&lt;Kredi &amp; Mevduat&gt;" in alert_html
    assert "BDDK &amp; Kurum &lt;Test&gt;" in alert_html
    assert 'href="https://www.bddk.org.tr/duyuru/1?a=1&amp;b=2"' in alert_html

    many_items = [
        _sample_enriched(
            i,
            category=NewsCategory.WORLD_ECONOMY if i % 2 else NewsCategory.TR_ECONOMY,
            score=8,
            queued_for_digest=True,
        )
        for i in range(1, 35)
    ]
    chunks = format_digest_messages(many_items)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MAX_MESSAGE_LENGTH


def test_parse_chat_ids_supports_multiple_friends_and_channels() -> None:
    assert parse_chat_ids("123456, -100987654321 ; 777888") == [
        "123456",
        "-100987654321",
        "777888",
    ]

