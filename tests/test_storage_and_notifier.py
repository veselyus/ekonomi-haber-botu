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
from news_bot.pipeline import determine_due_digest_slot, select_balanced_candidates
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
        summary_text=(
            "BDDK ticari kredilerde aylık büyüme sınırını yüzde 1,5 seviyesine indirdi. "
            "Kurum TL mevduat payı hedeflerini güncelledi."
        ),
        impact_note="Bu karar banka kredi büyümesini yavaşlatır.",
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
    # 05:00 UTC == 08:00 TSİ -> Daily 08:00 Morning digest window
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 5, 0, tzinfo=timezone.utc))
        == "2026-11-11-morning"
    )
    # 06:30 UTC == 09:30 TSİ -> Still inside delayed-cron morning grace window
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 6, 30, tzinfo=timezone.utc))
        == "2026-11-11-morning"
    )
    # 12:00 UTC == 15:00 TSİ -> Intraday critical check only (no digest)
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc))
        is None
    )
    # 19:30 UTC == 22:30 TSİ -> Night critical check only (no evening digest)
    assert (
        determine_due_digest_slot(datetime(2026, 11, 11, 19, 30, tzinfo=timezone.utc))
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
    assert "Etki:" not in alert_html
    assert "aylık büyüme sınırını" not in alert_html  # No body subtext

    # Verify 3 separate category messages in requested order:
    # 1) Bankacılık  2) Türkiye Ekonomisi  3) Dünya Ekonomisi
    three_cat_items = [
        _sample_enriched(1, category=NewsCategory.WORLD_ECONOMY, score=8, queued_for_digest=True),
        _sample_enriched(2, category=NewsCategory.TR_BANKING, score=9, queued_for_digest=True),
        _sample_enriched(3, category=NewsCategory.TR_ECONOMY, score=8, queued_for_digest=True),
    ]
    three_msgs = format_digest_messages(three_cat_items)
    assert len(three_msgs) == 3
    assert "Türkiye Bankacılık Sektörü" in three_msgs[0]
    assert "Türkiye Ekonomisi" in three_msgs[1]
    assert "Dünya Ekonomisi" in three_msgs[2]
    for msg in three_msgs:
        assert "Etki:" not in msg
        assert "aylık büyüme sınırını" not in msg

    # Verify single large category also chunks below Telegram 4000-char limit
    many_items = [
        _sample_enriched(
            i,
            category=NewsCategory.TR_BANKING,
            score=8,
            queued_for_digest=True,
        )
        for i in range(1, 50)
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


def test_tr_economy_header_never_disappears_and_balanced_selection_prevents_starvation() -> None:
    # Even if a scan only found TR_BANKING and WORLD_ECONOMY items,
    # include_empty_categories=True keeps all 3 headers visible!
    partial_items = [
        _sample_enriched(1, category=NewsCategory.TR_BANKING, score=8, queued_for_digest=True),
        _sample_enriched(2, category=NewsCategory.WORLD_ECONOMY, score=8, queued_for_digest=True),
    ]
    msgs = format_digest_messages(partial_items, include_empty_categories=True)
    assert len(msgs) == 3
    assert "Türkiye Bankacılık Sektörü" in msgs[0]
    assert "Türkiye Ekonomisi" in msgs[1]
    assert "yeni bir gelişme bulunmadı" in msgs[1]
    assert "Dünya Ekonomisi" in msgs[2]

    # Balanced candidate selection prevents 30 WORLD_ECONOMY items from starving TR_ECONOMY
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    world_flood = [
        RawArticle(
            article_id=f"w_{i}",
            source_id="cnbc",
            source_name="CNBC",
            default_category=NewsCategory.WORLD_ECONOMY,
            authority_tier=2,
            title=f"Global story {i}",
            url=f"https://cnbc.com/{i}",
            canonical_url=f"https://cnbc.com/{i}",
            published_at=now,
            fetched_at=now,
        )
        for i in range(30)
    ]
    tr_econ_undated = [
        RawArticle(
            article_id=f"tr_{i}",
            source_id="ekonomim",
            source_name="Ekonomim",
            default_category=NewsCategory.TR_ECONOMY,
            authority_tier=2,
            title=f"Türkiye ekonomi haberi {i}",
            url=f"https://ekonomim.com/{i}",
            canonical_url=f"https://ekonomim.com/{i}",
            published_at=None,  # Even without pubDate, TR_ECONOMY must get its quota!
            fetched_at=now,
        )
        for i in range(10)
    ]

    selected = select_balanced_candidates(world_flood + tr_econ_undated, max_total=30)
    tr_selected = [x for x in selected if x.default_category == NewsCategory.TR_ECONOMY]
    assert len(tr_selected) == 10

