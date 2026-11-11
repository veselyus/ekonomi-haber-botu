"""End-to-end orchestration for scanning critical news and dispatching daily digests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timezone

from news_bot.analyzer import GeminiNewsAnalyzer, enforce_category_domain_rules
from news_bot.config import Settings
from news_bot.deduplicator import (
    deduplicate_candidates,
    deduplicate_enriched_articles,
    suppress_post_eval_duplicates,
)
from news_bot.fetcher import NewsFetcher
from news_bot.market import MarketDataFetcher
from news_bot.models import EnrichedArticle, MarketQuote, NewsCategory, RawArticle
from news_bot.notifier import CATEGORY_ORDER, TURKEY_TZ, TelegramNotifier, format_market_snapshot
from news_bot.prefilter import prefilter_articles
from news_bot.sources import load_active_sources
from news_bot.storage import ArticleStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanStats:
    """Telemetry summary of a single scan run."""

    fetched_count: int
    prefiltered_dropped: int
    dedup_dropped: int
    evaluated_count: int
    critical_sent: int
    queued_for_digest: int
    ignored_irrelevant: int


def _candidate_sort_key(item: RawArticle) -> tuple[int, float]:
    """Sort key preferring higher authority tier and fresher timestamps (graceful on missing pubDate)."""
    effective_ts = (
        item.published_at.timestamp()
        if item.published_at is not None
        else (item.fetched_at.timestamp() - 3600.0)
    )
    return (item.authority_tier, -effective_ts)


def select_balanced_candidates(
    candidates: list[RawArticle],
    *,
    max_total: int,
) -> list[RawArticle]:
    """
    Select up to max_total candidates with balanced representation across
    TR_BANKING, TR_ECONOMY, and WORLD_ECONOMY so high-frequency global feeds
    can never starve Turkish economy or banking sources.
    """
    if len(candidates) <= max_total:
        return sorted(candidates, key=_candidate_sort_key)

    per_bucket = max(1, max_total // len(CATEGORY_ORDER))
    selected: list[RawArticle] = []
    leftovers: list[RawArticle] = []

    for category in CATEGORY_ORDER:
        bucket = sorted(
            (c for c in candidates if c.default_category == category),
            key=_candidate_sort_key,
        )
        selected.extend(bucket[:per_bucket])
        leftovers.extend(bucket[per_bucket:])

    other_items = sorted(
        (c for c in candidates if c.default_category not in CATEGORY_ORDER),
        key=_candidate_sort_key,
    )
    leftovers.extend(other_items)
    leftovers.sort(key=_candidate_sort_key)

    remaining_slots = max_total - len(selected)
    if remaining_slots > 0:
        selected.extend(leftovers[:remaining_slots])

    return selected


def determine_due_digest_slot(now_utc: datetime | None = None) -> str | None:
    """
    Determine whether current Turkey time (UTC+3) falls inside the daily 08:00 morning digest window.

    Using a 07:45-10:45 TSİ window ensures GitHub Actions cron queue delays never
    cause a missed 08:00 morning bulletin, while keeping the rest of the day silent
    except for extraordinary (10/10) critical alerts.
    """
    local_dt = (now_utc or datetime.now(timezone.utc)).astimezone(TURKEY_TZ)
    current_t = local_dt.timetz().replace(tzinfo=None)
    date_prefix = local_dt.strftime("%Y-%m-%d")

    if time(7, 45) <= current_t < time(10, 45):
        return f"{date_prefix}-morning"
    return None


class NewsPipeline:
    """Coordinates fetching, zero-noise filtering, Gemini scoring, storage, and Telegram."""

    def __init__(self, settings: Settings, *, dry_run: bool = False) -> None:
        self._settings = settings
        self._dry_run = dry_run
        self._storage = ArticleStorage(settings.database_path)
        self._notifier = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
            dry_run=dry_run,
            timeout_seconds=settings.http_timeout_seconds,
        )
        self._market_fetcher = MarketDataFetcher(
            timeout_seconds=settings.http_timeout_seconds
        )
        self._dry_run_digest_buffer: list[EnrichedArticle] = []

    def run_fetch_preview(self) -> ScanStats:
        """Fetch and run local pre-filter + deduplication without calling Gemini or Telegram."""
        sources = load_active_sources()
        fetcher = NewsFetcher(timeout_seconds=self._settings.http_timeout_seconds)
        raw_articles = fetcher.fetch_all(sources)

        kept_after_prefilter, prefiltered_dropped = prefilter_articles(
            raw_articles,
            max_age_hours=self._settings.max_article_age_hours,
        )
        candidates, dedup_dropped = deduplicate_candidates(
            kept_after_prefilter,
            self._storage,
            fuzzy_threshold=self._settings.fuzzy_duplicate_threshold,
        )

        candidates = select_balanced_candidates(
            candidates,
            max_total=self._settings.max_candidates_per_scan,
        )
        logger.info(
            "Önizleme Özeti -> Çekilen: %d | Clickbait/Eski Atılan: %d | Mükerrer Atılan: %d | Gemini Adayı: %d",
            len(raw_articles),
            prefiltered_dropped,
            dedup_dropped,
            len(candidates),
        )
        for idx, item in enumerate(candidates[:15], start=1):
            logger.info(
                "  [%02d] (Tier %d | %s) %s",
                idx,
                item.authority_tier,
                item.source_name,
                item.title,
            )

        if self._settings.include_market_snapshot:
            market_quotes = self._market_fetcher.fetch_snapshot()
            if market_quotes:
                logger.info(
                    "Piyasa Özeti Önizleme:\n%s",
                    format_market_snapshot(market_quotes),
                )

        return ScanStats(
            fetched_count=len(raw_articles),
            prefiltered_dropped=prefiltered_dropped,
            dedup_dropped=dedup_dropped,
            evaluated_count=0,
            critical_sent=0,
            queued_for_digest=0,
            ignored_irrelevant=0,
        )

    def run_scan(self) -> ScanStats:
        """
        Fetch sources, eliminate noise/duplicates, score via Gemini,
        send critical alerts (score >= 9) immediately, and queue 7-8 items for digest.
        """
        sources = load_active_sources()
        fetcher = NewsFetcher(timeout_seconds=self._settings.http_timeout_seconds)
        raw_articles = fetcher.fetch_all(sources)

        kept_after_prefilter, prefiltered_dropped = prefilter_articles(
            raw_articles,
            max_age_hours=self._settings.max_article_age_hours,
        )
        candidates, dedup_dropped = deduplicate_candidates(
            kept_after_prefilter,
            self._storage,
            fuzzy_threshold=self._settings.fuzzy_duplicate_threshold,
        )

        candidates = select_balanced_candidates(
            candidates,
            max_total=self._settings.max_candidates_per_scan,
        )

        logger.info(
            "Huni Özeti -> Çekilen: %d | Ön Eleme Atılan: %d | Mükerrer Atılan: %d | Yeni Aday: %d",
            len(raw_articles),
            prefiltered_dropped,
            dedup_dropped,
            len(candidates),
        )

        if not candidates:
            if not self._dry_run:
                pruned = self._storage.prune_older_than(self._settings.retention_days)
                if pruned:
                    logger.info("Eski %d kayıt veritabanından temizlendi.", pruned)
            return ScanStats(
                fetched_count=len(raw_articles),
                prefiltered_dropped=prefiltered_dropped,
                dedup_dropped=dedup_dropped,
                evaluated_count=0,
                critical_sent=0,
                queued_for_digest=0,
                ignored_irrelevant=0,
            )

        analyzer = GeminiNewsAnalyzer(
            api_key=self._settings.gemini_api_key,
            model=self._settings.gemini_model,
            batch_size=self._settings.batch_size,
        )
        recent_headlines = self._storage.get_recent_relevant_headlines()
        evaluations_by_id = analyzer.evaluate_articles(
            candidates,
            recent_headlines=recent_headlines,
        )
        evaluations_by_id, post_dedup_dropped = suppress_post_eval_duplicates(
            candidates,
            evaluations_by_id,
            self._storage,
            fuzzy_threshold=self._settings.fuzzy_duplicate_threshold,
        )
        dedup_dropped += post_dedup_dropped

        enriched_all: list[EnrichedArticle] = []
        critical_to_send: list[EnrichedArticle] = []
        queued_items: list[EnrichedArticle] = []
        ignored_count = 0

        for raw_item in candidates:
            ev = evaluations_by_id[raw_item.article_id]
            is_valid_domain = (
                ev.is_relevant and ev.category != NewsCategory.IRRELEVANT
            )

            if (
                is_valid_domain
                and ev.importance_score >= self._settings.critical_score_threshold
            ):
                record = EnrichedArticle(
                    raw=raw_item,
                    evaluation=ev,
                    instant_sent=False,
                    queued_for_digest=False,
                    digest_sent=False,
                )
                critical_to_send.append(record)
                enriched_all.append(record)
            elif (
                is_valid_domain
                and ev.importance_score >= self._settings.digest_score_threshold
            ):
                record = EnrichedArticle(
                    raw=raw_item,
                    evaluation=ev,
                    instant_sent=False,
                    queued_for_digest=True,
                    digest_sent=False,
                )
                queued_items.append(record)
                enriched_all.append(record)
            else:
                record = EnrichedArticle(
                    raw=raw_item,
                    evaluation=ev,
                    instant_sent=False,
                    queued_for_digest=False,
                    digest_sent=False,
                )
                ignored_count += 1
                enriched_all.append(record)

        if self._dry_run:
            self._dry_run_digest_buffer.extend(queued_items)
        else:
            self._storage.save_articles(enriched_all)

        critical_to_send.sort(
            key=lambda x: x.evaluation.importance_score, reverse=True
        )
        delivered_ids = self._notifier.send_critical_alerts(critical_to_send)
        if delivered_ids and not self._dry_run:
            self._storage.mark_instant_sent(delivered_ids)

        if not self._dry_run:
            pruned = self._storage.prune_older_than(self._settings.retention_days)
            if pruned:
                logger.info("Eski %d kayıt veritabanından temizlendi.", pruned)

        stats = ScanStats(
            fetched_count=len(raw_articles),
            prefiltered_dropped=prefiltered_dropped,
            dedup_dropped=dedup_dropped,
            evaluated_count=len(candidates),
            critical_sent=len(delivered_ids),
            queued_for_digest=len(queued_items),
            ignored_irrelevant=ignored_count,
        )
        logger.info(
            "Tarama Tamamlandı -> Kritik Gönderilen: %d | Bültene Eklenen: %d | Elenen Gürültü: %d",
            stats.critical_sent,
            stats.queued_for_digest,
            stats.ignored_irrelevant,
        )
        return stats

    def run_digest(self) -> int:
        """Send queued important articles (score 7-8) as a categorized bulletin."""
        pending = list(self._storage.get_pending_digest_articles())
        if self._dry_run and self._dry_run_digest_buffer:
            existing_ids = {x.raw.article_id for x in pending}
            pending.extend(
                x
                for x in self._dry_run_digest_buffer
                if x.raw.article_id not in existing_ids
            )
            pending.sort(
                key=lambda x: x.evaluation.importance_score, reverse=True
            )

        # Re-verify domain rules on queued items (protects against any legacy misclassified cache rows)
        cleaned_pending: list[EnrichedArticle] = []
        for item in pending:
            repaired_ev = enforce_category_domain_rules(item.raw, item.evaluation)
            if repaired_ev.is_relevant and repaired_ev.category != NewsCategory.IRRELEVANT:
                cleaned_pending.append(
                    item.model_copy(update={"evaluation": repaired_ev})
                )

        unique_pending = deduplicate_enriched_articles(
            cleaned_pending,
            fuzzy_threshold=self._settings.fuzzy_duplicate_threshold,
        )
        logger.info(
            "Bülten kuyruğu -> Ham: %d | Tekilleştirme Sonrası: %d önemli haber.",
            len(pending),
            len(unique_pending),
        )

        market_quotes: list[MarketQuote] = []
        if self._settings.include_market_snapshot:
            market_quotes = self._market_fetcher.fetch_snapshot()

        sent_ok = self._notifier.send_digest(
            unique_pending,
            market_quotes=market_quotes,
            send_empty=self._settings.send_empty_digest,
        )
        if sent_ok and pending and not self._dry_run:
            self._storage.mark_digest_sent([item.raw.article_id for item in pending])
            logger.info(
                "%d tekil haber bülten olarak gönderildi (%d kayıt işlendi).",
                len(unique_pending),
                len(pending),
            )
            return len(unique_pending)

        return len(unique_pending) if (sent_ok and unique_pending) else 0

    def run_auto(self, now_utc: datetime | None = None) -> None:
        """
        Scheduled cloud mode: always scan for critical alerts, and automatically
        dispatch morning (08:30 TSİ) or evening (18:30 TSİ) digest once per window.
        """
        self.run_scan()
        slot_key = determine_due_digest_slot(now_utc=now_utc)
        if slot_key is None:
            return
        if not self._dry_run and self._storage.has_digest_slot_run(slot_key):
            logger.info("Bülten dilimi (%s) daha önce çalıştırıldı; atlanıyor.", slot_key)
            return

        logger.info("Bülten zaman dilimi aktif (%s); günlük özet hazırlanıyor...", slot_key)
        self.run_digest()
        if not self._dry_run:
            self._storage.record_digest_slot_run(slot_key)
