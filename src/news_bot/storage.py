"""SQLite state persistence for deduplication, instant alerts, and daily digests."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from news_bot.models import (
    ArticleEvaluation,
    EnrichedArticle,
    NewsCategory,
    RawArticle,
)
from news_bot.prefilter import normalize_title_for_matching

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    authority_tier INTEGER NOT NULL,
    language TEXT NOT NULL DEFAULT 'tr',
    raw_title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    raw_summary TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    is_relevant INTEGER NOT NULL,
    category TEXT NOT NULL,
    importance_score INTEGER NOT NULL,
    duplicate_of_id TEXT,
    turkish_headline TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    impact_note TEXT NOT NULL DEFAULT '',
    instant_sent INTEGER NOT NULL DEFAULT 0,
    queued_for_digest INTEGER NOT NULL DEFAULT 0,
    digest_sent INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_articles_fetched_at ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_digest_queue ON articles(queued_for_digest, digest_sent);

CREATE TABLE IF NOT EXISTS digest_runs (
    slot_key TEXT PRIMARY KEY,
    executed_at TEXT NOT NULL
);
"""


class ArticleStorage:
    """Manages persistent SQLite storage for evaluated articles and delivery state."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def get_existing_ids_and_urls(self) -> tuple[set[str], set[str]]:
        """Return sets of all stored article IDs and canonical URLs."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT article_id, canonical_url FROM articles"
            ).fetchall()
        ids = {str(r["article_id"]) for r in rows}
        urls = {str(r["canonical_url"]) for r in rows}
        return ids, urls

    def get_recent_normalized_titles(
        self, lookback_hours: int = 48
    ) -> list[tuple[str, str, int]]:
        """Return (article_id, normalized_title, authority_tier) from recent history."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT article_id, normalized_title, turkish_headline, is_relevant, authority_tier
                FROM articles
                WHERE fetched_at >= ?
                """,
                (cutoff,),
            ).fetchall()
        results: list[tuple[str, str, int]] = []
        for r in rows:
            art_id = str(r["article_id"])
            tier = int(r["authority_tier"])
            norm_raw = str(r["normalized_title"] or "")
            if norm_raw:
                results.append((art_id, norm_raw, tier))
            if int(r["is_relevant"]) == 1 and r["turkish_headline"]:
                norm_tr = normalize_title_for_matching(str(r["turkish_headline"]))
                if norm_tr and norm_tr != norm_raw:
                    results.append((art_id, norm_tr, tier))
        return results

    def get_recent_relevant_signatures(
        self, lookback_hours: int = 48
    ) -> list[tuple[str, str, str, int]]:
        """
        Return (article_id, norm_raw_title, norm_turkish_headline, authority_tier)
        for recently accepted relevant articles.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT article_id, normalized_title, turkish_headline, authority_tier
                FROM articles
                WHERE fetched_at >= ?
                  AND is_relevant = 1
                  AND category != ?
                """,
                (cutoff, NewsCategory.IRRELEVANT.value),
            ).fetchall()
        return [
            (
                str(r["article_id"]),
                str(r["normalized_title"] or ""),
                normalize_title_for_matching(str(r["turkish_headline"] or "")),
                int(r["authority_tier"]),
            )
            for r in rows
        ]

    def get_recent_relevant_headlines(
        self, lookback_hours: int = 36, limit: int = 35
    ) -> list[str]:
        """Return recent accepted Turkish headlines so Gemini can flag cross-scan duplicates."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turkish_headline
                FROM articles
                WHERE fetched_at >= ?
                  AND is_relevant = 1
                  AND category != ?
                ORDER BY fetched_at DESC
                LIMIT ?
                """,
                (cutoff, NewsCategory.IRRELEVANT.value, limit),
            ).fetchall()
        return [
            str(r["turkish_headline"]).strip()
            for r in rows
            if r["turkish_headline"] and str(r["turkish_headline"]).strip()
        ]

    def save_articles(self, items: list[EnrichedArticle]) -> None:
        """Insert or update evaluated articles and their notification flags."""
        if not items:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO articles (
                    article_id, canonical_url, url, source_id, source_name,
                    authority_tier, language, raw_title, normalized_title, raw_summary,
                    published_at, fetched_at, is_relevant, category, importance_score,
                    duplicate_of_id, turkish_headline, summary_json, impact_note,
                    instant_sent, queued_for_digest, digest_sent
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?
                )
                ON CONFLICT(article_id) DO UPDATE SET
                    instant_sent = MAX(articles.instant_sent, excluded.instant_sent),
                    queued_for_digest = MAX(articles.queued_for_digest, excluded.queued_for_digest),
                    digest_sent = MAX(articles.digest_sent, excluded.digest_sent)
                """,
                [
                    (
                        item.raw.article_id,
                        item.raw.canonical_url,
                        item.raw.url,
                        item.raw.source_id,
                        item.raw.source_name,
                        item.raw.authority_tier,
                        item.raw.language,
                        item.raw.title,
                        normalize_title_for_matching(item.raw.title),
                        item.raw.summary,
                        item.raw.published_at.isoformat()
                        if item.raw.published_at
                        else None,
                        item.raw.fetched_at.isoformat(),
                        int(item.evaluation.is_relevant),
                        item.evaluation.category.value,
                        item.evaluation.importance_score,
                        item.evaluation.duplicate_of_id,
                        item.evaluation.turkish_headline,
                        json.dumps(item.evaluation.summary_text, ensure_ascii=False),
                        item.evaluation.impact_note,
                        int(item.instant_sent),
                        int(item.queued_for_digest),
                        int(item.digest_sent),
                    )
                    for item in items
                ],
            )

    def mark_instant_sent(self, article_ids: list[str]) -> None:
        """Mark critical articles as delivered via instant notification."""
        if not article_ids:
            return
        placeholders = ",".join("?" for _ in article_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE articles SET instant_sent = 1 WHERE article_id IN ({placeholders})",
                article_ids,
            )

    def get_pending_digest_articles(self) -> list[EnrichedArticle]:
        """Fetch all relevant articles queued for the next daily digest."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM articles
                WHERE queued_for_digest = 1
                  AND digest_sent = 0
                  AND is_relevant = 1
                  AND category != ?
                ORDER BY importance_score DESC, fetched_at DESC
                """,
                (NewsCategory.IRRELEVANT.value,),
            ).fetchall()
        return [self._row_to_enriched(row) for row in rows]

    def mark_digest_sent(self, article_ids: list[str]) -> None:
        """Mark digest-queued articles as sent."""
        if not article_ids:
            return
        placeholders = ",".join("?" for _ in article_ids)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE articles SET digest_sent = 1 WHERE article_id IN ({placeholders})",
                article_ids,
            )

    def prune_older_than(self, retention_days: int = 14) -> int:
        """Delete old records so the SQLite state file stays compact."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM articles WHERE fetched_at < ?", (cutoff,)
            )
            conn.execute(
                "DELETE FROM digest_runs WHERE executed_at < ?", (cutoff,)
            )
            return cursor.rowcount

    def has_digest_slot_run(self, slot_key: str) -> bool:
        """Check whether a scheduled digest slot (e.g. 2026-11-11-morning) already ran."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM digest_runs WHERE slot_key = ?", (slot_key,)
            ).fetchone()
        return row is not None

    def record_digest_slot_run(self, slot_key: str) -> None:
        """Record completion of a scheduled digest slot."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO digest_runs (slot_key, executed_at)
                VALUES (?, ?)
                ON CONFLICT(slot_key) DO UPDATE SET executed_at = excluded.executed_at
                """,
                (slot_key, now_iso),
            )

    @staticmethod
    def _row_to_enriched(row: sqlite3.Row) -> EnrichedArticle:
        published_raw = row["published_at"]
        published_dt = (
            datetime.fromisoformat(published_raw) if published_raw else None
        )
        fetched_dt = datetime.fromisoformat(str(row["fetched_at"]))
        parsed_summary = json.loads(str(row["summary_json"]))
        if isinstance(parsed_summary, list):
            summary_text = " ".join(
                str(b).strip().lstrip("•-* ").strip()
                for b in parsed_summary
                if str(b).strip()
            )
        elif isinstance(parsed_summary, str):
            summary_text = parsed_summary.strip()
        else:
            summary_text = ""

        raw = RawArticle(
            article_id=str(row["article_id"]),
            source_id=str(row["source_id"]),
            source_name=str(row["source_name"]),
            authority_tier=int(row["authority_tier"]),
            language=str(row["language"]),
            title=str(row["raw_title"]),
            url=str(row["url"]),
            canonical_url=str(row["canonical_url"]),
            summary=str(row["raw_summary"]),
            published_at=published_dt,
            fetched_at=fetched_dt,
        )
        evaluation = ArticleEvaluation(
            article_id=str(row["article_id"]),
            is_relevant=bool(row["is_relevant"]),
            category=NewsCategory(str(row["category"])),
            importance_score=int(row["importance_score"]),
            duplicate_of_id=row["duplicate_of_id"],
            turkish_headline=str(row["turkish_headline"]),
            summary_text=summary_text,
            impact_note=str(row["impact_note"]),
        )
        return EnrichedArticle(
            raw=raw,
            evaluation=evaluation,
            instant_sent=bool(row["instant_sent"]),
            queued_for_digest=bool(row["queued_for_digest"]),
            digest_sent=bool(row["digest_sent"]),
        )
