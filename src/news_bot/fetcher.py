"""Resilient RSS and official institution announcement fetcher."""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import feedparser
import httpx
from bs4 import BeautifulSoup

from news_bot.models import NewsSource, RawArticle, SourceType
from news_bot.prefilter import canonicalize_url, clean_html_text, compute_article_id

logger = logging.getLogger(__name__)

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "application/rss+xml, application/atom+xml, application/xml, "
        "text/xml, text/html;q=0.9, */*;q=0.8"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _extract_entry_datetime(entry: feedparser.FeedParserDict) -> datetime | None:
    """Parse RSS/Atom entry publication timestamp into timezone-aware UTC datetime."""
    for struct_key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_struct = entry.get(struct_key)
        if parsed_struct is not None:
            try:
                timestamp = calendar.timegm(parsed_struct)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (OverflowError, TypeError, ValueError):
                continue

    for text_key in ("published", "updated"):
        raw_date = entry.get(text_key)
        if isinstance(raw_date, str) and raw_date.strip():
            try:
                dt = parsedate_to_datetime(raw_date.strip())
                return (
                    dt.astimezone(timezone.utc)
                    if dt.tzinfo is not None
                    else dt.replace(tzinfo=timezone.utc)
                )
            except (TypeError, ValueError):
                continue
    return None


class NewsFetcher:
    """Fetches raw articles across RSS feeds and official regulator pages."""

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._timeout = timeout_seconds

    def fetch_all(
        self, sources: list[NewsSource], max_items_per_source: int = 25
    ) -> list[RawArticle]:
        """Fetch articles from all enabled sources without failing on single-source errors."""
        collected: list[RawArticle] = []
        with httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=self._timeout,
            follow_redirects=True,
        ) as client:
            for source in sources:
                try:
                    items = self._fetch_source(
                        client, source, max_items=max_items_per_source
                    )
                    logger.info(
                        "Kaynak tarandı: %-28s -> %d içerik alındı.",
                        source.name,
                        len(items),
                    )
                    collected.extend(items)
                except httpx.HTTPError as exc:
                    logger.warning(
                        "Kaynak ağ hatası nedeniyle atlandı (%s): %s",
                        source.name,
                        exc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Kaynak ayrıştırma hatası nedeniyle atlandı (%s): %s",
                        source.name,
                        exc,
                    )
        return collected

    def _fetch_source(
        self,
        client: httpx.Client,
        source: NewsSource,
        *,
        max_items: int,
    ) -> list[RawArticle]:
        response = client.get(source.url)
        response.raise_for_status()
        fetched_at = datetime.now(timezone.utc)

        if source.source_type == SourceType.RSS:
            rss_items = self._parse_rss_feed(
                response.content, source, fetched_at=fetched_at, max_items=max_items
            )
            if rss_items or not source.url.rstrip("/").endswith("/feed"):
                return rss_items
            homepage_url = source.url.rstrip("/")[:-5] + "/"
            home_resp = client.get(homepage_url)
            home_resp.raise_for_status()
            return self._parse_official_links(
                home_resp.text,
                source.model_copy(update={"url": homepage_url}),
                fetched_at=fetched_at,
                link_selector="article a[href], h2 a[href], h3 a[href]",
                max_items=max_items,
            )
        if source.source_type == SourceType.BDDK_ANNOUNCEMENTS:
            return self._parse_official_links(
                response.text,
                source,
                fetched_at=fetched_at,
                link_selector=(
                    "a[href*='Duyuru/Detay'], a[href*='Mevzuat/Detay'], "
                    "a[href*='Icerik'], a[href*='duyuru']"
                ),
                max_items=max_items,
            )
        if source.source_type == SourceType.TCMB_ANNOUNCEMENTS:
            return self._parse_official_links(
                response.text,
                source,
                fetched_at=fetched_at,
                link_selector=(
                    "a[href*='Basin+Duyurulari/20'], a[href*='DUY20'], "
                    "a[href*='duyurular/basin']"
                ),
                max_items=max_items,
            )
        return []

    def _parse_rss_feed(
        self,
        raw_bytes: bytes,
        source: NewsSource,
        *,
        fetched_at: datetime,
        max_items: int,
    ) -> list[RawArticle]:
        parsed = feedparser.parse(raw_bytes)
        articles: list[RawArticle] = []
        for entry in parsed.entries[:max_items]:
            title = clean_html_text(str(entry.get("title", "")), max_chars=300)
            link = str(entry.get("link", "")).strip()
            if not title or not link:
                continue

            canonical = canonicalize_url(urljoin(source.url, link))
            if not canonical:
                continue

            raw_summary = str(
                entry.get("summary") or entry.get("description") or ""
            )
            summary = clean_html_text(raw_summary, max_chars=600)
            published_at = _extract_entry_datetime(entry)

            articles.append(
                RawArticle(
                    article_id=compute_article_id(canonical),
                    source_id=source.id,
                    source_name=source.name,
                    authority_tier=source.authority_tier,
                    language=source.language,
                    title=title,
                    url=urljoin(source.url, link),
                    canonical_url=canonical,
                    summary=summary,
                    published_at=published_at,
                    fetched_at=fetched_at,
                )
            )
        return articles

    def _parse_official_links(
        self,
        html_text: str,
        source: NewsSource,
        *,
        fetched_at: datetime,
        link_selector: str,
        max_items: int,
    ) -> list[RawArticle]:
        """Extract official press release links from institutional announcement pages."""
        soup = BeautifulSoup(html_text, "html.parser")
        seen_urls: set[str] = set()
        articles: list[RawArticle] = []

        for anchor in soup.select(link_selector):
            href = anchor.get("href")
            if not isinstance(href, str) or not href.strip():
                continue
            full_url = urljoin(source.url, href.strip())
            canonical = canonicalize_url(full_url)
            if not canonical or canonical == canonicalize_url(source.url):
                continue
            if canonical in seen_urls:
                continue

            title = clean_html_text(anchor.get_text(" "), max_chars=300)
            if len(title) < 15:
                continue

            seen_urls.add(canonical)
            articles.append(
                RawArticle(
                    article_id=compute_article_id(canonical),
                    source_id=source.id,
                    source_name=source.name,
                    authority_tier=source.authority_tier,
                    language=source.language,
                    title=title,
                    url=full_url,
                    canonical_url=canonical,
                    summary=f"{source.name} resmi duyurusu: {title}",
                    published_at=None,
                    fetched_at=fetched_at,
                )
            )
            if len(articles) >= max_items:
                break

        return articles
