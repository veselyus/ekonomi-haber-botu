"""Telegram HTML formatter and message delivery client with per-category messages."""

from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timedelta, timezone

import httpx

from news_bot.models import EnrichedArticle, MarketQuote, NewsCategory

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4000  # Margin below Telegram's hard 4096-char limit
TURKEY_TZ = timezone(timedelta(hours=3))

# User-requested category message order:
# 1) Bankacılık Sektörü  2) Türkiye Ekonomisi  3) Dünya Ekonomisi
CATEGORY_ORDER: tuple[NewsCategory, ...] = (
    NewsCategory.TR_BANKING,
    NewsCategory.TR_ECONOMY,
    NewsCategory.WORLD_ECONOMY,
)


def _clean_single_line(raw_text: str) -> str:
    """Normalize headline text into a clean single line without trailing punctuation clutter."""
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    return re.sub(r"^(?:[•\-*]|\d+[.)])\s*", "", cleaned).strip()


def format_turkish_number(value: float, decimals: int = 2) -> str:
    """Format a float with Turkish '.' thousands and ',' decimal separators."""
    us_formatted = f"{value:,.{decimals}f}"
    return us_formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def format_change_badge(change_pct: float) -> str:
    """Render 24-hour percentage change with directional trend icon in Turkish format."""
    rounded = round(change_pct, 2)
    abs_str = format_turkish_number(abs(rounded), 2)
    if rounded > 0:
        return f"🟢 +%{abs_str}"
    if rounded < 0:
        return f"🔴 -%{abs_str}"
    return f"⚪ %{abs_str}"


def format_market_snapshot(
    quotes: list[MarketQuote],
    *,
    now: datetime | None = None,
) -> str:
    """Format live financial indicators and 24-hour % changes as a Telegram HTML card."""
    if not quotes:
        return ""

    local_now = (now or datetime.now(timezone.utc)).astimezone(TURKEY_TZ)
    timestamp_str = local_now.strftime("%d.%m.%Y | %H:%M TSİ")

    lines: list[str] = [
        "📊 <b>Piyasa Özeti (24 Saatlik Değişim)</b>",
        f"🗓 <i>{timestamp_str}</i>",
        "━━━━━━━━━━━━━━━━━━━━",
    ]

    for quote in quotes:
        safe_label = html.escape(quote.label)
        safe_suffix = html.escape(quote.unit_suffix)
        price_str = format_turkish_number(quote.price, quote.decimals)
        badge = format_change_badge(quote.change_pct_24h)
        lines.append(
            f"{quote.icon} <b>{safe_label}:</b> {price_str}{safe_suffix} ({badge})"
        )

    return "\n".join(lines)


def format_critical_alert(item: EnrichedArticle) -> str:
    """Format an instant critical news alert (score 10) with headline and link only."""
    ev = item.evaluation
    category_label = html.escape(ev.category.display_title)
    headline = html.escape(_clean_single_line(ev.turkish_headline or item.raw.title))
    source = html.escape(item.raw.source_name)
    safe_url = html.escape(item.raw.url, quote=True)

    return (
        f"🚨 <b>KRİTİK HABER | {category_label}</b>\n\n"
        f'{headline} — <a href="{safe_url}">{source}</a>'
    )


def format_digest_messages(
    items: list[EnrichedArticle],
    *,
    now: datetime | None = None,
) -> list[str]:
    """
    Format the daily digest as 3 separate category messages:
    1) Türkiye Bankacılık Sektörü
    2) Türkiye Ekonomisi
    3) Dünya Ekonomisi

    Each item contains only its explanatory ASD-STE100 headline and clickable source link.
    """
    if not items:
        return []

    local_now = (now or datetime.now(timezone.utc)).astimezone(TURKEY_TZ)
    date_str = local_now.strftime("%d.%m.%Y")

    all_messages: list[str] = []

    for category in CATEGORY_ORDER:
        cat_items = [x for x in items if x.evaluation.category == category]
        if not cat_items:
            continue

        cat_header = (
            f"<b>{html.escape(category.display_title)}</b>\n"
            f"🗓 <i>{date_str} | Günlük Bülten</i>\n"
            "━━━━━━━━━━━━━━━━━━━━"
        )
        entry_blocks: list[str] = [cat_header]

        for item in cat_items:
            ev = item.evaluation
            headline = html.escape(
                _clean_single_line(ev.turkish_headline or item.raw.title)
            )
            source = html.escape(item.raw.source_name)
            safe_url = html.escape(item.raw.url, quote=True)
            entry_blocks.append(
                f'🔹 {headline} — <a href="{safe_url}">{source}</a>'
            )

        category_chunks = _pack_blocks_into_messages(
            entry_blocks, max_len=TELEGRAM_MAX_MESSAGE_LENGTH
        )
        all_messages.extend(category_chunks)

    return all_messages


def _pack_blocks_into_messages(blocks: list[str], *, max_len: int) -> list[str]:
    """Pack formatted text blocks into chunks guaranteed to fit within max_len."""
    messages: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for block in blocks:
        block_len = len(block) + (2 if current_parts else 0)
        if current_parts and current_len + block_len > max_len:
            messages.append("\n\n".join(current_parts))
            current_parts = [block.lstrip()]
            current_len = len(current_parts[0])
        else:
            current_parts.append(block if not current_parts else block.lstrip("\n"))
            current_len += block_len

    if current_parts:
        messages.append("\n\n".join(current_parts))

    return messages


def parse_chat_ids(raw_chat_ids: str) -> list[str]:
    """Parse one or more comma/semicolon-separated Telegram chat IDs."""
    return [
        part.strip()
        for part in raw_chat_ids.replace(";", ",").split(",")
        if part.strip()
    ]


class TelegramNotifier:
    """Delivers formatted HTML alerts and per-category digests via Telegram Bot API."""

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        dry_run: bool = False,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._bot_token = bot_token.strip()
        self._chat_ids = parse_chat_ids(chat_id)
        self._dry_run = dry_run
        self._timeout = timeout_seconds

    def send_critical_alerts(self, items: list[EnrichedArticle]) -> list[str]:
        """Send instant alerts for critical items and return IDs successfully delivered."""
        delivered_ids: list[str] = []
        for item in items:
            html_message = format_critical_alert(item)
            if self._send_html(html_message):
                delivered_ids.append(item.raw.article_id)
        return delivered_ids

    def send_digest(
        self,
        items: list[EnrichedArticle],
        *,
        market_quotes: list[MarketQuote] | None = None,
        send_empty: bool = False,
    ) -> bool:
        """Send live market snapshot card followed by separate category digest messages."""
        messages: list[str] = []
        if market_quotes:
            snapshot_msg = format_market_snapshot(market_quotes)
            if snapshot_msg:
                messages.append(snapshot_msg)

        if items:
            messages.extend(format_digest_messages(items))
        elif send_empty and not messages:
            messages.append(
                "📊 <b>Günlük Ekonomi &amp; Bankacılık Bülteni</b>\n"
                "Son 24 saatte eşik değerini aşan yeni bir gelişme bulunmadı."
            )

        if not messages:
            logger.info("Bülten kuyruğunda gönderilecek yeni haber yok; sessiz kalındı.")
            return False

        for index, msg in enumerate(messages):
            if index > 0 and not self._dry_run:
                time.sleep(0.3)
            if not self._send_html(msg):
                return False
        return True

    def _send_html(self, html_text: str) -> bool:
        if self._dry_run:
            logger.info("DRY-RUN Telegram Mesajı:\n%s\n%s", "-" * 50, html_text)
            return True

        if not self._chat_ids:
            logger.error("Geçerli hiçbir TELEGRAM_CHAT_ID bulunamadı.")
            return False

        endpoint = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        any_succeeded = False

        with httpx.Client(timeout=self._timeout) as client:
            for target_chat_id in self._chat_ids:
                payload = {
                    "chat_id": target_chat_id,
                    "text": html_text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                }
                try:
                    response = client.post(endpoint, json=payload)
                    response.raise_for_status()
                    any_succeeded = True
                except httpx.HTTPStatusError as exc:
                    logger.error(
                        "Telegram API HTTP hatası (chat_id=%s, kod=%s): %s",
                        target_chat_id,
                        exc.response.status_code,
                        exc.response.text,
                    )
                except httpx.HTTPError as exc:
                    logger.error(
                        "Telegram bağlantı hatası (chat_id=%s): %s",
                        target_chat_id,
                        exc,
                    )

        return any_succeeded
