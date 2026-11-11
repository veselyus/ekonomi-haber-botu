"""Telegram HTML formatter and message delivery client with auto-chunking."""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx

from news_bot.models import EnrichedArticle, NewsCategory

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4000  # Margin below Telegram's hard 4096-char limit
TURKEY_TZ = timezone(timedelta(hours=3))

CATEGORY_ORDER: tuple[NewsCategory, ...] = (
    NewsCategory.TR_ECONOMY,
    NewsCategory.TR_BANKING,
    NewsCategory.WORLD_ECONOMY,
)


def _normalize_plain_paragraph(raw_text: str) -> str:
    """Strip any stray bullet markers or line breaks into a single clean prose paragraph."""
    if not raw_text.strip():
        return ""
    cleaned_lines = [
        re.sub(r"^(?:[•\-*]|\d+[.)])\s*", "", line.strip())
        for line in raw_text.splitlines()
        if line.strip()
    ]
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def format_critical_alert(item: EnrichedArticle) -> str:
    """Format an instant critical news alert (score 10) in plain ASD-STE100 Telegram HTML."""
    ev = item.evaluation
    category_label = html.escape(ev.category.display_title)
    headline = html.escape(ev.turkish_headline or item.raw.title)
    source = html.escape(item.raw.source_name)
    safe_url = html.escape(item.raw.url, quote=True)
    paragraph = html.escape(_normalize_plain_paragraph(ev.summary_text))

    lines: list[str] = [
        f"🚨 <b>KRİTİK HABER | {category_label}</b>",
        f"<b>{headline}</b>",
    ]
    if paragraph:
        lines.append("")
        lines.append(paragraph)

    if ev.impact_note.strip():
        lines.append("")
        lines.append(
            f"💡 <i>Etki: {html.escape(_normalize_plain_paragraph(ev.impact_note))}</i>"
        )

    lines.append("")
    lines.append(
        f'📌 Kaynak: <b>{source}</b> | <a href="{safe_url}">Habere Git</a>'
    )
    return "\n".join(lines)


def format_digest_messages(
    items: list[EnrichedArticle],
    *,
    now: datetime | None = None,
) -> list[str]:
    """
    Format categorized daily digest messages in clean, bullet-free ASD-STE100 paragraphs.

    Automatically splits into multiple messages if total length exceeds Telegram's limit.
    """
    if not items:
        return []

    local_now = (now or datetime.now(timezone.utc)).astimezone(TURKEY_TZ)
    period_label = (
        "Günlük Sabah Bülteni" if local_now.hour < 12 else "Günlük Özet Bülteni"
    )
    date_str = local_now.strftime("%d.%m.%Y %H:%M")

    header = (
        f"📊 <b>EKONOMİ &amp; BANKACILIK ÖZETİ ({period_label})</b>\n"
        f"🗓 <i>{date_str} (TSİ) — {len(items)} gelişme</i>\n"
        "━━━━━━━━━━━━━━━━━━━━"
    )

    blocks: list[str] = [header]

    for category in CATEGORY_ORDER:
        cat_items = [x for x in items if x.evaluation.category == category]
        if not cat_items:
            continue

        blocks.append(f"\n<b>{html.escape(category.display_title)}</b>")
        for item in cat_items:
            ev = item.evaluation
            headline = html.escape(ev.turkish_headline or item.raw.title)
            source = html.escape(item.raw.source_name)
            safe_url = html.escape(item.raw.url, quote=True)
            paragraph = html.escape(_normalize_plain_paragraph(ev.summary_text))

            entry_lines = [f"🔹 <b>{headline}</b>"]
            if paragraph:
                entry_lines.append(paragraph)
            if ev.impact_note.strip():
                clean_impact = html.escape(_normalize_plain_paragraph(ev.impact_note))
                entry_lines.append(
                    f'<i>Etki: {clean_impact}</i> — <a href="{safe_url}">{source}</a>'
                )
            else:
                entry_lines.append(f'<a href="{safe_url}">{source}</a>')
            blocks.append("\n".join(entry_lines))

    return _pack_blocks_into_messages(blocks, max_len=TELEGRAM_MAX_MESSAGE_LENGTH)


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
    """Delivers formatted HTML alerts and digests via Telegram Bot API."""

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
        send_empty: bool = False,
    ) -> bool:
        """Send categorized digest messages. Returns True if delivered (or nothing to send)."""
        if not items:
            if send_empty:
                return self._send_html(
                    "📊 <b>Ekonomi &amp; Bankacılık Bülteni</b>\n"
                    "Son dönemde eşik değerini aşan yeni bir önemli gelişme bulunmadı."
                )
            logger.info("Bülten kuyruğunda gönderilecek yeni haber yok; sessiz kalındı.")
            return False

        messages = format_digest_messages(items)
        for index, msg in enumerate(messages, start=1):
            if len(messages) > 1:
                msg = f"{msg}\n\n<i>({index}/{len(messages)})</i>"
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

