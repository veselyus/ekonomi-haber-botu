"""Rule-based pre-filtering, URL normalization, and clickbait elimination."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from news_bot.models import RawArticle

TRACKING_QUERY_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "yclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "rss",
        "source",
    }
)

# Turkish & English clickbait / SEO filler patterns compiled for zero-cost pre-filtering.
NOISE_HEADLINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        # Günlük SEO fiyat sorgu spam'leri (altın, döviz, akaryakıt tablosu)
        r"\b(gram|çeyrek|yarım|tam|cumhuriyet|ata|ons)\s+altın(ın)?\s+(fiyatı|fiyatları|ne\s+kadar|kaç\s+tl)",
        r"\baltın\s+fiyatları\s+(ne\s+kadar|kaç\s+tl|canlı|ne\s+olacak)",
        r"\b(gram|çeyrek|ons)\s+(altın|gümüş).*\bgörür\s+mü\b",
        r"\buzman\s+isim.*rakam\s+verdi\b",
        r"\b(dolar|euro|sterlin)\s+(kaç\s+tl|ne\s+kadar)\s*(oldu)?\b",
        r"\b(kaç\s+tl|ne\s+kadar)\s+oldu\?",
        r"\b(benzin|motorin|lpg|akaryakıt)\s+fiyatları.*(kaç\s+tl|ne\s+kadar)",
        # Market aktüel ürün katalog spam'leri
        r"\b(a101|bim|şok|migros|tarım\s+kredi)\s+(aktüel|kataloğu|kataloğunda|indirimli\s+ürünler)\b",
        # Soru kalıplı clickbait, promosyon ve rutin iş ilanı spam'leri
        r"\b(belli\s+oldu\s+mu|açıklandı\s+mı|yatacak\s+mı|yattı\s+mı|ne\s+zaman\s+yatacak)\b",
        r"\bhangi\s+banka\s+(ne\s+kadar|kaç\s+tl)\s+promosyon\b",
        r"\bemekli\s+promosyon(u|ları)\s+(ne\s+kadar|kaç\s+tl|güncel)\b",
        r"\b(personel\s+alımı|iş\s+ilanı|iş\s+ilanları|gişe\s+yetkilisi\s+alımı|mülakat\s+soruları)\b",
        r"\bkuruşu\s+kuruşuna\b",
        # Ekonomi dışı sızıntılar (şans oyunları, spor, magazin, astroloji, hava durumu)
        r"\b(süper\s+loto|çılgın\s+sayısal(\s+loto)?|sayısal\s+loto|şans\s+topu|on\s+numara|milli\s+piyango)\b",
        r"\b(burç\s+yorumları|günlük\s+burç|astroloji|fal)\b",
        r"\b(hava\s+durumu|fırtına\s+uyarısı|sağanak\s+uyarısı|kar\s+yağışı\s+uyarısı|rüzgarın\s+hızı)\b",
        r"\b(süper\s+lig|puan\s+durumu|maç\s+özeti|ilk\s+11'ler|bonservis)\b",
        # İngilizce bireysel hisse pazarlama / magazin / kripto spam'leri
        r"\b\d+\s+(unstoppable\s+|top\s+|best\s+)?stocks?\s+to\s+buy\b",
        r"\bmillionaire[-\s]maker\b",
        r"\bmotley\s+fool\b",
        r"\b(powerball|mega\s+millions|lottery\s+numbers|horoscope)\b",
        r"\b(memecoin|shiba\s+inu|dogecoin\s+price\s+prediction)\b",
    )
)

WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_TR_RE = re.compile(r"[^\w\s.çğıöşü]", flags=re.UNICODE)
STRAY_DOT_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
APOSTROPHE_SUFFIX_RE = re.compile(r"['’ʼ`][a-zçğıöşü]+")
DECIMAL_COMMA_RE = re.compile(r"(\d),(\d{1,2})\b")
LEADING_PCT_RE = re.compile(r"%\s*(\d)")
TRAILING_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

CANONICAL_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("türkiye cumhuriyet merkez bankası", "tcmb"),
    ("bankacılık düzenleme ve denetleme kurumu", "bddk"),
    ("türkiye bankalar birliği", "tbb"),
    ("türkiye istatistik kurumu", "tüik"),
    ("sermaye piyasası kurulu", "spk"),
    ("hazine ve maliye bakanlığı", "hazine"),
    ("abd merkez bankası", "fed"),
    ("federal reserve", "fed"),
    ("avrupa merkez bankası", "ecb"),
    ("european central bank", "ecb"),
    ("uluslararası para fonu", "imf"),
    ("garanti bbva", "garanti"),
    ("iş bankası", "işbankası"),
    ("yapı kredi", "yapıkredi"),
    ("kuveyt türk", "kuveyttürk"),
    ("qnb finansbank", "qnb"),
    ("qnb türkiye", "qnb"),
)


def normalize_turkish_lower(text: str) -> str:
    """Lowercase text safely respecting Turkish dotted/dotless I characters."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def normalize_title_for_matching(title: str) -> str:
    """Produce normalized canonical text for cross-source event deduplication."""
    lowered = normalize_turkish_lower(title)
    lowered = APOSTROPHE_SUFFIX_RE.sub("", lowered)
    for phrase, canonical in CANONICAL_SYNONYMS:
        if phrase in lowered:
            lowered = lowered.replace(phrase, canonical)
    lowered = DECIMAL_COMMA_RE.sub(r"\1.\2", lowered)
    lowered = LEADING_PCT_RE.sub(r"yüzde \1", lowered)
    lowered = TRAILING_PCT_RE.sub(r"yüzde \1", lowered)
    cleaned = NON_WORD_TR_RE.sub(" ", lowered)
    cleaned = STRAY_DOT_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def clean_html_text(raw_html: str, max_chars: int = 600) -> str:
    """Strip HTML tags/entities and truncate summary text cleanly."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = WHITESPACE_RE.sub(" ", soup.get_text(separator=" ")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def canonicalize_url(raw_url: str) -> str:
    """Normalize URL and strip analytics/tracking query parameters."""
    stripped = raw_url.strip()
    if not stripped:
        return ""
    parts = urlsplit(stripped)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in TRACKING_QUERY_PARAMS and not k.lower().startswith("utm_")
    ]
    filtered_query.sort()
    query_str = urlencode(filtered_query)
    return urlunsplit((scheme, netloc, path, query_str, ""))


def compute_article_id(canonical_url: str) -> str:
    """Compute short deterministic hex identifier from canonical URL."""
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]


def is_noise_headline(title: str) -> bool:
    """Return True if the headline matches known clickbait or off-topic patterns."""
    normalized = normalize_turkish_lower(title.strip())
    if len(normalized) < 12:
        return True
    return any(pattern.search(normalized) for pattern in NOISE_HEADLINE_PATTERNS)


def is_article_too_old(
    article: RawArticle,
    *,
    max_age_hours: int,
    now: datetime | None = None,
) -> bool:
    """Return True if article has a timestamp older than max_age_hours."""
    if article.published_at is None:
        return False
    reference_now = now or datetime.now(timezone.utc)
    published_utc = (
        article.published_at
        if article.published_at.tzinfo is not None
        else article.published_at.replace(tzinfo=timezone.utc)
    )
    cutoff = reference_now - timedelta(hours=max_age_hours)
    return published_utc < cutoff


def prefilter_articles(
    articles: list[RawArticle],
    *,
    max_age_hours: int,
    now: datetime | None = None,
) -> tuple[list[RawArticle], int]:
    """Filter out old or clickbait articles. Returns (kept_articles, dropped_count)."""
    kept: list[RawArticle] = []
    dropped = 0
    for article in articles:
        if not article.canonical_url or is_article_too_old(
            article, max_age_hours=max_age_hours, now=now
        ):
            dropped += 1
            continue
        if is_noise_headline(article.title):
            dropped += 1
            continue
        kept.append(article)
    return kept, dropped
