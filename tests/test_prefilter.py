"""Unit tests for rule-based clickbait pre-filtering and URL canonicalization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from news_bot.models import RawArticle
from news_bot.prefilter import (
    canonicalize_url,
    clean_html_text,
    compute_article_id,
    is_noise_headline,
    prefilter_articles,
)


def _make_article(title: str, hours_ago: int = 1) -> RawArticle:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    url = f"https://example.com/haber/{abs(hash(title))}?utm_source=rss&fbclid=123"
    canonical = canonicalize_url(url)
    return RawArticle(
        article_id=compute_article_id(canonical),
        source_id="test",
        source_name="Test Kaynak",
        authority_tier=2,
        title=title,
        url=url,
        canonical_url=canonical,
        summary="<p>Örnek <b>haber</b> özeti.</p>",
        published_at=now - timedelta(hours=hours_ago),
        fetched_at=now,
    )


@pytest.mark.parametrize(
    "noise_title",
    [
        "Gram altın ne kadar oldu? 11 Kasım çeyrek altın fiyatları!",
        "Dolar kaç TL oldu? Güncel kur tablosu",
        "Emekli promosyonları güncel: Hangi banka kaç TL promosyon veriyor?",
        "Asgari ücret zammı belli oldu mu?",
        "Çılgın Sayısal Loto sonuçları açıklandı",
        "11 Kasım günlük burç yorumları ve astroloji",
        "3 Unstoppable Stocks to Buy Right Now — Motley Fool",
        "Dogecoin price prediction: memecoin rally incoming?",
        "A101 KATALOĞUNDA KASIM AYININ EN BÜYÜK İNDİRİMİ YOLDA!",
        "Gram altın 9.500 TL'yi görür mü? Uzman isim yıl sonu için rakam verdi",
        "İstanbul'a fırtına uyarısı: Rüzgarın hızı 90 kilometreye ulaşacak",
        "X Bankası gişe yetkilisi personel alımı iş ilanı yayımlandı",
    ],
)
def test_noise_headlines_are_eliminated(noise_title: str) -> None:
    assert is_noise_headline(noise_title) is True


@pytest.mark.parametrize(
    "valid_title",
    [
        "TCMB Para Politikası Kurulu politika faizini yüzde 45'te sabit tuttu",
        "BDDK ticari kredi büyüme sınırlarında değişikliğe gitti",
        "TÜİK Ekim ayı enflasyon verilerini açıkladı",
        "İhracat rakamları Ekim ayında yıllık bazda yüzde 5,2 arttı",
        "Federal Reserve holds benchmark interest rate steady",
        "Garanti BBVA 500 milyon dolarlık sendikasyon kredisi sağladı",
    ],
)
def test_legitimate_economy_and_banking_headlines_pass(valid_title: str) -> None:
    assert is_noise_headline(valid_title) is False


def test_canonicalize_url_strips_tracking_params() -> None:
    dirty = "HTTPS://www.BloombergHT.com/ekonomi/haber-123/?utm_source=twitter&id=42&fbclid=abc#section"
    assert canonicalize_url(dirty) == "https://www.bloomberght.com/ekonomi/haber-123?id=42"


def test_clean_html_text_strips_tags_and_truncates() -> None:
    raw = "<div>Merhaba   <b>TCMB</b> &amp; BDDK!</div>"
    assert clean_html_text(raw) == "Merhaba TCMB & BDDK!"
    assert len(clean_html_text("A" * 100, max_chars=20)) == 20


def test_prefilter_drops_stale_and_clickbait_articles() -> None:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    fresh_valid = _make_article("BDDK haftalık bankacılık sektörü verilerini yayımladı", hours_ago=2)
    stale_valid = _make_article("ECB faiz kararını duyurdu", hours_ago=40)
    fresh_noise = _make_article("Çeyrek altın kaç TL oldu? İşte fiyatlar", hours_ago=1)

    kept, dropped = prefilter_articles(
        [fresh_valid, stale_valid, fresh_noise],
        max_age_hours=24,
        now=now,
    )
    assert len(kept) == 1
    assert kept[0].title == fresh_valid.title
    assert dropped == 2
