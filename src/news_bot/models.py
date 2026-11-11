"""Data models and schemas for the Economy & Banking News Bot."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NewsCategory(str, Enum):
    """Supported high-signal news domains plus irrelevant bucket."""

    TR_ECONOMY = "TR_ECONOMY"
    TR_BANKING = "TR_BANKING"
    WORLD_ECONOMY = "WORLD_ECONOMY"
    IRRELEVANT = "IRRELEVANT"

    @property
    def display_title(self) -> str:
        """Return localized Turkish section header with icon."""
        titles: dict[NewsCategory, str] = {
            NewsCategory.TR_ECONOMY: "🇹🇷 Türkiye Ekonomisi",
            NewsCategory.TR_BANKING: "🏦 Türkiye Bankacılık Sektörü",
            NewsCategory.WORLD_ECONOMY: "🌍 Dünya Ekonomisi",
            NewsCategory.IRRELEVANT: "🚫 Kapsam Dışı",
        }
        return titles[self]


class SourceType(str, Enum):
    """Fetching strategy for a configured news source."""

    RSS = "rss"
    BDDK_ANNOUNCEMENTS = "bddk_announcements"
    TCMB_ANNOUNCEMENTS = "tcmb_announcements"


class NewsSource(BaseModel):
    """Metadata and configuration for a curated news source."""

    id: str
    name: str
    url: str
    source_type: SourceType = SourceType.RSS
    default_category: NewsCategory
    authority_tier: int = Field(
        default=2,
        ge=1,
        le=3,
        description="1: Resmi kurum/düzenleyici, 2: Birincil ekonomi basını, 3: Genel kaynak",
    )
    language: str = "tr"
    enabled: bool = True


class RawArticle(BaseModel):
    """Normalized article fetched from RSS or official web page before AI triage."""

    article_id: str
    source_id: str
    source_name: str
    authority_tier: int = Field(default=2, ge=1, le=3)
    language: str = "tr"
    title: str
    url: str
    canonical_url: str
    summary: str = ""
    published_at: datetime | None = None
    fetched_at: datetime


class ArticleEvaluation(BaseModel):
    """Structured editorial assessment produced by Gemini API."""

    article_id: str = Field(
        description="Değerlendirilen haberin benzersiz kimliği (girdideki article_id ile birebir aynı olmalı)."
    )
    is_relevant: bool = Field(
        description="Haber doğrudan Türkiye ekonomisi, dünya ekonomisi veya Türkiye bankacılık sektörü ile ilgili somut bir gelişme mi?"
    )
    category: NewsCategory = Field(
        description="Haberin ana kategorisi. İlgisiz, tık tuzağı veya spekülatif ise IRRELEVANT seçilmeli."
    )
    importance_score: int = Field(
        ge=1,
        le=10,
        description=(
            "1-10 arası önem puanı. "
            "9-10: Kritik resmi karar, faiz/enflasyon verisi, BDDK regülasyonu veya küresel şok (anlık bildirim). "
            "7-8: Önemli makroekonomik veya bankacılık sektörü gelişmesi (günlük bülten). "
            "1-6: Rutin piyasa yorumu, düşük etkili haber veya tekrar."
        ),
    )
    duplicate_of_id: str | None = Field(
        default=None,
        description=(
            "Aynı listedeki başka bir haberle aynı olayı anlatıyorsa, daha yetkili/kapsamlı olan haberin "
            "article_id değeri; benzersiz ise null."
        ),
    )
    turkish_headline: str = Field(
        description="ASD-STE100 kurallarına uygun, etken çatılı, net ve sade Türkçe başlık."
    )
    summary_text: str = Field(
        default="",
        description="ASD-STE100 kurallarına göre yazılmış, madde işareti içermeyen, 2-3 kısa ve sade cümlelik düz paragraf özeti.",
    )
    impact_note: str = Field(
        default="",
        description="ASD-STE100 kurallarına uygun, en fazla 12 kelimelik etken çatılı tek cümlelik etki notu.",
    )

    @field_validator("duplicate_of_id", mode="before")
    @classmethod
    def _empty_str_to_none(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        return text


class BatchEvaluationResponse(BaseModel):
    """Top-level wrapper for Gemini structured batch evaluation."""

    evaluations: list[ArticleEvaluation]


class EnrichedArticle(BaseModel):
    """Combined article record ready for storage and Telegram delivery."""

    raw: RawArticle
    evaluation: ArticleEvaluation
    instant_sent: bool = False
    queued_for_digest: bool = False
    digest_sent: bool = False


class MarketQuote(BaseModel):
    """Live snapshot of a financial instrument with its 24-hour percentage change."""

    symbol_id: str
    icon: str
    label: str
    price: float
    change_pct_24h: float
    unit_suffix: str = ""
    decimals: int = Field(default=2, ge=0, le=6)
