"""Curated high-signal news sources for Turkish Economy, Banking, and Global Economy."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from news_bot.models import NewsCategory, NewsSource, SourceType

logger = logging.getLogger(__name__)

DEFAULT_SOURCES: tuple[NewsSource, ...] = (
    # --- Tier 1: Resmi Kurumlar ve Merkez Bankaları ---
    NewsSource(
        id="tcmb_duyurular",
        name="TCMB Basın Duyuruları",
        url="https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular+ve+Konusmalar/Basin+Duyurulari/",
        source_type=SourceType.TCMB_ANNOUNCEMENTS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=1,
        language="tr",
    ),
    NewsSource(
        id="bddk_duyurular",
        name="BDDK Resmi Duyurular",
        url="https://www.bddk.org.tr/",
        source_type=SourceType.BDDK_ANNOUNCEMENTS,
        default_category=NewsCategory.TR_BANKING,
        authority_tier=1,
        language="tr",
    ),
    NewsSource(
        id="fed_monetary",
        name="Federal Reserve (Fed)",
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
        source_type=SourceType.RSS,
        default_category=NewsCategory.WORLD_ECONOMY,
        authority_tier=1,
        language="en",
    ),
    NewsSource(
        id="ecb_press",
        name="Avrupa Merkez Bankası (ECB)",
        url="https://www.ecb.europa.eu/rss/press.html",
        source_type=SourceType.RSS,
        default_category=NewsCategory.WORLD_ECONOMY,
        authority_tier=1,
        language="en",
    ),
    # --- Tier 2: Türkiye Bankacılık & Ekonomi Odaklı Akışlar ---
    NewsSource(
        id="bankadunyasi",
        name="Banka Dünyası",
        url="https://bankadunyasi.com/feed/",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_BANKING,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="tr_banking_focus",
        name="Türkiye Bankacılık & BDDK Akışı",
        url=(
            "https://news.google.com/rss/search?"
            "q=(BDDK+OR+TBB+OR+%22bankac%C4%B1l%C4%B1k+sekt%C3%B6r%C3%BC%22+OR+"
            "%22zorunlu+kar%C5%9F%C4%B1l%C4%B1k%22+OR+%22sendikasyon+kredisi%22+OR+"
            "%22mevduat+faizi%22+OR+%22kredi+b%C3%BCy%C3%BCme%22)+when:1d"
            "&hl=tr&gl=TR&ceid=TR:tr"
        ),
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_BANKING,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="bloomberght",
        name="Bloomberg HT",
        url="https://www.bloomberght.com/rss/tum-haberler.xml",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="ekonomim",
        name="Ekonomim",
        url="https://www.ekonomim.com/rss",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="dunya_gazetesi",
        name="Dünya Gazetesi",
        url="https://www.dunya.com/rss",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="aa_ekonomi",
        name="Anadolu Ajansı Ekonomi",
        url="https://www.aa.com.tr/tr/rss/default?cat=ekonomi",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=2,
        language="tr",
    ),
    NewsSource(
        id="trthaber_ekonomi",
        name="TRT Haber Ekonomi",
        url="https://www.trthaber.com/ekonomi_articles.rss",
        source_type=SourceType.RSS,
        default_category=NewsCategory.TR_ECONOMY,
        authority_tier=2,
        language="tr",
    ),
    # --- Tier 2: Dünya Ekonomisi ve Küresel Finans Basını ---
    NewsSource(
        id="cnbc_economy",
        name="CNBC Economy",
        url="https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        source_type=SourceType.RSS,
        default_category=NewsCategory.WORLD_ECONOMY,
        authority_tier=2,
        language="en",
    ),
    NewsSource(
        id="ft_global_economy",
        name="Financial Times Global Economy",
        url="https://www.ft.com/global-economy?format=rss",
        source_type=SourceType.RSS,
        default_category=NewsCategory.WORLD_ECONOMY,
        authority_tier=2,
        language="en",
    ),
    NewsSource(
        id="bbc_business",
        name="BBC Business",
        url="https://feeds.bbci.co.uk/news/business/rss.xml",
        source_type=SourceType.RSS,
        default_category=NewsCategory.WORLD_ECONOMY,
        authority_tier=2,
        language="en",
    ),
)


def load_active_sources(custom_file: Path | None = None) -> list[NewsSource]:
    """Load default curated sources and merge any user-defined custom sources JSON."""
    sources_by_id: dict[str, NewsSource] = {
        src.id: src for src in DEFAULT_SOURCES
    }
    target_path = custom_file or Path("config/custom_sources.json")
    if target_path.exists():
        try:
            raw_data = json.loads(target_path.read_text(encoding="utf-8"))
            if isinstance(raw_data, list):
                for item in raw_data:
                    custom_src = NewsSource.model_validate(item)
                    sources_by_id[custom_src.id] = custom_src
                logger.info(
                    "%s dosyasından %d özel kaynak yüklendi.",
                    target_path,
                    len(raw_data),
                )
        except (OSError, ValueError) as exc:
            logger.warning(
                "Özel kaynak dosyası okunamadı (%s): %s", target_path, exc
            )

    return [src for src in sources_by_id.values() if src.enabled]
