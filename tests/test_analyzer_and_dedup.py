"""Unit tests for authority-aware deduplication and Gemini structured triage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from news_bot.analyzer import GeminiNewsAnalyzer, enforce_category_domain_rules
from news_bot.deduplicator import (
    compute_headline_similarity,
    deduplicate_candidates,
    deduplicate_enriched_articles,
    suppress_post_eval_duplicates,
)
from news_bot.fetcher import _strip_aggregator_publisher_suffix
from news_bot.models import (
    ArticleEvaluation,
    EnrichedArticle,
    NewsCategory,
    NewsSource,
    RawArticle,
)
from news_bot.prefilter import (
    canonicalize_url,
    compute_article_id,
    normalize_title_for_matching,
)
from news_bot.storage import ArticleStorage


def _build_raw(
    url: str,
    title: str,
    *,
    source_name: str = "Kaynak",
    tier: int = 2,
    language: str = "tr",
) -> RawArticle:
    canonical = canonicalize_url(url)
    return RawArticle(
        article_id=compute_article_id(canonical),
        source_id=source_name.lower().replace(" ", "_"),
        source_name=source_name,
        authority_tier=tier,
        language=language,
        title=title,
        url=url,
        canonical_url=canonical,
        summary=title,
        published_at=datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 11, 11, 10, 5, tzinfo=timezone.utc),
    )


def test_deduplicator_prefers_tier1_authority_and_keeps_distinct_reports(
    tmp_path: Path,
) -> None:
    storage = ArticleStorage(tmp_path / "state.db")

    media_copy = _build_raw(
        "https://portal.example.com/tcmb-faiz-1",
        "Merkez Bankası (TCMB) politika faizini yüzde 45 seviyesinde sabit bıraktı",
        source_name="Ekonomi Portalı",
        tier=2,
    )
    official_tcmb = _build_raw(
        "https://tcmb.gov.tr/duyuru-faiz-1",
        "TCMB politika faizini yüzde 45 seviyesinde sabit tuttu",
        source_name="TCMB Basın Duyuruları",
        tier=1,
    )
    distinct_tcmb_report = _build_raw(
        "https://tcmb.gov.tr/duyuru-odeme-dengesi",
        "TCMB Eylül ayı ödemeler dengesi istatistiklerini açıkladı",
        source_name="TCMB Basın Duyuruları",
        tier=1,
    )

    accepted, dropped = deduplicate_candidates(
        [media_copy, official_tcmb, distinct_tcmb_report],
        storage,
        fuzzy_threshold=80.0,
    )

    assert dropped == 1
    assert len(accepted) == 2
    accepted_sources = [a.source_name for a in accepted]
    assert "TCMB Basın Duyuruları" in accepted_sources
    assert "Ekonomi Portalı" not in accepted_sources


def test_gemini_analyzer_parses_structured_output_and_suppresses_cross_lang_duplicate() -> None:
    fed_official = _build_raw(
        "https://federalreserve.gov/newsevents/pressreleases/monetary20261111a.htm",
        "Federal Reserve issues FOMC statement keeping rates unchanged",
        source_name="Federal Reserve (Fed)",
        tier=1,
        language="en",
    )
    tr_copy = _build_raw(
        "https://bloomberght.com/fed-faiz-kararini-acikladi-1",
        "Fed faiz kararını açıkladı: Faizler sabit bırakıldı",
        source_name="Bloomberg HT",
        tier=2,
        language="tr",
    )

    mock_payload = {
        "evaluations": [
            {
                "article_id": fed_official.article_id,
                "is_relevant": True,
                "category": "WORLD_ECONOMY",
                "importance_score": 10,
                "duplicate_of_id": "",
                "turkish_headline": "Fed politika faizini yüzde 4,50-4,75 aralığında sabit tuttu",
            },
            {
                "article_id": tr_copy.article_id,
                "is_relevant": True,
                "category": "WORLD_ECONOMY",
                "importance_score": 9,
                "duplicate_of_id": fed_official.article_id,
                "turkish_headline": "Fed politika faizini değiştirmedi",
            },
        ]
    }

    captured_kwargs: dict[str, object] = {}

    class DummyInteractions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(output_text=json.dumps(mock_payload, ensure_ascii=False))

    dummy_client = SimpleNamespace(interactions=DummyInteractions())
    analyzer = GeminiNewsAnalyzer(
        api_key="fake-key",
        model="gemini-3.8-flash",
        batch_size=15,
        client=dummy_client,  # type: ignore[arg-type]
    )

    results = analyzer.evaluate_articles([fed_official, tr_copy])

    assert captured_kwargs["model"] == "gemini-3.8-flash"
    assert captured_kwargs["store"] is False
    assert results[fed_official.article_id].is_relevant is True
    assert results[fed_official.article_id].category == NewsCategory.WORLD_ECONOMY
    assert results[fed_official.article_id].importance_score == 10
    assert results[fed_official.article_id].duplicate_of_id is None

    # Cross-language duplicate automatically suppressed so user isn't pinged twice!
    assert results[tr_copy.article_id].duplicate_of_id == fed_official.article_id
    assert results[tr_copy.article_id].is_relevant is False


def test_paraphrased_same_event_across_two_sources_is_deduplicated_while_different_banks_stay_separate(
    tmp_path: Path,
) -> None:
    storage = ArticleStorage(tmp_path / "cross_source.db")

    # Same inflation story written differently by two outlets
    aa_inflation = _build_raw(
        "https://aa.com.tr/tr/ekonomi/enflasyon-ekim",
        "TÜİK ekim ayı enflasyon rakamlarını açıkladı: Yıllık yüzde 48,58 oldu",
        source_name="Anadolu Ajansı Ekonomi",
        tier=2,
    )
    bht_inflation = _build_raw(
        "https://bloomberght.com/ekim-enflasyonu-aciklandi",
        "Ekim ayında tüketici enflasyonu aylık %2,88, yıllık %48,58 seviyesinde gerçekleşti",
        source_name="Bloomberg HT",
        tier=2,
    )
    # Different TÜİK release with different figure must NOT be merged
    tuik_industrial = _build_raw(
        "https://ekonomim.com/sanayi-uretimi-eylul",
        "TÜİK eylül ayı sanayi üretimi endeksini yıllık yüzde 2,4 düşüşle açıkladı",
        source_name="Ekonomim",
        tier=2,
    )
    # Two DIFFERENT banks taking identical-sized syndication loans must BOTH pass!
    akbank_loan = _build_raw(
        "https://bankadunyasi.com/akbank-sendikasyon",
        "Akbank uluslararası piyasalardan 500 milyon dolar sendikasyon kredisi sağladı",
        source_name="Banka Dünyası",
        tier=2,
    )
    garanti_loan = _build_raw(
        "https://dunya.com/garanti-sendikasyon",
        "Garanti BBVA uluslararası piyasalardan 500 milyon dolar sendikasyon kredisi sağladı",
        source_name="Dünya Gazetesi",
        tier=2,
    )

    accepted, dropped = deduplicate_candidates(
        [aa_inflation, bht_inflation, tuik_industrial, akbank_loan, garanti_loan],
        storage,
        fuzzy_threshold=74.0,
    )
    assert dropped == 1
    assert len(accepted) == 4
    accepted_urls = {a.url for a in accepted}
    assert "https://ekonomim.com/sanayi-uretimi-eylul" in accepted_urls
    assert "https://bankadunyasi.com/akbank-sendikasyon" in accepted_urls
    assert "https://dunya.com/garanti-sendikasyon" in accepted_urls


def test_post_eval_and_digest_dedup_catch_missed_gemini_duplicates(
    tmp_path: Path,
) -> None:
    storage = ArticleStorage(tmp_path / "post_eval.db")

    src_one = _build_raw(
        "https://bddk.org.tr/duyuru-kredi-1",
        "Kurul Kararı: Ticari kredi büyüme sınırı hakkında basın açıklaması",
        source_name="BDDK Resmi Duyurular",
        tier=1,
    )
    src_two = _build_raw(
        "https://gazeteoksijen.com/ekonomi/bddkdan-ticari-kredilere-yeni-sinir",
        "Bankacılık Düzenleme ve Denetleme Kurumu'ndan ticari kredilere fren",
        source_name="Gazete Oksijen",
        tier=2,
    )

    # Suppose raw headlines looked too different before Gemini, and Gemini forgot duplicate_of_id,
    # producing standardized ASD-STE100 headlines for both:
    evals = {
        src_one.article_id: ArticleEvaluation(
            article_id=src_one.article_id,
            is_relevant=True,
            category=NewsCategory.TR_BANKING,
            importance_score=9,
            duplicate_of_id=None,
            turkish_headline="BDDK ticari kredi aylık büyüme sınırını yüzde 1,5 seviyesine indirdi",
        ),
        src_two.article_id: ArticleEvaluation(
            article_id=src_two.article_id,
            is_relevant=True,
            category=NewsCategory.TR_BANKING,
            importance_score=8,
            duplicate_of_id=None,
            turkish_headline="BDDK ticari kredilerde aylık büyüme oranını yüzde 1,5 seviyesine çekti",
        ),
    }

    cleaned_evals, suppressed = suppress_post_eval_duplicates(
        [src_two, src_one], evals, storage, fuzzy_threshold=74.0
    )
    assert suppressed == 1
    assert cleaned_evals[src_one.article_id].is_relevant is True
    assert cleaned_evals[src_two.article_id].is_relevant is False
    assert cleaned_evals[src_two.article_id].duplicate_of_id == src_one.article_id

    # Also verify digest-time gate cleans legacy duplicates already in queue
    legacy_queue = [
        EnrichedArticle(raw=src_two, evaluation=evals[src_two.article_id], queued_for_digest=True),
        EnrichedArticle(raw=src_one, evaluation=evals[src_one.article_id], queued_for_digest=True),
    ]
    unique_digest = deduplicate_enriched_articles(legacy_queue, fuzzy_threshold=74.0)
    assert len(unique_digest) == 1
    assert unique_digest[0].raw.source_name == "BDDK Resmi Duyurular"


def test_google_news_aggregator_publisher_suffix_is_stripped() -> None:
    agg_source = NewsSource(
        id="tr_banking_focus",
        name="Türkiye Bankacılık & BDDK Akışı",
        url="https://news.google.com/rss/search?q=BDDK",
        default_category=NewsCategory.TR_BANKING,
        authority_tier=3,
    )
    entry = {"source": {"title": "BloombergHT"}}
    cleaned_title, effective_source = _strip_aggregator_publisher_suffix(
        "BDDK ticari kredilerde büyüme sınırını düşürdü - BloombergHT",
        entry,  # type: ignore[arg-type]
        agg_source,
    )
    assert cleaned_title == "BDDK ticari kredilerde büyüme sınırını düşürdü"
    assert effective_source == "BloombergHT"

    # Full institutional synonym + apostrophe normalization check
    norm_a = normalize_title_for_matching("ABD Merkez Bankası'ndan %4,50 faiz kararı")
    norm_b = normalize_title_for_matching("Fed'den yüzde 4,50 faiz kararı")
    assert compute_headline_similarity(norm_a, norm_b) >= 90.0


def test_domain_guard_rejects_turkcell_from_banking_and_repairs_macro_to_tr_economy() -> None:
    # 1) Non-bank corporate earnings on 'Banka Dünyası' must NEVER pass as TR_BANKING
    turkcell_raw = _build_raw(
        "https://bankadunyasi.com/turkcell-konsolide-gelir",
        "Turkcell yılın üçüncü çeyreğinde konsolide gelirini yüzde 8 artırdı",
        source_name="Banka Dünyası",
        tier=2,
    )
    wrong_banking_ev = ArticleEvaluation(
        article_id=turkcell_raw.article_id,
        is_relevant=True,
        category=NewsCategory.TR_BANKING,
        importance_score=8,
        duplicate_of_id=None,
        turkish_headline="Turkcell üçüncü çeyrekte konsolide gelirini yüzde 8 artırdı",
    )
    fixed_turkcell = enforce_category_domain_rules(turkcell_raw, wrong_banking_ev)
    assert fixed_turkcell.is_relevant is False
    assert fixed_turkcell.category == NewsCategory.IRRELEVANT

    # 2) Genuine macro news accidentally tagged TR_BANKING moves cleanly to TR_ECONOMY
    macro_raw = _build_raw(
        "https://bankadunyasi.com/tuik-enflasyon",
        "TÜİK ekim ayı tüketici enflasyonu verilerini açıkladı",
        source_name="Banka Dünyası",
        tier=2,
    )
    mislabeled_macro = ArticleEvaluation(
        article_id=macro_raw.article_id,
        is_relevant=True,
        category=NewsCategory.TR_BANKING,
        importance_score=9,
        duplicate_of_id=None,
        turkish_headline="TÜİK ekim ayı yıllık enflasyonunu yüzde 48,58 olarak açıkladı",
    )
    fixed_macro = enforce_category_domain_rules(macro_raw, mislabeled_macro)
    assert fixed_macro.is_relevant is True
    assert fixed_macro.category == NewsCategory.TR_ECONOMY

    # 3) Real bank earnings (Akbank) remain valid TR_BANKING
    akbank_raw = _build_raw(
        "https://bankadunyasi.com/akbank-bilanco",
        "Akbank dokuz aylık dönemde 33 milyar TL konsolide net kâr açıkladı",
        source_name="Banka Dünyası",
        tier=2,
    )
    akbank_ev = ArticleEvaluation(
        article_id=akbank_raw.article_id,
        is_relevant=True,
        category=NewsCategory.TR_BANKING,
        importance_score=8,
        duplicate_of_id=None,
        turkish_headline="Akbank dokuz aylık dönemde 33 milyar TL net kâr elde etti",
    )
    fixed_akbank = enforce_category_domain_rules(akbank_raw, akbank_ev)
    assert fixed_akbank.is_relevant is True
    assert fixed_akbank.category == NewsCategory.TR_BANKING


def test_gemini_analyzer_instant_failover_and_bounded_sdk_retries() -> None:
    # 1) Verify real SDK client sets max_retries == 0 (attempts=1) so Speakeasy NEVER sleeps on 429 Retry-After
    real_analyzer = GeminiNewsAnalyzer(api_key="dummy-key")
    assert real_analyzer._client.interactions.sdk_configuration.retry_config.max_retries == 0

    # 2) Verify multi-pool cascade (3.5-flash-lite 503 -> 3.1-flash-lite 503 -> 3.8-flash 429 -> 4-argon OK)
    #    and verify Batch 2 skips the 429-exhausted model!
    batch_one = _build_raw(
        "https://bddk.org.tr/duyuru-99",
        "BDDK zorunlu karşılık ve kredi düzenlemesini yayımladı",
        source_name="BDDK Resmi Duyurular",
        tier=1,
    )
    batch_two = _build_raw(
        "https://tcmb.gov.tr/duyuru-100",
        "TCMB haftalık para ve banka istatistiklerini yayımladı",
        source_name="TCMB Resmi Duyurular",
        tier=1,
    )
    calls: list[tuple[str, object]] = []

    class CascadeInteractions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            model_name = str(kwargs["model"])
            gen_cfg = kwargs.get("generation_config")
            calls.append((model_name, gen_cfg))
            if model_name in {"gemini-3.5-flash-lite", "gemini-3.1-flash-lite"}:
                raise RuntimeError("503 Service Unavailable: high demand")
            if model_name == "gemini-3.8-flash":
                raise RuntimeError("429 Too Many Requests: quota exhausted")

            # Succeeds on gemini-4-argon!
            target_id = batch_one.article_id if len(calls) <= 4 else batch_two.article_id
            payload = {
                "evaluations": [
                    {
                        "article_id": target_id,
                        "is_relevant": True,
                        "category": "TR_BANKING" if target_id == batch_one.article_id else "TR_ECONOMY",
                        "importance_score": 8,
                        "duplicate_of_id": "",
                        "turkish_headline": "Kurum resmi düzenleme duyurusunu yayımladı",
                    }
                ]
            }
            return SimpleNamespace(output_text=json.dumps(payload, ensure_ascii=False))

    analyzer = GeminiNewsAnalyzer(
        api_key="fake",
        model="gemini-3.5-flash-lite",
        batch_size=1,
        cooldown_seconds=0.0,
        client=SimpleNamespace(interactions=CascadeInteractions()),  # type: ignore[arg-type]
    )
    res = analyzer.evaluate_articles([batch_one, batch_two])

    # Batch 1 tried: 3.5-flash-lite (minimal) -> 3.1-flash-lite (None) -> 3.8-flash (minimal, 429!) -> 4-argon (low, OK!)
    # Batch 2 skipped 3.8-flash because it hit 429 on Batch 1: 3.5-flash-lite -> 3.1-flash-lite -> 4-argon (OK!)
    assert [m for m, _ in calls] == [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-4-argon",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-4-argon",
    ]
    assert calls[0][1] == {"thinking_level": "minimal"}
    assert calls[1][1] is None
    assert calls[3][1] == {"thinking_level": "low"}
    assert res[batch_one.article_id].is_relevant is True
    assert res[batch_two.article_id].is_relevant is True


def test_emergency_heuristic_fallback_when_all_pools_hit_503_or_429() -> None:
    bddk_item = _build_raw(
        "https://bddk.org.tr/duyuru-201",
        "BDDK ticari kredi büyüme sınırında değişiklik yaptı",
        source_name="BDDK Resmi Duyurular",
        tier=1,
    )
    turkcell_noise = _build_raw(
        "https://bankadunyasi.com/turkcell-3c",
        "Turkcell üçüncü çeyrek konsolide gelirini açıkladı",
        source_name="Banka Dünyası",
        tier=2,
    )

    class AllOverloadedInteractions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    analyzer = GeminiNewsAnalyzer(
        api_key="fake",
        batch_size=15,
        cooldown_seconds=0.0,
        allow_heuristic_fallback=True,
        client=SimpleNamespace(interactions=AllOverloadedInteractions()),  # type: ignore[arg-type]
    )
    res = analyzer.evaluate_articles([bddk_item, turkcell_noise])

    assert res[bddk_item.article_id].is_relevant is True
    assert res[bddk_item.article_id].category == NewsCategory.TR_BANKING
    assert res[bddk_item.article_id].importance_score == 8
    assert res[turkcell_noise.article_id].is_relevant is False
    assert res[turkcell_noise.article_id].category == NewsCategory.IRRELEVANT


