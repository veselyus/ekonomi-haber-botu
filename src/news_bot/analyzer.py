"""Gemini API Chief Editor triage, classification, scoring, and Turkish summarization."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from news_bot.models import (
    ArticleEvaluation,
    BatchEvaluationResponse,
    NewsCategory,
    RawArticle,
)
from news_bot.prefilter import normalize_turkish_lower

logger = logging.getLogger(__name__)

BANKING_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "banka",
    "bddk",
    "tbb",
    "tmsf",
    "tkbb",
    "kredi",
    "mevduat",
    "sendikasyon",
    "zorunlu karşılık",
    "sermaye yeterlili",
    "takipteki alacak",
    "faktoring",
    "leasing",
    "finansal kiralama",
    "katılım finans",
    "kmh",
    "akbank",
    "garanti bbva",
    "iş bankası",
    "yapı kredi",
    "vakıfbank",
    "ziraat",
    "halkbank",
    "denizbank",
    "qnb",
    "türk ekonomi bankası",
    "şekerbank",
    "albaraka",
    "kuveyt türk",
    "tskb",
    "hsbc",
    "ing türkiye",
    "odeabank",
    "fibabanka",
    "alternatif bank",
    "anadolubank",
    "burgan",
    "enpara",
    "eximbank",
)

NON_BANK_COMPANIES: tuple[str, ...] = (
    "turkcell",
    "türk telekom",
    "vodafone",
    "türk hava yolları",
    "pegasus",
    "tüpraş",
    "petkim",
    "socar",
    "aselsan",
    "havelsan",
    "tusaş",
    "roketsan",
    "ereğli demir",
    "erdemir",
    "kardemir",
    "arçelik",
    "beko",
    "vestel",
    "ford otosan",
    "tofaş",
    "şişecam",
    "migros",
    "bim birleşik",
    "bimaş",
    "şok market",
    "ülker",
    "anadolu efes",
    "coca-cola içecek",
    "hektaş",
    "astor enerji",
    "enerjisa",
    "zorlu enerji",
    "koza altın",
    "emlak konut",
    "türk traktör",
    "otokar",
    "tav havalimanları",
)

CORPORATE_EARNINGS_NOISE_KEYWORDS: tuple[str, ...] = (
    "konsolide gelir",
    "üçüncü çeyrek",
    "ikinci çeyrek",
    "birinci çeyrek",
    "dördüncü çeyrek",
    "3. çeyrek",
    "2. çeyrek",
    "1. çeyrek",
    "4. çeyrek",
    "çeyrek bilanço",
    "çeyrek finansal",
    "net kâr",
    "net kar",
    "favök",
    "ebitda",
    "brüt kâr",
    "satış geliri",
    "cirosu",
    "cirosunu",
    "temettü",
    "hisse geri alım",
    "bedelsiz sermaye",
)

TR_MACRO_KEYWORDS: tuple[str, ...] = (
    "tüik",
    "tcmb",
    "merkez bankası",
    "enflasyon",
    "tüfe",
    "üfe",
    "gsyh",
    "cari açık",
    "cari denge",
    "ihracat",
    "ithalat",
    "dış ticaret",
    "işsizlik",
    "istihdam",
    "sanayi üretimi",
    "hazine",
    "maliye",
    "bütçe",
    "vergi",
    "ovp",
    "orta vadeli program",
    "asgari ücret",
    "rezerv",
    "moody",
    "fitch",
    "s&p",
)

WORLD_MACRO_KEYWORDS: tuple[str, ...] = (
    "fed",
    "federal reserve",
    "fomc",
    "ecb",
    "european central bank",
    "avrupa merkez bankası",
    "imf",
    "world bank",
    "dünya bankası",
    "oecd",
    "inflation",
    "cpi",
    "interest rate",
    "rate cut",
    "rate hike",
    "gdp",
    "recession",
    "tariff",
    "tarife",
    "treasury",
    "unemployment",
    "payroll",
    "central bank",
    "boj",
    "bank of england",
    "eurozone",
    "china",
    "brent",
    "opec",
)

DEFAULT_MODEL_CASCADE: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-4-argon",
    "gemma-4-31b-it",
)


def build_model_cascade(primary_model: str) -> tuple[str, ...]:
    """Order multi-pool failover cascade starting with primary_model without duplicates."""
    ordered: list[str] = []
    cleaned_primary = primary_model.strip() or DEFAULT_MODEL_CASCADE[0]
    for candidate in (cleaned_primary, *DEFAULT_MODEL_CASCADE):
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


def _generation_config_for_model(model_name: str) -> dict[str, Any] | None:
    """Return compatible thinking_level config per Gemini model generation."""
    normalized = model_name.strip().lower()
    if normalized.startswith("gemini-4-"):
        # Gemini 4 Argon/Carbon supports low/medium/high ('minimal' returns HTTP 400).
        return {"thinking_level": "low"}
    if normalized.startswith(
        ("gemini-3.5-", "gemini-3.6-", "gemini-3.7-", "gemini-3.8-")
    ):
        return {"thinking_level": "minimal"}
    return None


def _is_auth_or_config_error(exc: BaseException) -> bool:
    """Return True for non-retryable API key or permission errors."""
    err = str(exc).lower()
    return any(
        token in err
        for token in (
            "401",
            "403",
            "api_key_invalid",
            "api key not valid",
            "permission_denied",
            "unauthenticated",
        )
    )


def _is_quota_exhausted_error(exc: BaseException) -> bool:
    """Return True when a specific model's RPM/TPM/RPD quota returned 429."""
    err = str(exc).lower()
    return any(
        token in err
        for token in (
            "429",
            "resource_exhausted",
            "too many requests",
            "quota",
        )
    )


def _is_transient_capacity_error(exc: BaseException) -> bool:
    """Return True for temporary 503 high-demand spikes or network timeouts."""
    err = str(exc).lower()
    return any(
        token in err
        for token in (
            "503",
            "service_unavailable",
            "unavailable",
            "high demand",
            "overloaded",
            "500",
            "502",
            "504",
            "timed out",
            "timeout",
            "connection",
        )
    )


def enforce_category_domain_rules(
    raw: RawArticle,
    evaluation: ArticleEvaluation,
) -> ArticleEvaluation:
    """
    Deterministic post-LLM domain guard:
    1) Reject non-bank single-company corporate earnings/revenue (e.g., Turkcell consolidated revenue)
       from polluting TR_BANKING or TR_ECONOMY.
    2) Require genuine banking/regulator entities for TR_BANKING; reclassify mislabeled
       macro stories to TR_ECONOMY instead of letting source names like 'Banka Dünyası' mislead classification.
    """
    if not evaluation.is_relevant or evaluation.category == NewsCategory.IRRELEVANT:
        return evaluation

    text = normalize_turkish_lower(
        f"{raw.title} {raw.summary} {evaluation.turkish_headline}"
    )

    has_banking_term = any(kw in text for kw in BANKING_DOMAIN_KEYWORDS)
    has_non_bank_company = any(comp in text for comp in NON_BANK_COMPANIES)
    has_earnings_noise = any(kw in text for kw in CORPORATE_EARNINGS_NOISE_KEYWORDS)
    has_tr_macro = any(kw in text for kw in TR_MACRO_KEYWORDS)

    # 1) Non-bank corporate quarterly revenue/profit/KAP news is micro stock noise -> IRRELEVANT
    if has_non_bank_company and not has_banking_term:
        if evaluation.category == NewsCategory.TR_BANKING or has_earnings_noise:
            logger.info(
                "Banka dışı tekil şirket/bilanço haberi elendi (%s): %s",
                raw.source_name,
                evaluation.turkish_headline,
            )
            return evaluation.model_copy(
                update={
                    "is_relevant": False,
                    "category": NewsCategory.IRRELEVANT,
                    "importance_score": min(evaluation.importance_score, 3),
                }
            )

    # 2) Strict TR_BANKING verification (ignores source_name so 'Banka Dünyası' non-bank news cannot slip in)
    if evaluation.category == NewsCategory.TR_BANKING and not has_banking_term:
        if has_tr_macro and not has_earnings_noise:
            logger.info(
                "Bankacılık dışı makro haber TR_ECONOMY kategorisine taşındı: %s",
                evaluation.turkish_headline,
            )
            return evaluation.model_copy(update={"category": NewsCategory.TR_ECONOMY})

        logger.info(
            "Bankacılık terimi içermeyen sözde TR_BANKING haberi elendi (%s): %s",
            raw.source_name,
            evaluation.turkish_headline,
        )
        return evaluation.model_copy(
            update={
                "is_relevant": False,
                "category": NewsCategory.IRRELEVANT,
                "importance_score": min(evaluation.importance_score, 3),
            }
        )

    return evaluation


EDITOR_SYSTEM_PROMPT = """Sen üst düzey bir ekonomi ve bankacılık haber ajansının Baş Editörüsün.
Görevin; ham haber listesini inceleyip SIFIR GÜRÜLTÜ ilkesiyle yalnızca gerçekten önemli gelişmeleri seçmek,
sınıflandırmak, 1-10 arası önem puanı vermek ve net Türkçe manşet başlık çıkarmaktır.

KATEGORİ KURALLARI (ÇOK SIKI UYGULA):
1. TR_ECONOMY (Türkiye Makroekonomisi): TCMB para politikası/faiz/rezerv/enflasyon raporu, TÜİK enflasyon/GSYH büyüme/sanayi üretimi/işgücü/dış ticaret/güven endeksleri, Hazine ve Maliye Bakanlığı bütçe/borçlanma/vergi kararları, OVP, cari denge, ihracat/ithalat verileri, asgari ücret, kredi derecelendirme kuruluşlarının (Moody's, Fitch, S&P) Türkiye kararları ve Türkiye genelini etkileyen makroekonomik gelişmeler. (Dikkat: Haber kaynağı 'Banka Dünyası' olsa bile haber Türkiye makroekonomisi hakkındaysa TR_ECONOMY seç!)
2. TR_BANKING (YALNIZCA Türkiye Bankacılık Sektörü): SADECE lisanslı bankalar (mevduat, katılım, kalkınma ve yatırım bankaları), BDDK, TBB, TMSF, TKBB ve doğrudan bankacılık mevzuatı, zorunlu karşılıklar, kredi/mevduat faiz ve büyüme sınırları, kredi kartı/KMH düzenlemeleri, sermaye yeterliliği, takipteki krediler, Türk BANKALARININ sendikasyon kredileri, tahvil ihraçları, banka bilançoları ve banka üst düzey atamaları.
   - KESİN YASAK: Banka OLMAYAN şirketlerin (ör. Turkcell, Türk Telekom, THY, Tüpraş, Aselsan, BİM, Migros, Ereğli, otomotiv, telekom, havacılık, sanayi, perakende veya enerji şirketleri) konsolide gelir, ciro, net kâr, çeyrek bilanço, temettü veya KAP haberleri ASLA TR_BANKING OLAMAZ! Kaynak adı 'Banka Dünyası' olsa bile içeriği banka olmayan şirket bilançolarına kesinlikle IRRELEVANT ver!
3. WORLD_ECONOMY (Dünya Makroekonomisi): Fed, ECB, BoJ, BoE faiz ve bilanço kararları, ABD/Avrupa/Çin büyüme, istihdam ve enflasyon verileri, küresel ticaret tarifeleri/yaptırımlar, küresel enerji ve emtia arz şokları, IMF/Dünya Bankası/OECD raporları, küresel bankacılık ve finansal istikrar gelişmeleri.
4. IRRELEVANT (Kapsam Dışı ve Gürültü): Yukarıdaki 3 alana doğrudan girmeyen her şey! Özellikle:
   - Banka dışı tekil BIST/şirket çeyrek bilanço, konsolide gelir, ciro, FAVÖK ve hisse haberleri (ör. Turkcell gelir haberi -> KESİNLİKLE IRRELEVANT ve is_relevant=false!),
   - Magazin, spor, yerel asayiş, hava durumu, burç, bireysel hisse tüyoları, altcoin spekülasyonu,
   - Clickbait ("altın kaç TL oldu", "emekliye müjde"), rutin banka personel alım/iş ilanları, mülakat rehberleri, somut veri içermeyen köşe yazıları ve şirket reklam bültenleri KESİNLİKLE IRRELEVANT olmalı ve is_relevant=false işaretlenmelidir.

PUANLAMA CETVELİ (importance_score: 1-10):
- 10 (OLAĞANÜSTÜ KRİTİK - Anlık Alarm): Yalnızca çok büyük resmi kararlar ve şoklar! (TCMB, Fed veya ECB faiz kararı ilanı, BDDK'nın bankacılık/kredi kurallarını değiştiren kritik regülasyon kararı, TÜİK aylık enflasyon veya çeyreklik GSYH büyüme verisi açıklanması, Moody's/Fitch/S&P Türkiye veya Türk bankaları kredi notu değişikliği, olağanüstü finansal şoklar). Sıradan haberlere asla 10 verme!
- 7-9 (ÖNEMLİ - Sabah 08:00 Günlük Bülteni): Önemli makroekonomik veriler (cari denge, sanayi üretimi, ihracat, PMI, ABD istihdam/TÜFE), ekonomi yönetimi/TCMB/BDDK/TBB açıklamaları, haftalık bankacılık verileri, Türk bankalarının sendikasyon kredileri, tahvil ihraçları, bilançoları, üst düzey atamaları ve IMF/Dünya Bankası raporları.
- 1-6 (DÜŞÜK ÖNEM - Gönderilmeyecek): Rutin günlük piyasa açılış/kapanış tekrarı, somut karar içermeyen tahmin/yorumlar, ikincil mikro haberler.

MÜKERRER HABER KURALI (duplicate_of_id - ÇOK ÖNEMLİ):
- Farklı haber kaynakları (ör. Bloomberg HT, Ekonomim, Dünya Gazetesi, Gazete Oksijen, Banka Dünyası, AA, TRT, Fed, CNBC) AYNI OLAYI, KARARI veya VERİYİ farklı başlıklarla yazabilir.
- Aday listede aynı gerçek dünya olayını anlatan birden fazla haber varsa YALNIZCA EN YETKİLİ TEK BİR HABERE (authority_tier=1 resmi kurum > authority_tier=2 doğrudan basın > authority_tier=3 akış) is_relevant=true ve duplicate_of_id="" ver.
- Aynı olayı anlatan DİĞER TÜM kopyalarda kesinlikle is_relevant=false yap ve duplicate_of_id alanına seçtiğin ana haberin article_id değerini yaz! Asla aynı olayı iki farklı kaynaktan geçirme!
- Ayrıca sana verilen "DAHA ÖNCE YAYIMLANAN HABERLER" listesinde yer alan bir olayın aynısını veya devam tekrarını anlatan adaylar varsa duplicate_of_id alanına "PREVIOUS" yaz ve is_relevant=false yap.

MANŞET BAŞLIK KURALLARI (ASD-STE100 BASİTLEŞTİRİLMİŞ TEKNİK DİL STANDARDI):
Alt özet veya etki metni yazılmayacaktır; kullanıcı yalnızca senin yazacağın manşet başlığı (turkish_headline) görecektir. Bu nedenle turkish_headline alanını ASD-STE100 prensiplerine göre şu kurallarla yaz:
1. Başlık tek başına haberi AÇIKLAYICI olmalı; okuyan kişi linke tıklamadan da ne olduğunu ve varsa kritik rakamı/kararı anlamalıdır:
   - YANLIŞ (Belirsiz): "TCMB faiz kararını açıkladı"
   - DOĞRU (Açıklayıcı + ASD-STE100): "TCMB politika faizini yüzde 45 seviyesinde sabit tuttu"
   - YANLIŞ (Edilgen ve karmaşık): "BDDK tarafından ticari kredi büyüme sınırlarında değişikliğe gidildiği duyuruldu"
   - DOĞRU (Açıklayıcı + ASD-STE100): "BDDK ticari kredi aylık büyüme sınırını yüzde 1,5 seviyesine indirdi"
2. Daima ETKEN ÇATI kullan (Özne + Somut Veri/Nesne + Eylem).
3. Sade, kısa ve tek anlamlı kelimeler kullan (en fazla 14-18 kelime).
4. Mecaz, deyim, soru işareti, ünlem ve gazeteci klişesi KULLANMA ("soğuk duş", "kritik karar", "rekor kırdı", "dikkat çekti", "flas" YASAK).
5. Yabancı dildeki haberleri de aynı kurallarla doğrudan sade Türkçeye çevirerek yaz.
"""

# Flat, union-free JSON Schema strictly compatible with Gemini Interactions API structured output.
EVALUATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "description": "Girdideki haberin birebir article_id değeri.",
                    },
                    "is_relevant": {
                        "type": "boolean",
                        "description": "Somut Türkiye ekonomisi, dünya ekonomisi veya Türkiye bankacılık haberi mi?",
                    },
                    "category": {
                        "type": "string",
                        "enum": [cat.value for cat in NewsCategory],
                        "description": "Haberin kategorisi.",
                    },
                    "importance_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "1-10 arası önem puanı.",
                    },
                    "duplicate_of_id": {
                        "type": "string",
                        "description": "Aynı listedeki başka bir haberin tekrarıysa ana haberin article_id değeri, önceki bültenlerin tekrarıysa 'PREVIOUS', benzersizse boş metin ('').",
                    },
                    "turkish_headline": {
                        "type": "string",
                        "description": "ASD-STE100 kurallarına uygun, etken çatılı, sonucu ve rakamı doğrudan veren açıklayıcı Türkçe manşet başlık.",
                    },
                },
                "required": [
                    "article_id",
                    "is_relevant",
                    "category",
                    "importance_score",
                    "duplicate_of_id",
                    "turkish_headline",
                ],
            },
        }
    },
    "required": ["evaluations"],
}


def _extract_interaction_text(interaction: Any) -> str:
    """Safely extract output text from a Gemini Interaction response."""
    direct_text = getattr(interaction, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    parts: list[str] = []
    for step in getattr(interaction, "steps", []) or []:
        if getattr(step, "type", None) == "model_output":
            for content_item in getattr(step, "content", []) or []:
                text_val = getattr(content_item, "text", None)
                if isinstance(text_val, str):
                    parts.append(text_val)
    return "".join(parts).strip()


def _strip_json_fences(raw_text: str) -> str:
    """Remove optional markdown fences or surrounding text around JSON object."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    return cleaned


def _heuristic_fallback_evaluate(batch: list[RawArticle]) -> BatchEvaluationResponse:
    """
    Emergency local rule-based Chief Editor fallback used ONLY if every single
    Gemini/Gemma serving pool in the cascade is unreachable due to 503/429 outages.
    Caps scores at 8 so false 10/10 critical alerts never fire.
    """
    evaluations: list[ArticleEvaluation] = []
    for item in batch:
        text = normalize_turkish_lower(f"{item.title} {item.summary}")
        has_banking = any(kw in text for kw in BANKING_DOMAIN_KEYWORDS)
        has_tr_macro = any(kw in text for kw in TR_MACRO_KEYWORDS)
        has_world_macro = any(kw in text for kw in WORLD_MACRO_KEYWORDS)
        has_non_bank_corp = any(comp in text for comp in NON_BANK_COMPANIES)
        has_earnings_noise = any(kw in text for kw in CORPORATE_EARNINGS_NOISE_KEYWORDS)

        if has_banking and item.language == "tr":
            category = NewsCategory.TR_BANKING
        elif (
            has_tr_macro
            and item.language == "tr"
            and not (has_non_bank_corp and has_earnings_noise)
        ):
            category = NewsCategory.TR_ECONOMY
        elif (
            item.default_category == NewsCategory.WORLD_ECONOMY
            or item.language == "en"
        ) and has_world_macro:
            category = NewsCategory.WORLD_ECONOMY
        else:
            category = NewsCategory.IRRELEVANT

        is_relevant = category != NewsCategory.IRRELEVANT
        score = (8 if item.authority_tier == 1 else 7) if is_relevant else 2
        headline = " ".join(item.title.split()).strip() or "Ekonomi gelişmesi"

        candidate_eval = ArticleEvaluation(
            article_id=item.article_id,
            is_relevant=is_relevant,
            category=category,
            importance_score=score,
            duplicate_of_id=None,
            turkish_headline=headline,
            summary_text="",
            impact_note="",
        )
        evaluations.append(enforce_category_domain_rules(item, candidate_eval))

    return BatchEvaluationResponse(evaluations=evaluations)


class GeminiNewsAnalyzer:
    """Evaluates candidate articles in batches using multi-pool Gemini Interactions failover."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        batch_size: int = 15,
        client: genai.Client | None = None,
        cooldown_seconds: float = 6.0,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        self._model = model
        self._model_cascade: tuple[str, ...] = build_model_cascade(model)
        self._quota_exhausted_models: set[str] = set()
        self._batch_size = batch_size
        self._cooldown_seconds = cooldown_seconds
        self._allow_heuristic_fallback = allow_heuristic_fallback
        if client is not None:
            self._client = client
        else:
            # attempts=1 sets internal max_retries=0 so Speakeasy's retry handler NEVER
            # sleeps for minutes honoring Google's 429 Retry-After header or repeating 503s.
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=1,
                    )
                ),
            )

    def evaluate_articles(
        self,
        articles: list[RawArticle],
        *,
        recent_headlines: list[str] | None = None,
    ) -> dict[str, ArticleEvaluation]:
        """Evaluate all candidate articles in batches and return mapping by article_id."""
        results: dict[str, ArticleEvaluation] = {}
        known_headlines: list[str] = list(recent_headlines or [])
        total = len(articles)
        for start in range(0, total, self._batch_size):
            if start > 0 and self._cooldown_seconds > 0:
                time.sleep(min(1.5, self._cooldown_seconds))
            batch = articles[start : start + self._batch_size]
            batch_evals = self._evaluate_batch(batch, recent_headlines=known_headlines)
            results.update(batch_evals)
            for ev in batch_evals.values():
                if ev.is_relevant and ev.category != NewsCategory.IRRELEVANT and ev.turkish_headline:
                    known_headlines.append(ev.turkish_headline)
        return results

    def _call_model_once(self, prompt: str, model_name: str) -> BatchEvaluationResponse:
        if model_name.strip().lower().startswith("gemma-"):
            gemma_prompt = (
                f"{EDITOR_SYSTEM_PROMPT}\n\n"
                "ÇIKTI FORMATI KURALI: Yalnızca geçerli bir JSON nesnesi döndür; başka hiçbir açıklama yazma. "
                'JSON şeması: {"evaluations": [{"article_id": "...", "is_relevant": true, '
                '"category": "TR_ECONOMY|TR_BANKING|WORLD_ECONOMY|IRRELEVANT", '
                '"importance_score": 8, "duplicate_of_id": "", "turkish_headline": "..."}]}\n\n'
                f"{prompt}"
            )
            interaction = self._client.interactions.create(
                model=model_name,
                input=gemma_prompt,
                store=False,
                timeout=35.0,
            )
        else:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": prompt,
                "system_instruction": EDITOR_SYSTEM_PROMPT,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": EVALUATION_JSON_SCHEMA,
                },
                "store": False,
                "timeout": 30.0,
            }
            gen_cfg = _generation_config_for_model(model_name)
            if gen_cfg is not None:
                kwargs["generation_config"] = gen_cfg
            interaction = self._client.interactions.create(**kwargs)

        raw_text = _strip_json_fences(_extract_interaction_text(interaction))
        if not raw_text:
            raise RuntimeError(f"Model ({model_name}) boş yanıt döndürdü.")
        return BatchEvaluationResponse.model_validate_json(raw_text)

    def _evaluate_with_cascade(
        self,
        prompt: str,
        batch: list[RawArticle],
    ) -> BatchEvaluationResponse:
        active_models = [
            m for m in self._model_cascade if m not in self._quota_exhausted_models
        ]
        if not active_models:
            active_models = list(self._model_cascade)

        transient_models: list[str] = []
        last_exc: Exception | None = None
        all_capacity_or_quota = True

        for model_name in active_models:
            try:
                return self._call_model_once(prompt, model_name)
            except Exception as exc:
                if _is_auth_or_config_error(exc):
                    raise
                last_exc = exc
                if _is_quota_exhausted_error(exc):
                    self._quota_exhausted_models.add(model_name)
                    logger.warning(
                        "Model (%s) kota sınırına (429) ulaştı; beklemeden sıradaki bağımsız model havuzuna geçiliyor...",
                        model_name,
                    )
                elif _is_transient_capacity_error(exc):
                    transient_models.append(model_name)
                    logger.warning(
                        "Model (%s) anlık yoğunlukta (503/zaman aşımı); beklemeden sıradaki modele geçiliyor...",
                        model_name,
                    )
                else:
                    all_capacity_or_quota = False
                    logger.warning(
                        "Model (%s) beklenmeyen hata verdi (%s); sıradaki modele geçiliyor...",
                        model_name,
                        exc,
                    )

        # Single bounded cooldown pass on transient 503 models (never on 429 quota-exhausted models)
        if transient_models:
            if self._cooldown_seconds > 0:
                logger.warning(
                    "Tüm birincil havuzlar meşgul; %.1f sn beklenip geçici yoğunluktaki modellere son deneme yapılıyor...",
                    self._cooldown_seconds,
                )
                time.sleep(self._cooldown_seconds)
            for retry_model in transient_models[:2]:
                try:
                    return self._call_model_once(prompt, retry_model)
                except Exception as retry_exc:
                    if _is_auth_or_config_error(retry_exc):
                        raise
                    last_exc = retry_exc
                    if not (
                        _is_quota_exhausted_error(retry_exc)
                        or _is_transient_capacity_error(retry_exc)
                    ):
                        all_capacity_or_quota = False

        if self._allow_heuristic_fallback and all_capacity_or_quota:
            logger.warning(
                "Tüm yapay zeka model havuzları yoğun veya kota sınırında (503/429); "
                "iş akışının çökmemesi için yerel kural tabanlı Baş Editör yedek değerlendirmesi devreye alındı."
            )
            return _heuristic_fallback_evaluate(batch)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Değerlendirme yapılabilecek aktif model bulunamadı.")

    def _evaluate_batch(
        self,
        batch: list[RawArticle],
        *,
        recent_headlines: list[str] | None = None,
    ) -> dict[str, ArticleEvaluation]:
        payload_items = [
            {
                "article_id": item.article_id,
                "source": item.source_name,
                "authority_tier": item.authority_tier,
                "language": item.language,
                "title": item.title,
                "summary": item.summary[:240],
            }
            for item in batch
        ]
        recent_section = ""
        if recent_headlines:
            bullet_lines = "\n".join(f"- {h}" for h in recent_headlines[:40])
            recent_section = (
                "DAHA ÖNCE YAYIMLANAN HABERLER (Bunlarla aynı olayı/veriyi anlatan adaylara "
                "duplicate_of_id='PREVIOUS' ve is_relevant=false ver):\n"
                f"{bullet_lines}\n\n"
            )

        prompt = (
            recent_section
            + "Aşağıdaki aday haber listesini incele, aynı olayı anlatan farklı kaynakları tekilleştir "
            "ve her bir article_id için eksiksiz değerlendirme üret:\n\n"
            + json.dumps(payload_items, ensure_ascii=False, indent=2)
        )

        parsed = self._evaluate_with_cascade(prompt, batch)
        raw_by_id = {item.article_id: item for item in batch}
        valid_batch_ids = set(raw_by_id.keys())
        by_id: dict[str, ArticleEvaluation] = {}

        for evaluation in parsed.evaluations:
            if evaluation.article_id not in valid_batch_ids:
                continue
            dup_ref = (evaluation.duplicate_of_id or "").strip()
            is_prev_dup = dup_ref.upper().startswith("PREV")
            is_batch_dup = (
                bool(dup_ref)
                and dup_ref in valid_batch_ids
                and dup_ref != evaluation.article_id
            )
            # Enforce consistency if Gemini marked an item as a duplicate
            if is_prev_dup or is_batch_dup:
                evaluation = evaluation.model_copy(
                    update={
                        "is_relevant": False,
                        "importance_score": min(evaluation.importance_score, 4),
                    }
                )
            elif evaluation.category == NewsCategory.IRRELEVANT or not evaluation.is_relevant:
                evaluation = evaluation.model_copy(
                    update={
                        "is_relevant": False,
                        "category": NewsCategory.IRRELEVANT,
                    }
                )
            else:
                evaluation = enforce_category_domain_rules(
                    raw_by_id[evaluation.article_id], evaluation
                )
            by_id[evaluation.article_id] = evaluation

        # Safety fallback if the model omitted any item from the batch
        for item in batch:
            if item.article_id not in by_id:
                logger.warning(
                    "Model çıktısında eksik haber kimliği tespit edildi: %s",
                    item.article_id,
                )
                by_id[item.article_id] = ArticleEvaluation(
                    article_id=item.article_id,
                    is_relevant=False,
                    category=NewsCategory.IRRELEVANT,
                    importance_score=1,
                    duplicate_of_id=None,
                    turkish_headline=item.title,
                    summary_text="",
                    impact_note="",
                )

        return by_id
