"""Gemini API Chief Editor triage, classification, scoring, and Turkish summarization."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from google import genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from news_bot.models import (
    ArticleEvaluation,
    BatchEvaluationResponse,
    NewsCategory,
    RawArticle,
)

logger = logging.getLogger(__name__)

EDITOR_SYSTEM_PROMPT = """Sen üst düzey bir ekonomi ve bankacılık haber ajansının Baş Editörüsün.
Görevin; ham haber listesini inceleyip SIFIR GÜRÜLTÜ ilkesiyle yalnızca gerçekten önemli gelişmeleri seçmek,
sınıflandırmak, 1-10 arası önem puanı vermek ve net Türkçe yönetici özeti çıkarmaktır.

KATEGORİ KURALLARI:
1. TR_ECONOMY: Türkiye makroekonomisi (TCMB para politikası/faiz/rezerv, TÜİK enflasyon/büyüme/işgücü/dış ticaret, Hazine ve Maliye kararları, OVP, vergi düzenlemeleri, kredi derecelendirme kuruluşlarının Türkiye kararları, cari denge, önemli kur/tahvil şokları).
2. TR_BANKING: Türkiye bankacılık ve finans sektörü (BDDK düzenlemeleri ve haftalık verileri, TBB açıklamaları, zorunlu karşılıklar, kredi/mevduat faiz ve büyüme sınırları, sermaye yeterliliği, takipteki krediler, Türk bankalarının sendikasyon kredileri, tahvil ihraçları, bilançoları ve üst düzey KAP bildirimleri).
3. WORLD_ECONOMY: Küresel makroekonomi (Fed, ECB, BoJ, BoE faiz ve bilanço kararları, ABD/Avrupa/Çin büyüme ve enflasyon verileri, küresel ticaret tarifeleri/yaptırımlar, enerji ve emtia arz şokları, IMF/Dünya Bankası raporları, küresel bankacılık/finansal istikrar gelişmeleri).
4. IRRELEVANT: Yukarıdaki 3 alana doğrudan girmeyen veya gürültü niteliğindeki her şey! Magazin, spor, yerel asayiş, hava durumu, burç, bireysel hisse senedi tüyoları, altcoin spekülasyonu, clickbait ("altın kaç TL oldu", "emekliye müjde"), rutin banka personel alım/iş ilanları, mülakat/sınav rehberleri, somut veri içermeyen rutin köşe yazıları ve şirket reklam bültenleri KESİNLİKLE IRRELEVANT olmalı ve is_relevant=false işaretlenmelidir. (Not: Banka Genel Müdür / Yönetim Kurulu gibi üst düzey yönetici atamaları ilgili bankacılık haberidir ve TR_BANKING sayılır.)

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
    """Remove optional markdown fences around JSON payload if present."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


class GeminiNewsAnalyzer:
    """Evaluates candidate articles in batches using the Gemini Interactions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        batch_size: int = 40,
        client: genai.Client | None = None,
    ) -> None:
        self._model = model
        self._fallback_model = (
            "gemini-3.8-flash"
            if model == "gemini-3.5-flash-lite"
            else "gemini-3.5-flash-lite"
        )
        self._batch_size = batch_size
        self._client = client or genai.Client(api_key=api_key)

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
            if start > 0:
                time.sleep(1.5)
            batch = articles[start : start + self._batch_size]
            batch_evals = self._evaluate_batch(batch, recent_headlines=known_headlines)
            results.update(batch_evals)
            for ev in batch_evals.values():
                if ev.is_relevant and ev.category != NewsCategory.IRRELEVANT and ev.turkish_headline:
                    known_headlines.append(ev.turkish_headline)
        return results

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def _call_model_once(self, prompt: str, model_name: str) -> Any:
        return self._client.interactions.create(
            model=model_name,
            input=prompt,
            system_instruction=EDITOR_SYSTEM_PROMPT,
            generation_config={"thinking_level": "minimal"},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": EVALUATION_JSON_SCHEMA,
            },
            store=False,
            timeout=90.0,
        )

    def _call_gemini_interaction(self, prompt: str) -> Any:
        try:
            return self._call_model_once(prompt, self._model)
        except Exception as primary_exc:
            logger.warning(
                "Birincil model (%s) yanıt vermedi (%s); yedek modele (%s) geçiliyor...",
                self._model,
                primary_exc,
                self._fallback_model,
            )
            return self._call_model_once(prompt, self._fallback_model)

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
                "summary": item.summary,
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

        interaction = self._call_gemini_interaction(prompt)

        raw_text = _strip_json_fences(_extract_interaction_text(interaction))
        if not raw_text:
            raise RuntimeError("Gemini API boş yanıt döndürdü.")

        parsed = BatchEvaluationResponse.model_validate_json(raw_text)
        valid_batch_ids = {item.article_id for item in batch}
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
