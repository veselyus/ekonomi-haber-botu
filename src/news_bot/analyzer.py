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
- 9-10 (KRİTİK - Anlık Bildirim): TCMB/Fed/ECB faiz ve politika kararları, BDDK yeni regülasyon/kredi sınırı kararları, TÜİK enflasyon veya GSYH açıklaması, Türkiye veya Türk bankaları kredi notu değişiklikleri, büyük Türk bankalarının kritik sendikasyon/sermaye kararları, küresel piyasa şokları.
- 7-8 (ÖNEMLİ - Günlük Bülten): Önemli makro veriler (cari açık, sanayi üretimi, PMI, ABD tarım dışı istihdam/TÜFE), haftalık bankacılık kredi/mevduat eğilimleri, IMF/Dünya Bankası tahmin revizyonları, ekonomi yönetiminin somut politika açıklamaları.
- 1-6 (DÜŞÜK ÖNEM - Gönderilmeyecek): Rutin günlük piyasa açılış/kapanış tekrarı, somut karar içermeyen tahmin/yorumlar, ikincil mikro haberler.

MÜKERRER HABER KURALI (duplicate_of_id):
- Aynı listede (farklı dilde veya farklı kaynakta olsa bile, örn. Fed'in İngilizce duyurusu ile Bloomberg HT'nin Türkçe haberi) AYNI OLAYI anlatan birden fazla haber varsa:
  En yetkili kaynağa (authority_tier=1 resmi kurum öncelikli) tam değerlendirme yap (duplicate_of_id=""), diğer kopyalarda ise duplicate_of_id alanına o ana haberin article_id değerini yaz.

ÖZET KURALLARI:
- turkish_headline: Yabancı dildeki haberler dahil tüm ilgili haberler için tık tuzağından arındırılmış, rakam/kurum odaklı, kurumsal Türkçe başlık yaz.
- summary_bullets: İlgili (puan >= 7) haberler için 2 veya 3 maddelik, rakamları koruyan, yorumsuz ve özlü Türkçe maddeler yaz. İlgisiz veya düşük puanlı haberlerde boş liste [] bırakabilirsin.
- impact_note: İlgili (puan >= 7) haberler için piyasa veya bankacılık etkisini 1 kısa Türkçe cümleyle belirt.
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
                        "description": "Aynı listedeki başka bir haberin tekrarıysa ana haberin article_id değeri, değilse boş metin ('').",
                    },
                    "turkish_headline": {
                        "type": "string",
                        "description": "Kurumsal, net Türkçe başlık.",
                    },
                    "summary_bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "En fazla 3 maddelik özlü Türkçe özet.",
                    },
                    "impact_note": {
                        "type": "string",
                        "description": "Tek cümlelik Türkçe etki notu.",
                    },
                },
                "required": [
                    "article_id",
                    "is_relevant",
                    "category",
                    "importance_score",
                    "duplicate_of_id",
                    "turkish_headline",
                    "summary_bullets",
                    "impact_note",
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
        model: str = "gemini-3.8-flash",
        batch_size: int = 15,
        client: genai.Client | None = None,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._client = client or genai.Client(api_key=api_key)

    def evaluate_articles(
        self, articles: list[RawArticle]
    ) -> dict[str, ArticleEvaluation]:
        """Evaluate all candidate articles in batches and return mapping by article_id."""
        results: dict[str, ArticleEvaluation] = {}
        total = len(articles)
        for start in range(0, total, self._batch_size):
            if start > 0:
                # Polite pacing across batches to respect free-tier RPM quotas
                time.sleep(2.0)
            batch = articles[start : start + self._batch_size]
            batch_evals = self._evaluate_batch(batch)
            results.update(batch_evals)
        return results

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def _call_gemini_interaction(self, prompt: str) -> Any:
        return self._client.interactions.create(
            model=self._model,
            input=prompt,
            system_instruction=EDITOR_SYSTEM_PROMPT,
            generation_config={"thinking_level": "low"},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": EVALUATION_JSON_SCHEMA,
            },
            store=False,
        )

    def _evaluate_batch(
        self, batch: list[RawArticle]
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
        prompt = (
            "Aşağıdaki aday haber listesini incele ve her bir article_id için eksiksiz "
            "değerlendirme üret:\n\n"
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
            # Enforce consistency if Gemini marked an item as a duplicate within the batch
            if (
                evaluation.duplicate_of_id
                and evaluation.duplicate_of_id in valid_batch_ids
                and evaluation.duplicate_of_id != evaluation.article_id
            ):
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
                    summary_bullets=[],
                    impact_note="",
                )

        return by_id
