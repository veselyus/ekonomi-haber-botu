"""Multi-stage URL, entity/number-aware, and post-Gemini event deduplication."""

from __future__ import annotations

import logging
import re

from rapidfuzz import fuzz

from news_bot.models import ArticleEvaluation, EnrichedArticle, NewsCategory, RawArticle
from news_bot.prefilter import normalize_title_for_matching
from news_bot.storage import ArticleStorage

logger = logging.getLogger(__name__)

NUMBER_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Commercial banks whose separate corporate announcements (syndications, balance sheets)
# look structurally identical ("X 500 milyon dolar sendikasyon kredisi sağladı")
# and must NEVER be merged across different banks.
DISTINCT_BANKS: frozenset[str] = frozenset(
    {
        "akbank",
        "garanti",
        "işbankası",
        "yapıkredi",
        "vakıfbank",
        "ziraat",
        "halkbank",
        "denizbank",
        "qnb",
        "teb",
        "şekerbank",
        "albaraka",
        "kuveyttürk",
        "tskb",
        "ing",
        "hsbc",
    }
)

REPORTING_STOPWORDS: frozenset[str] = frozenset(
    {
        "ve",
        "ile",
        "için",
        "bir",
        "bu",
        "da",
        "de",
        "den",
        "dan",
        "ten",
        "tan",
        "olan",
        "olarak",
        "göre",
        "karşı",
        "sonra",
        "önce",
        "yeni",
        "ayı",
        "ayında",
        "yılı",
        "yılında",
        "açıkladı",
        "açıklandı",
        "duyurdu",
        "duyuruldu",
        "bildirdi",
        "belirtti",
        "paylaştı",
        "söyledi",
        "etti",
        "oldu",
        "gerçekleşti",
        "seviyesinde",
        "seviyesine",
        "oranında",
        "oranı",
        "bazda",
        "bandında",
        "aralığında",
        "rakamları",
        "rakamlarını",
        "verileri",
        "verilerini",
        "verisi",
        "kararını",
        "kararı",
        "ilişkin",
        "yönelik",
        "hakkında",
        "tutarında",
        "sağladı",
        "temin",
        "the",
        "of",
        "in",
        "to",
        "for",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "its",
        "says",
        "said",
        "announces",
        "announced",
    }
)

IGNORED_CALENDAR_NUMBERS: frozenset[str] = frozenset(
    {"2024", "2025", "2026", "2027", "2028"}
)


def _stem_token(tok: str) -> str:
    """Reduce Turkish/English inflected words to a compact root for event matching."""
    if NUMBER_TOKEN_RE.match(tok):
        return tok
    if tok.startswith("faiz"):
        return "faiz"
    if tok.startswith("enflasyon"):
        return "enflas"
    if tok.startswith("kredi"):
        return "kredi"
    if tok.startswith("mevduat"):
        return "mevdu"
    if tok.startswith("rezerv"):
        return "rezer"
    if tok.startswith("ihracat"):
        return "ihrac"
    if len(tok) >= 6:
        return tok[:5]
    return tok


def _extract_event_features(norm_text: str) -> tuple[set[str], list[float], set[str]]:
    """Extract (core_stems, key_numbers, mentioned_banks) from a normalized headline."""
    tokens = norm_text.split()
    banks = {t for t in tokens if t in DISTINCT_BANKS}
    numbers: list[float] = []
    core_stems: set[str] = set()

    for idx, tok in enumerate(tokens):
        if tok in REPORTING_STOPWORDS:
            continue
        if NUMBER_TOKEN_RE.match(tok):
            if tok in IGNORED_CALENDAR_NUMBERS:
                continue
            # Ignore '100' when part of 'bist 100'
            if tok == "100" and idx > 0 and tokens[idx - 1] == "bist":
                core_stems.add("bist100")
                continue
            try:
                numbers.append(float(tok))
            except ValueError:
                pass
            core_stems.add(tok)
        elif len(tok) >= 2:
            core_stems.add(_stem_token(tok))

    return core_stems, numbers, banks


def _numbers_match(nums_a: list[float], nums_b: list[float]) -> bool:
    """Return True if at least one numeric figure matches (allowing minor rounding like 48.58 vs 48.6)."""
    for a in nums_a:
        for b in nums_b:
            if abs(a - b) <= 0.12:
                return True
            denom = max(abs(a), abs(b), 1.0)
            if abs(a - b) / denom <= 0.015:
                return True
    return False


def compute_headline_similarity(norm_a: str, norm_b: str) -> float:
    """
    Compute event similarity score (0..100) between two normalized headlines.
    Combines RapidFuzz with Turkish stem + numeric figure matching while guarding
    against merging separate corporate bank announcements.
    """
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 100.0

    core_a, nums_a, banks_a = _extract_event_features(norm_a)
    core_b, nums_b, banks_b = _extract_event_features(norm_b)

    # Guard 1: Different specific banks (e.g., Akbank vs Garanti BBVA) are never duplicates
    if banks_a and banks_b and banks_a.isdisjoint(banks_b):
        return 0.0

    sort_score = float(fuzz.token_sort_ratio(norm_a, norm_b))
    set_score = float(fuzz.token_set_ratio(norm_a, norm_b))
    base_score = 0.55 * sort_score + 0.45 * set_score

    if not core_a or not core_b:
        return base_score

    shared_core = core_a & core_b
    # Also count rounded numeric matches (e.g., 48.58 vs 48.6) toward shared_core
    has_shared_num = _numbers_match(nums_a, nums_b)
    has_conflicting_nums = bool(nums_a and nums_b and not has_shared_num)

    # Guard 2: Both headlines report explicit numbers, and none of the numbers match
    if has_conflicting_nums and base_score < 88.0:
        return min(base_score, 60.0)

    min_len = min(len(core_a), len(core_b))
    union_len = len(core_a | core_b)
    subset_ratio = len(shared_core) / min_len if min_len > 0 else 0.0
    jaccard_ratio = len(shared_core) / union_len if union_len > 0 else 0.0

    # Boost 1: Shared numeric figure + strong core stem overlap (catches paraphrased same-event stories)
    if has_shared_num and len(shared_core) >= 3 and subset_ratio >= 0.60:
        return max(base_score, 84.0)

    # Boost 2: Non-numeric policy/regulatory stories with high stem overlap
    if not has_conflicting_nums and len(shared_core) >= 4 and subset_ratio >= 0.72 and jaccard_ratio >= 0.52:
        return max(base_score, 82.0)

    return base_score


def deduplicate_candidates(
    articles: list[RawArticle],
    storage: ArticleStorage,
    *,
    fuzzy_threshold: float = 74.0,
    lookback_hours: int = 48,
) -> tuple[list[RawArticle], int]:
    """
    Eliminate already-stored URLs and fuzzy/event headline duplicates before Gemini.

    Higher-authority sources (Tier 1 regulators/central banks before Tier 2 direct media
    before Tier 3 aggregators) win whenever multiple outlets report the same event.
    """
    existing_ids, existing_urls = storage.get_existing_ids_and_urls()
    recent_titles = storage.get_recent_normalized_titles(lookback_hours=lookback_hours)

    prioritized = sorted(
        articles,
        key=lambda item: (item.authority_tier, -len(item.summary)),
    )

    accepted: list[RawArticle] = []
    accepted_norm_titles: list[tuple[str, str]] = [
        (stored_id, norm_title) for stored_id, norm_title, _ in recent_titles
    ]
    seen_batch_ids: set[str] = set()
    seen_batch_urls: set[str] = set()
    dropped = 0

    for candidate in prioritized:
        if (
            candidate.article_id in existing_ids
            or candidate.canonical_url in existing_urls
            or candidate.article_id in seen_batch_ids
            or candidate.canonical_url in seen_batch_urls
        ):
            dropped += 1
            continue

        norm_candidate = normalize_title_for_matching(candidate.title)
        if not norm_candidate:
            dropped += 1
            continue

        is_fuzzy_dup = False
        for _, existing_norm in accepted_norm_titles:
            if compute_headline_similarity(norm_candidate, existing_norm) >= fuzzy_threshold:
                is_fuzzy_dup = True
                break

        if is_fuzzy_dup:
            dropped += 1
            continue

        seen_batch_ids.add(candidate.article_id)
        seen_batch_urls.add(candidate.canonical_url)
        accepted_norm_titles.append((candidate.article_id, norm_candidate))
        accepted.append(candidate)

    return accepted, dropped


def suppress_post_eval_duplicates(
    candidates: list[RawArticle],
    evaluations_by_id: dict[str, ArticleEvaluation],
    storage: ArticleStorage,
    *,
    fuzzy_threshold: float = 74.0,
    lookback_hours: int = 48,
) -> tuple[dict[str, ArticleEvaluation], int]:
    """
    Second-stage deduplication after Gemini generates standardized ASD-STE100 Turkish headlines.

    Catches cross-source, cross-language, and cross-scan duplicates whose raw headlines
    differed stylistically in newspapers but converge once rewritten into canonical ASD-STE100.
    """
    updated = dict(evaluations_by_id)
    stored_sigs = storage.get_recent_relevant_signatures(lookback_hours=lookback_hours)

    reference_norms: list[tuple[str, str]] = []
    for stored_id, norm_raw, norm_tr, _tier in stored_sigs:
        if norm_raw:
            reference_norms.append((stored_id, norm_raw))
        if norm_tr and norm_tr != norm_raw:
            reference_norms.append((stored_id, norm_tr))

    ordered_candidates = sorted(
        candidates,
        key=lambda raw: (
            raw.authority_tier,
            -updated[raw.article_id].importance_score,
            -len(raw.summary),
        ),
    )

    suppressed_count = 0
    for raw in ordered_candidates:
        ev = updated[raw.article_id]
        if not ev.is_relevant or ev.category == NewsCategory.IRRELEVANT:
            continue

        norm_raw = normalize_title_for_matching(raw.title)
        norm_tr = normalize_title_for_matching(ev.turkish_headline)
        probe_texts = [t for t in (norm_tr, norm_raw) if t]

        matched_id: str | None = None
        for ref_id, ref_norm in reference_norms:
            if ref_id == raw.article_id:
                continue
            if any(
                compute_headline_similarity(probe, ref_norm) >= fuzzy_threshold
                for probe in probe_texts
            ):
                matched_id = ref_id
                break

        if matched_id is not None:
            logger.info(
                "Mükerrer olay elendi (%s -> %s ile aynı gelişme): %s",
                raw.source_name,
                matched_id,
                ev.turkish_headline,
            )
            updated[raw.article_id] = ev.model_copy(
                update={
                    "is_relevant": False,
                    "duplicate_of_id": matched_id,
                    "importance_score": min(ev.importance_score, 4),
                }
            )
            suppressed_count += 1
            continue

        for probe in probe_texts:
            reference_norms.append((raw.article_id, probe))

    return updated, suppressed_count


def deduplicate_enriched_articles(
    items: list[EnrichedArticle],
    *,
    fuzzy_threshold: float = 74.0,
) -> list[EnrichedArticle]:
    """
    Final safety gate before Telegram delivery: ensure no two queued articles
    in the same digest describe the same underlying news event.
    """
    if len(items) <= 1:
        return list(items)

    prioritized = sorted(
        items,
        key=lambda x: (
            x.raw.authority_tier,
            -x.evaluation.importance_score,
            -len(x.raw.summary),
        ),
    )

    unique_items: list[EnrichedArticle] = []
    seen_norms: list[str] = []

    for item in prioritized:
        norm_tr = normalize_title_for_matching(item.evaluation.turkish_headline)
        norm_raw = normalize_title_for_matching(item.raw.title)
        probes = [t for t in (norm_tr, norm_raw) if t]

        if any(
            compute_headline_similarity(probe, existing) >= fuzzy_threshold
            for probe in probes
            for existing in seen_norms
        ):
            logger.info(
                "Bülten gönderimi öncesi çiftleyen haber elendi (%s): %s",
                item.raw.source_name,
                item.evaluation.turkish_headline,
            )
            continue

        unique_items.append(item)
        seen_norms.extend(probes)

    return unique_items
