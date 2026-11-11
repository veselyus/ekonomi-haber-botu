"""URL and fuzzy headline deduplication with institutional authority priority."""

from __future__ import annotations

from rapidfuzz import fuzz

from news_bot.models import RawArticle
from news_bot.prefilter import normalize_title_for_matching
from news_bot.storage import ArticleStorage


def compute_headline_similarity(norm_a: str, norm_b: str) -> float:
    """Compute a balanced similarity score (0..100) between two normalized headlines."""
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 100.0
    sort_score = float(fuzz.token_sort_ratio(norm_a, norm_b))
    set_score = float(fuzz.token_set_ratio(norm_a, norm_b))
    return 0.55 * sort_score + 0.45 * set_score


def deduplicate_candidates(
    articles: list[RawArticle],
    storage: ArticleStorage,
    *,
    fuzzy_threshold: float = 80.0,
    lookback_hours: int = 48,
) -> tuple[list[RawArticle], int]:
    """
    Eliminate already-stored URLs and fuzzy headline duplicates.

    Higher-authority sources (Tier 1 regulators/central banks before Tier 2 media)
    are evaluated first so they win whenever multiple outlets report the same event.
    """
    existing_ids, existing_urls = storage.get_existing_ids_and_urls()
    recent_titles = storage.get_recent_normalized_titles(lookback_hours=lookback_hours)

    # Sort so Tier 1 comes before Tier 2, and richer summaries break ties.
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
        # Only fuzzy-match same-language headlines locally; cross-language duplicates
        # (e.g., English Fed release vs Turkish Bloomberg HT story) are caught by Gemini.
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

