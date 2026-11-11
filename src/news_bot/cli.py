"""Command-line interface for the Economy & Banking News Bot."""

from __future__ import annotations

import argparse
import logging
import sys

from news_bot.config import Settings
from news_bot.pipeline import NewsPipeline


def configure_logging(level_name: str) -> None:
    """Configure structured console logging."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="news-bot",
        description="Türkiye Ekonomisi, Dünya Ekonomisi ve Türkiye Bankacılık Haber Botu",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "scan", "digest", "both", "fetch-preview"),
        default="auto",
        help=(
            "Çalışma modu: "
            "'auto' (kritik tarama + saat dilimine göre sabah/akşam bülteni), "
            "'scan' (yalnızca tarama ve kritik anlık bildirim), "
            "'digest' (bekleyen bülten kuyruğunu gönder), "
            "'both' (önce tarama, hemen ardından bülten gönder), "
            "'fetch-preview' (API anahtarı gerekmeden kaynak çekme ve yerel filtre testi)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Telegram'a mesaj göndermeden ve veritabanını değiştirmeden konsolda önizleme yap.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("news_bot.cli")

    require_gemini = args.mode in {"auto", "scan", "both"}
    try:
        if args.mode != "fetch-preview":
            settings.validate_for_live_run(
                dry_run=args.dry_run, require_gemini=require_gemini
            )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    pipeline = NewsPipeline(settings, dry_run=args.dry_run)

    try:
        if args.mode == "fetch-preview":
            pipeline.run_fetch_preview()
        elif args.mode == "auto":
            pipeline.run_auto()
        elif args.mode == "scan":
            pipeline.run_scan()
        elif args.mode == "digest":
            pipeline.run_digest()
        elif args.mode == "both":
            pipeline.run_scan()
            pipeline.run_digest()
    except Exception as exc:
        logger.exception("Çalıştırma sırasında beklenmeyen hata oluştu: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

