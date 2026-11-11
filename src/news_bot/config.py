"""Application configuration powered by Pydantic Settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    critical_score_threshold: int = Field(
        default=10, ge=1, le=10, alias="CRITICAL_SCORE_THRESHOLD"
    )
    digest_score_threshold: int = Field(
        default=7, ge=1, le=10, alias="DIGEST_SCORE_THRESHOLD"
    )

    max_article_age_hours: int = Field(
        default=24, ge=1, le=168, alias="MAX_ARTICLE_AGE_HOURS"
    )
    fuzzy_duplicate_threshold: float = Field(
        default=80.0, ge=50.0, le=100.0, alias="FUZZY_DUPLICATE_THRESHOLD"
    )
    batch_size: int = Field(default=20, ge=1, le=40, alias="BATCH_SIZE")
    max_candidates_per_scan: int = Field(
        default=40, ge=5, le=200, alias="MAX_CANDIDATES_PER_SCAN"
    )
    http_timeout_seconds: float = Field(
        default=15.0, ge=3.0, le=60.0, alias="HTTP_TIMEOUT_SECONDS"
    )
    retention_days: int = Field(default=14, ge=2, le=90, alias="RETENTION_DAYS")
    send_empty_digest: bool = Field(default=False, alias="SEND_EMPTY_DIGEST")

    database_path: Path = Field(
        default=Path("data/news_state.db"), alias="DATABASE_PATH"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def validate_for_live_run(self, *, dry_run: bool, require_gemini: bool) -> None:
        """Raise ValueError if required credentials are missing for the chosen mode."""
        missing: list[str] = []
        if require_gemini and not self.gemini_api_key.strip():
            missing.append("GEMINI_API_KEY")
        if not dry_run:
            if not self.telegram_bot_token.strip():
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.telegram_chat_id.strip():
                missing.append("TELEGRAM_CHAT_ID")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Eksik ortam değişkenleri tespit edildi: {joined}. "
                "Lütfen .env dosyasını veya GitHub Secrets ayarlarını kontrol edin."
            )
