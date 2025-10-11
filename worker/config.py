# worker/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Centralized, env-driven configuration for the worker.
    Override via environment variables or a .env file at repo root.
    """
    api_key: str | None = Field(default=None, env="WORKER_API_KEY")
    log_level: str = Field(default="INFO", env="WORKER_LOG_LEVEL")
    default_timeout: int = Field(default=1800, env="WORKER_DEFAULT_TIMEOUT")  # seconds
    max_threads: int = Field(default=4, env="WORKER_MAX_THREADS")
    reports_dir: str = Field(default="reports", env="RUNNER_REPORTS_DIR")
    max_body_bytes: int = Field(default=2_000_000)  # ~2MB payload cap
    enable_metrics: bool = Field(default=False, env="WORKER_ENABLE_METRICS")
    metrics_port: int = Field(default=9108, env="WORKER_METRICS_PORT")

    # Pydantic v2 config: ignore unknown envs, load .env in UTF-8
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
