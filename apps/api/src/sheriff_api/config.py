import os
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "pixel-sheriff-api"
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_pass: str = "postgres"
    db_name: str = "pixel_sheriff"
    database_url: str | None = None
    cors_origins: str = "http://localhost:3000"
    storage_root: str = "./data"
    redis_url: str = "redis://localhost:6379/0"

    @model_validator(mode="after")
    def apply_database_url_default(self) -> "Settings":
        if self.database_url:
            return self

        has_db_env = any(os.getenv(name) for name in ("DB_HOST", "DB_PORT", "DB_USER", "DB_PASS", "DB_NAME"))
        if has_db_env:
            user = quote_plus(self.db_user)
            password = quote_plus(self.db_pass)
            self.database_url = f"postgresql+asyncpg://{user}:{password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            self.database_url = "sqlite+aiosqlite:///./pixel_sheriff.db"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
