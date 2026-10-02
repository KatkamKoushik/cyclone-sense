from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "CycloneSense"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Host
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database Configuration (PostgreSQL primary; SQLite async fallback for local zero-dependency development)
    DATABASE_URL: str = "sqlite+aiosqlite:///./cyclonesense.db"
    DB_ECHO: bool = False
    
    # Redis & Asynchronous Job Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_TASK_ALWAYS_EAGER: bool = True  # Automatically executes jobs in-process if Redis is offline
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODEL_CHECKPOINT_DIR: Path = BASE_DIR / "models" / "checkpoints"
    
    # External Satellite Services Authentication (Real credentials, no fake data)
    NASA_EARTHDATA_USERNAME: Optional[str] = None
    NASA_EARTHDATA_PASSWORD: Optional[str] = None
    NASA_EARTHDATA_BEARER_TOKEN: Optional[str] = None
    ISRO_MOSDAC_API_KEY: Optional[str] = None
    
    # Scientific Limits
    MAX_ALLOWABLE_MISSING_PIXEL_PCT: float = 15.0
    MIN_BRIGHTNESS_TEMP_KELVIN: float = 160.0
    MAX_BRIGHTNESS_TEMP_KELVIN: float = 340.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure directories exist
settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
