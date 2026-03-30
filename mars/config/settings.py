"""
MARS application settings.

Uses Pydantic BaseSettings for typed configuration with
automatic environment variable loading and .env file support.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# .env file path (project root, two levels up from this file)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class MarsSettings(BaseSettings):
    """Centralised settings loaded from environment variables / .env file."""

    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}

    # ---- LLM Provider API keys ----
    DASHSCOPE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    MOONSHOT_API_KEY: str = ""
    ZHIPU_API_KEY: str = ""

    # ---- Model identifiers ----
    QWEN_MODEL: str = "qwen-max"
    QWEN_PLUS_MODEL: str = "qwen-plus"
    QWEN_TURBO_MODEL: str = "qwen-turbo"

    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODER_MODEL: str = "deepseek-coder"

    KIMI_MODEL: str = "moonshot-v1-8k"
    KIMI_LONG_MODEL: str = "moonshot-v1-128k"

    GLM_MODEL: str = "glm-4-plus"
    GLM_AIR_MODEL: str = "glm-4-air"

    # ---- API endpoints ----
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    # ---- Academic search ----
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PAPERS_PER_SEARCH: int = Field(default=50, gt=0)
    MAX_PAPERS_FOR_ANALYSIS: int = Field(default=20, gt=0)

    # ---- Application ----
    DEFAULT_LLM_PROVIDER: str = "qwen"
    LOG_LEVEL: str = "INFO"
    OUTPUT_DIR: Path = Path("./output")

    # ---- Database ----
    DATABASE_URL: str = "sqlite:///./mars.db"

    # ---- API server ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


# Module-level singleton so that ``from mars.config import settings`` keeps
# working everywhere without any change to call sites.
settings = MarsSettings()
