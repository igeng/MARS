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
    QWEN_MODEL: str = "qwen3.5-flash"

    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_CODER_MODEL: str = "deepseek-coder"

    KIMI_MODEL: str = "kimi-k2.5"

    GLM_MODEL: str = "glm-4.7-flash"  # 免费模型，作为兜底

    # ---- API endpoints ----
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

    # ---- Academic search ----
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    MAX_PAPERS_PER_SEARCH: int = Field(default=50, gt=0)
    MAX_PAPERS_FOR_ANALYSIS: int = Field(default=20, gt=0)
    # HTTP read timeout (seconds) for arXiv API requests.
    # Increase if you experience frequent timeout errors on slow networks.
    ARXIV_SEARCH_TIMEOUT: int = Field(default=30, gt=0)

    # ---- Application ----
    DEFAULT_LLM_PROVIDER: str = "qwen"
    LOG_LEVEL: str = "INFO"
    OUTPUT_DIR: Path = Path("./output")
    # Maximum tool-call iterations per agent (CrewAI Agent max_iter).
    # Increase for complex multi-step workflows; decrease to reduce cost.
    AGENT_MAX_ITER: int = Field(default=10, ge=1)

    # ---- GLM rate-limit handling ----
    # Retry attempts on RateLimitError before switching to the next provider.
    GLM_RATE_LIMIT_MAX_RETRIES: int = 3
    # Base delay in seconds for exponential back-off (doubles each retry).
    GLM_RATE_LIMIT_RETRY_DELAY: float = 5.0

    # ---- Crew memory ----
    # Set to True only when an OpenAI-compatible embedding API key is available.
    # CrewAI's built-in memory layer defaults to OpenAI embeddings; enabling this
    # without a compatible key raises a CHROMA_OPENAI_API_KEY validation error.
    ENABLE_MEMORY: bool = False

    # ---- Database ----
    DATABASE_URL: str = "sqlite:///./mars.db"

    # ---- API server ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


# Module-level singleton so that ``from mars.config import settings`` keeps
# working everywhere without any change to call sites.
settings = MarsSettings()
