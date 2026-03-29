"""
MARS application settings.

Loads configuration from environment variables / .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)


# ---------------------------------------------------------------------------
# LLM Provider API keys
# ---------------------------------------------------------------------------

DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
MOONSHOT_API_KEY: str = os.getenv("MOONSHOT_API_KEY", "")
ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-max")
QWEN_PLUS_MODEL: str = os.getenv("QWEN_PLUS_MODEL", "qwen-plus")
QWEN_TURBO_MODEL: str = os.getenv("QWEN_TURBO_MODEL", "qwen-turbo")

DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_CODER_MODEL: str = os.getenv("DEEPSEEK_CODER_MODEL", "deepseek-coder")

KIMI_MODEL: str = os.getenv("KIMI_MODEL", "moonshot-v1-8k")
KIMI_LONG_MODEL: str = os.getenv("KIMI_LONG_MODEL", "moonshot-v1-128k")

GLM_MODEL: str = os.getenv("GLM_MODEL", "glm-4-plus")
GLM_AIR_MODEL: str = os.getenv("GLM_AIR_MODEL", "glm-4-air")

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
MOONSHOT_BASE_URL: str = "https://api.moonshot.cn/v1"

# ---------------------------------------------------------------------------
# Academic search settings
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_API_KEY: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
MAX_PAPERS_PER_SEARCH: int = int(os.getenv("MAX_PAPERS_PER_SEARCH", "50"))
MAX_PAPERS_FOR_ANALYSIS: int = int(os.getenv("MAX_PAPERS_FOR_ANALYSIS", "20"))

# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------

DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "qwen")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
