"""
config.py — Centralised configuration loader for Research Agent.

Reads from .env (via python-dotenv) and exposes a frozen Config dataclass.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # IBM watsonx.ai credentials
    ibm_cloud_api_key: str = field(default_factory=lambda: _get("IBM_CLOUD_API_KEY"))
    watsonx_url: str = field(default_factory=lambda: _get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))
    watsonx_project_id: str = field(default_factory=lambda: _get("WATSONX_PROJECT_ID"))

    # IBM Granite model IDs
    granite_model_id: str = field(default_factory=lambda: _get("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"))
    granite_embedding_model_id: str = field(
        default_factory=lambda: _get("GRANITE_EMBEDDING_MODEL_ID", "ibm/slate-125m-english-rtrvr")
    )

    # IBM Watson Discovery (optional)
    discovery_api_key: str = field(default_factory=lambda: _get("DISCOVERY_API_KEY"))
    discovery_url: str = field(default_factory=lambda: _get("DISCOVERY_URL"))
    discovery_project_id: str = field(default_factory=lambda: _get("DISCOVERY_PROJECT_ID"))

    # Agent behaviour
    max_search_results: int = field(default_factory=lambda: _get_int("MAX_SEARCH_RESULTS", 10))
    max_new_tokens: int = field(default_factory=lambda: _get_int("MAX_NEW_TOKENS", 800))
    temperature: float = field(default_factory=lambda: _get_float("TEMPERATURE", 0.3))
    top_p: float = field(default_factory=lambda: _get_float("TOP_P", 0.9))

    # Paths
    output_dir: Path = field(default_factory=lambda: Path(_get("OUTPUT_DIR", "./output")))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))

    def validate(self) -> None:
        """Raise ValueError if mandatory IBM credentials are missing."""
        missing = []
        if not self.ibm_cloud_api_key or self.ibm_cloud_api_key == "your_ibm_cloud_api_key_here":
            missing.append("IBM_CLOUD_API_KEY")
        if not self.watsonx_project_id or self.watsonx_project_id == "your_watsonx_project_id_here":
            missing.append("WATSONX_PROJECT_ID")
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please fill in your .env file (copy .env.example → .env)."
            )


# Singleton instance
config = Config()


def setup_logging() -> logging.Logger:
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger("research_agent")


logger = setup_logging()
