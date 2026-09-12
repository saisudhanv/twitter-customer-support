"""
Configuration loader for the Hiver support agent.

Loads from .env file and environment variables, with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration for the Hiver support agent."""

    # Environment
    debug: bool = Field(default=False, description="Enable debug logging")
    verbose: bool = Field(default=True, description="Enable verbose output")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider: gemini, openai, anthropic, or ollama")
    gemini_api_key: str | None = Field(default=None, description="Gemini API key")
    gemini_model: str = Field(default="gemini-3.6-flash", description="Gemini model")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo", description="OpenAI model")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    claude_model: str = Field(default="claude-3-opus-20240229", description="Claude model")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="mistral", description="Ollama model")

    # Data Configuration
    data_sample_size: int = Field(default=500, description="Sample size for quick runs")
    eval_sample_size: int = Field(default=200, description="Evaluation set size")

    # Model Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for retrieval"
    )
    retrieval_top_k: int = Field(default=3, description="Top K retrieved examples")

    # Paths
    data_raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    data_processed_dir: Path = Field(default=Path("data/processed"), description="Processed data directory")
    data_golden_dir: Path = Field(default=Path("data/golden"), description="Golden set directory")
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Artifacts directory")
    cache_dir: Path = Field(default=Path("artifacts/cached_outputs"), description="Cache directory")

    # Caching
    use_llm_cache: bool = Field(default=True, description="Cache LLM outputs")
    use_embedding_cache: bool = Field(default=True, description="Cache embeddings")

    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_config() -> Config:
    """
    Load configuration from .env file and environment.
    
    Returns:
        Config: Configuration object
    """
    # Load .env if it exists
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    # Never load .env.example: it is documentation, not a credentials file.

    # Map environment variables to config
    config_dict = {
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "verbose": os.getenv("VERBOSE", "true").lower() == "true",
        "random_seed": int(os.getenv("RANDOM_SEED", "42")),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "claude_model": os.getenv("CLAUDE_MODEL", "claude-3-opus-20240229"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "mistral"),
        "data_sample_size": int(os.getenv("DATA_SAMPLE_SIZE", "500")),
        "eval_sample_size": int(os.getenv("EVAL_SAMPLE_SIZE", "200")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "retrieval_top_k": int(os.getenv("RETRIEVAL_TOP_K", "3")),
        "use_llm_cache": os.getenv("USE_LLM_CACHE", "true").lower() == "true",
        "use_embedding_cache": os.getenv("USE_EMBEDDING_CACHE", "true").lower() == "true",
    }

    return Config(**config_dict)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    config = get_config()
    print("Configuration loaded:")
    print(config.model_dump_json(indent=2))
