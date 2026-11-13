from dataclasses import dataclass
from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = ROOT / "corpus"
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    model: str | None = os.getenv("OPENAI_MODEL")
    embedding_model: str | None = os.getenv("OPENAI_EMBEDDING_MODEL")

settings = Settings()
