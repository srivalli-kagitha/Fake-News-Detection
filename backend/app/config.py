"""Application settings, loaded from backend/.env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "TruthLens AI"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'truthlens.db').as_posix()}"

    text_en_model: str = "models/text_en"
    text_te_model: str = "models/text_te"
    image_model: str = "models/image_detector"

    nli_en_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_multi_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    embed_model: str = "BAAI/bge-m3"

    evidence_dir: str = "data/evidence"
    google_factcheck_api_key: str = ""

    max_text_length: int = 256
    lime_num_samples: int = 200
    max_claims: int = 8
    evidence_top_k: int = 3
    evidence_min_similarity: float = 0.45
    max_batch_rows: int = 500
    device: str = "auto"

    @property
    def outputs_dir(self) -> Path:
        path = BACKEND_DIR / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def evidence_path(self) -> Path:
        return resolve_local(self.evidence_dir)


def is_local_reference(ref: str) -> bool:
    """True when `ref` looks like a folder path rather than a HuggingFace id."""
    return (
        ref.startswith(("models/", "models\\", "./", ".\\", "../", "..\\", "data/"))
        or Path(ref).is_absolute()
        or (BACKEND_DIR / ref).exists()
    )


def resolve_local(ref: str) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else BACKEND_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
