"""SQLite database (history of all checks)."""
import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class CheckRecord(Base):
    __tablename__ = "checks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)  # text | image | claims | batch_text | batch_image
    input_preview: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    verdict: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "input_preview": self.input_preview,
            "language": self.language,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "details": json.loads(self.details_json or "{}"),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def save_check(
    kind: str,
    input_preview: str,
    verdict: str,
    confidence: float | None = None,
    language: str | None = None,
    details: dict | None = None,
) -> int:
    with SessionLocal() as db:
        record = CheckRecord(
            kind=kind,
            input_preview=input_preview[:500],
            language=language,
            verdict=verdict,
            confidence=confidence,
            details_json=json.dumps(details or {}, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
        return record.id
