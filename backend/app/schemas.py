from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["auto", "en", "te"]


class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000)
    language: Language = "auto"
    explain: bool = True
    verify_claims: bool = False


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000)
    language: Language = "auto"
    record_id: int | None = None  # history entry to add the highlights to


class ClaimsRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000)
    language: Language = "auto"
