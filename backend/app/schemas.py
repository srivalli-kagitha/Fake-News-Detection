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


class ClaimsRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000)
    language: Language = "auto"
