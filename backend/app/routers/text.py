from fastapi import APIRouter

from ..config import get_settings
from ..database import save_check
from ..schemas import ClaimsRequest, ExplainRequest, TextRequest
from ..services.claim_verifier import verify_text
from ..services.common import ModelNotReadyError
from ..services.language import resolve_language
from ..services.text_classifier import get_text_classifier
from ..services.text_explainer import explain_text

router = APIRouter(tags=["text"])


def combine(classification: dict | None, verification: dict | None) -> dict | None:
    """Overall verdict.

    - Claims checked against evidence: weighted average of the evidence (EVIDENCE_WEIGHT, default 0.7)
      and the writing-style classifier (the rest).
    - Claim checking ran but found no evidence for any claim: "Unverified". The classifier only judges
      how the text is written, so it is reported as a style signal, not as a true/false verdict.
    - Claim checking switched off: the classifier alone, as the user asked.
    """
    has_evidence = bool(verification) and verification["summary"]["verdict"] != "Unverified"
    if verification and not has_evidence:
        return {"verdict": "Unverified", "fake_risk": None, "based_on": [],
                "style_fake_probability": round(classification["fake_probability"], 4) if classification else None}
    if not classification and not has_evidence:
        return None
    if not has_evidence:
        risk, based_on = classification["fake_probability"], ["classifier"]
    elif not classification:
        risk, based_on = 1.0 - verification["summary"]["credibility_score"], ["claim_evidence"]
    else:
        w = get_settings().evidence_weight
        risk = (1 - w) * classification["fake_probability"] + w * (1.0 - verification["summary"]["credibility_score"])
        based_on = ["classifier", "claim_evidence"]
    return {"verdict": "Fake" if risk >= 0.5 else "Real", "fake_risk": round(risk, 4), "based_on": based_on}


@router.post("/text/analyze")
def analyze_text(req: TextRequest):
    language = resolve_language(req.text, req.language)
    classification, classifier_error = None, None

    try:
        classifier = get_text_classifier(language)
        classification = classifier.predict(req.text)
        if req.explain:
            classification["explanation"] = explain_text(classifier, req.text)
    except ModelNotReadyError as exc:
        if not req.verify_claims:
            raise
        classifier_error = str(exc)

    verification = verify_text(req.text, language) if req.verify_claims else None
    overall = combine(classification, verification)

    verdict = (overall or {}).get("verdict") or (verification or {}).get("summary", {}).get("verdict", "Unknown")
    record_id = save_check(
        kind="text",
        input_preview=req.text,
        language=language,
        verdict=verdict,
        confidence=(max(overall["fake_risk"], 1 - overall["fake_risk"]) if overall and overall["fake_risk"] is not None
                    else None if overall else classification["confidence"] if classification else None),
        details={
            "classification": {k: v for k, v in (classification or {}).items() if k != "explanation"},
            "claims_summary": verification["summary"] if verification else None,
        },
    )
    return {
        "id": record_id,
        "language": language,
        "classification": classification,
        "classifier_error": classifier_error,
        "verification": verification,
        "overall": overall,
    }


@router.post("/text/explain")
def explain(req: ExplainRequest):
    """LIME word weights on their own, so the UI can show the verdict first and the highlights after."""
    language = resolve_language(req.text, req.language)
    return explain_text(get_text_classifier(language), req.text)


@router.post("/claims/verify")
def verify_claims(req: ClaimsRequest):
    language = resolve_language(req.text, req.language)
    result = verify_text(req.text, language)
    result["id"] = save_check(
        kind="claims",
        input_preview=req.text,
        language=language,
        verdict=result["summary"]["verdict"],
        confidence=result["summary"]["credibility_score"],
        details={"summary": result["summary"]},
    )
    return result
