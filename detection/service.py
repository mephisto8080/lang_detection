import logging

from fastapi import APIRouter, HTTPException

import config
from detection.detector import detector
from detection.schemas import (
    DetectRequest,
    DetectResponse,
    BatchDetectRequest,
    BatchDetectResponse,
    LanguagePrediction,
    HealthResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Language Detection"])


def _build_response(text: str, top_n: int) -> DetectResponse:
    """Run detection on a single text and build the response."""
    predictions_raw = detector.detect(text, top_n=top_n)

    if not predictions_raw:
        raise HTTPException(status_code=400, detail="Could not detect language. Text may be empty or invalid.")

    predictions = [LanguagePrediction(**p) for p in predictions_raw]
    detected = predictions[0]
    text_length = len(text.strip())

    confidence_note = None
    if text_length < config.SHORT_TEXT_CHAR_LIMIT:
        confidence_note = (
            f"Short text ({text_length} chars). "
            "Results may be less accurate. Provide more text for better detection."
        )
    elif detected.confidence < config.MIN_CONFIDENCE_THRESHOLD:
        confidence_note = (
            f"Low confidence ({detected.confidence:.2f}). "
            "The text may contain mixed languages or be ambiguous."
        )

    return DetectResponse(
        detected_language=detected,
        predictions=predictions,
        text_length=text_length,
        confidence_note=confidence_note,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect_language(request: DetectRequest):
    """Detect the language of a given text string."""
    logger.info(f"Detecting language for text of length {len(request.text)}")
    return _build_response(request.text, request.top_n)


@router.post("/detect/batch", response_model=BatchDetectResponse)
async def detect_language_batch(request: BatchDetectRequest):
    """Detect languages for multiple texts in a single request."""
    logger.info(f"Batch detection for {len(request.texts)} texts")

    results = []
    for text in request.texts:
        results.append(_build_response(text, request.top_n))

    return BatchDetectResponse(results=results, total_texts=len(results))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    return HealthResponse(
        status="healthy" if detector.model_loaded else "unhealthy",
        model_loaded=detector.model_loaded,
        supported_languages=176,
    )
