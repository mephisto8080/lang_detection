from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging

from .schemas import DetectRequest, DetectResponse, BatchDetectRequest
from .detector import detector
import config as settings

router = APIRouter(prefix="/api/v1", tags=["Language Detection"])
logger = logging.getLogger(__name__)


@router.post("/detect", response_model=DetectResponse)
async def detect_language(request: DetectRequest):

    try:
        predictions = detector.detect(request.text, request.top_n)

        response = {
            "detected_language": predictions[0] if predictions else None,
            "predictions": predictions,
            "text_length": len(request.text),
            "confidence_note": None
        }

        if predictions and predictions[0]["confidence"] < settings.MIN_CONFIDENCE_THRESHOLD:
            response["confidence_note"] = "Low confidence detection."

        return response

    except Exception as e:
        logger.exception("Detection failed.")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/detect/batch")
async def detect_batch(request: BatchDetectRequest):

    results = []

    for text in request.texts:
        predictions = detector.detect(text, request.top_n)
        results.append({
            "text": text,
            "detected_language": predictions[0] if predictions else None,
            "predictions": predictions
        })

    return {"results": results}


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "supported_languages": detector.supported_languages,
        "hinglish_enabled": settings.HINGLISH_ENABLED
    }