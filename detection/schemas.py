from pydantic import BaseModel, Field
from typing import Optional


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to detect language for")
    top_n: int = Field(default=3, ge=1, le=20, description="Number of top dominant language predictions to return")


class BatchDetectRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of texts to detect languages for")
    top_n: int = Field(default=3, ge=1, le=20, description="Number of top predictions per text")


class LanguagePrediction(BaseModel):
    language_code: str
    language_name: str
    confidence: float


class DetectResponse(BaseModel):
    detected_language: LanguagePrediction
    predictions: list[LanguagePrediction]
    text_length: int
    confidence_note: Optional[str] = None


class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]
    total_texts: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_languages: int
