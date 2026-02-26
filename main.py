import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from detection.detector import detector
from detection.service import router as detection_router

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/lang_detection.log"),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Language Detection Service",
    description="FastAPI service for detecting languages in text using FastText (176 languages supported)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)


@app.on_event("startup")
async def startup_event():
    """Load the FastText model on application startup."""
    logger.info("Starting Language Detection Service...")
    detector.load_model()
    logger.info("Service ready.")


@app.get("/")
async def root():
    return {
        "service": "Language Detection Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
