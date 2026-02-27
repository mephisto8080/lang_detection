from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from detection.service import router
from detection.detector import detector

# ==================================================
# LOGGING
# ==================================================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/lang_detection.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==================================================
# FASTAPI INIT
# ==================================================
app = FastAPI(title="Language Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    detector.load_model()