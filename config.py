import os

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8010"))

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "models/lid.176.bin")
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Detection defaults
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "5"))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.3"))
SHORT_TEXT_CHAR_LIMIT = int(os.getenv("SHORT_TEXT_CHAR_LIMIT", "50"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
