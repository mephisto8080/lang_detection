import os

# ==============================
# Server Configuration
# ==============================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8010))

# ==============================
# Model Configuration
# ==============================
MODEL_PATH = os.getenv("MODEL_PATH", "models/lid.176.bin")
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# ==============================
# Detection Settings
# ==============================
DEFAULT_TOP_N = 3
MIN_CONFIDENCE_THRESHOLD = 0.50
SHORT_TEXT_CHAR_LIMIT = 3

# ==============================
# Hinglish Detection Settings
# ==============================
HINGLISH_ENABLED = True
ENGLISH_WORD_THRESHOLD = 0.70      # 70%
MIN_WORDS_FOR_CHECK = 5            # minimum words before ratio check
ZIPF_FREQUENCY_THRESHOLD = 3.0     # English validity threshold