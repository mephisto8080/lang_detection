# Language Detection Service

FastAPI-based microservice for detecting languages in text using **FastText**. Supports **176 languages** including Burmese, Arabic, Hindi, Chinese, Japanese, Korean and more.

## Features

- Detect language from short or long text
- Returns top-N dominant languages with confidence scores
- Batch detection for multiple texts in a single request
- Auto-downloads the FastText model on first run
- Runs on CPU (no GPU required)
- Docker-ready for deployment

## Project Structure

```
lang_detection/
├── main.py                  # FastAPI app entry point
├── config.py                # Configuration (environment variables)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose for deployment
├── detection/
│   ├── __init__.py
│   ├── detector.py          # FastText model wrapper
│   ├── schemas.py           # Request/response models
│   └── service.py           # API endpoints
├── models/                  # FastText model (auto-downloaded)
└── logs/                    # Application logs
```

## Setup & Run

### 1. Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

The service starts on **http://localhost:8010**. The FastText model (~130MB) will be downloaded automatically on first startup.

### 2. Docker Setup

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up --build -d
```

## API Endpoints

### Swagger Docs

Open **http://localhost:8010/docs** in your browser for the interactive API documentation.

### POST `/api/v1/detect` — Detect Language

Detect the language of a single text.

**Request:**
```bash
curl -X POST http://localhost:8010/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a test", "top_n": 3}'
```

**Response:**
```json
{
  "detected_language": {
    "language_code": "en",
    "language_name": "English",
    "confidence": 0.9821
  },
  "predictions": [
    {"language_code": "en", "language_name": "English", "confidence": 0.9821},
    {"language_code": "de", "language_name": "German", "confidence": 0.0052},
    {"language_code": "nl", "language_name": "Dutch", "confidence": 0.0031}
  ],
  "text_length": 27,
  "confidence_note": null
}
```

### POST `/api/v1/detect/batch` — Batch Detection

Detect languages for multiple texts in one request.

**Request:**
```bash
curl -X POST http://localhost:8010/api/v1/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Bonjour le monde", "Hallo Welt"], "top_n": 3}'
```

### GET `/api/v1/health` — Health Check

```bash
curl http://localhost:8010/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "supported_languages": 176
}
```

## Configuration

All settings can be configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8010` | Server port |
| `MODEL_PATH` | `models/lid.176.bin` | Path to FastText model |
| `DEFAULT_TOP_N` | `5` | Default number of predictions |
| `MIN_CONFIDENCE_THRESHOLD` | `0.3` | Threshold for low confidence warning |
| `SHORT_TEXT_CHAR_LIMIT` | `50` | Char limit below which short text warning is shown |
| `LOG_LEVEL` | `INFO` | Logging level |

## Accuracy by Text Length

| Text Length | Accuracy |
|---|---|
| 1-2 words (< 10 chars) | ~70-80% |
| Short text (10-50 chars) | ~85-90% |
| Medium text (50-200 chars) | ~93-97% |
| Long text (200+ chars) | ~97-99% |

For best results, provide at least one full sentence (50+ characters).
