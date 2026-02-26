import os
import re
import logging
import requests
import numpy as np
import fasttext

import config

# Fix NumPy 2.x compatibility with fasttext
# fasttext uses np.array(obj, copy=False) which is incompatible with NumPy 2.x
_original_np_array = np.array

def _patched_np_array(obj, *args, **kwargs):
    if "copy" in kwargs and kwargs["copy"] is False:
        kwargs.pop("copy")
        return np.asarray(obj, *args, **kwargs)
    return _original_np_array(obj, *args, **kwargs)

np.array = _patched_np_array

logger = logging.getLogger(__name__)

# Mapping of fasttext language codes to human-readable names
LANGUAGE_NAMES = {
    "af": "Afrikaans", "als": "Alemannic", "am": "Amharic", "an": "Aragonese",
    "ar": "Arabic", "arz": "Egyptian Arabic", "as": "Assamese", "ast": "Asturian",
    "av": "Avaric", "az": "Azerbaijani", "azb": "South Azerbaijani", "ba": "Bashkir",
    "bar": "Bavarian", "bcl": "Central Bicolano", "be": "Belarusian", "bg": "Bulgarian",
    "bh": "Bihari", "bn": "Bengali", "bo": "Tibetan", "bpy": "Bishnupriya",
    "br": "Breton", "bs": "Bosnian", "bxr": "Buriat", "ca": "Catalan",
    "cbk": "Chavacano", "ce": "Chechen", "ceb": "Cebuano", "ckb": "Central Kurdish",
    "co": "Corsican", "cs": "Czech", "cv": "Chuvash", "cy": "Welsh",
    "da": "Danish", "de": "German", "diq": "Zazaki", "dsb": "Lower Sorbian",
    "dty": "Doteli", "dv": "Divehi", "el": "Greek", "eml": "Emilian-Romagnol",
    "en": "English", "eo": "Esperanto", "es": "Spanish", "et": "Estonian",
    "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "frr": "Northern Frisian", "fy": "Western Frisian", "ga": "Irish", "gd": "Scottish Gaelic",
    "gl": "Galician", "gn": "Guarani", "gom": "Goan Konkani", "gu": "Gujarati",
    "gv": "Manx", "he": "Hebrew", "hi": "Hindi", "hif": "Fiji Hindi",
    "hr": "Croatian", "hsb": "Upper Sorbian", "ht": "Haitian Creole", "hu": "Hungarian",
    "hy": "Armenian", "ia": "Interlingua", "id": "Indonesian", "ie": "Interlingue",
    "ilo": "Ilocano", "io": "Ido", "is": "Icelandic", "it": "Italian",
    "ja": "Japanese", "jbo": "Lojban", "jv": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "krc": "Karachay-Balkar", "ku": "Kurdish", "kv": "Komi", "kw": "Cornish",
    "ky": "Kyrgyz", "la": "Latin", "lb": "Luxembourgish", "lez": "Lezgian",
    "li": "Limburgish", "lmo": "Lombard", "lo": "Lao", "lrc": "Northern Luri",
    "lt": "Lithuanian", "lv": "Latvian", "mai": "Maithili", "mg": "Malagasy",
    "mhr": "Eastern Mari", "min": "Minangkabau", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "mrj": "Western Mari", "ms": "Malay",
    "mt": "Maltese", "mwl": "Mirandese", "my": "Burmese", "myv": "Erzya",
    "mzn": "Mazanderani", "nah": "Nahuatl", "nap": "Neapolitan", "nds": "Low German",
    "ne": "Nepali", "new": "Newar", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "or": "Odia", "os": "Ossetian",
    "pa": "Punjabi", "pam": "Pampanga", "pfl": "Palatine German", "pl": "Polish",
    "pms": "Piedmontese", "pnb": "Western Punjabi", "ps": "Pashto", "pt": "Portuguese",
    "qu": "Quechua", "rm": "Romansh", "ro": "Romanian", "ru": "Russian",
    "rue": "Rusyn", "sa": "Sanskrit", "sah": "Sakha", "sc": "Sardinian",
    "scn": "Sicilian", "sco": "Scots", "sd": "Sindhi", "sh": "Serbo-Croatian",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik",
    "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish",
    "tt": "Tatar", "tyv": "Tuvan", "ug": "Uyghur", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "vec": "Venetian", "vep": "Veps",
    "vi": "Vietnamese", "vls": "West Flemish", "vo": "Volapük", "wa": "Walloon",
    "war": "Waray", "wuu": "Wu Chinese", "xal": "Kalmyk", "xmf": "Mingrelian",
    "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}


class LanguageDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = False

    def load_model(self):
        """Load the fasttext language identification model. Downloads if not present."""
        model_path = config.MODEL_PATH

        if not os.path.exists(model_path):
            logger.info(f"Model not found at {model_path}. Downloading...")
            self._download_model(model_path)

        logger.info(f"Loading fasttext model from {model_path}...")
        self.model = fasttext.load_model(model_path)
        self.model_loaded = True
        logger.info("FastText language detection model loaded successfully.")

    def _download_model(self, model_path: str):
        """Download the fasttext lid.176.bin model."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info(f"Downloading model from {config.MODEL_URL}...")

        response = requests.get(config.MODEL_URL, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (10 * 1024 * 1024) < 8192:
                        logger.info(f"Download progress: {pct:.1f}%")

        logger.info(f"Model downloaded successfully to {model_path}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize input text for detection."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\n", " ")
        return text

    def detect(self, text: str, top_n: int = 5) -> list[dict]:
        """
        Detect the language of the given text.

        Returns a list of predictions, each with:
          - language_code
          - language_name
          - confidence
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        cleaned = self._preprocess_text(text)
        if not cleaned:
            return []

        labels, scores = self.model.predict(cleaned, k=top_n)

        predictions = []
        for label, score in zip(labels, scores):
            # fasttext labels are like "__label__en"
            lang_code = label.replace("__label__", "")
            predictions.append({
                "language_code": lang_code,
                "language_name": LANGUAGE_NAMES.get(lang_code, lang_code),
                "confidence": round(float(score), 4),
            })

        return predictions


# Singleton instance
detector = LanguageDetector()
