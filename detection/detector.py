import os
import re
import logging
import fasttext
import requests
from wordfreq import zipf_frequency

import config as settings

logger = logging.getLogger(__name__)


class LanguageDetector:

    # ==================================================
    # LANGUAGE NAME MAP (DEFINED LOCALLY)
    # ==================================================
    LANGUAGE_NAME_MAP = {
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

    def __init__(self):
        self.model = None
        self.supported_languages = 176

    # ==================================================
    # MODEL LOADING
    # ==================================================
    def load_model(self):
        if not os.path.exists(settings.MODEL_PATH):
            logger.info("Model not found. Downloading...")
            self._download_model()

        logger.info("Loading FastText model...")
        self.model = fasttext.load_model(settings.MODEL_PATH)
        logger.info("Model loaded successfully.")

    def _download_model(self):
        os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

        response = requests.get(settings.MODEL_URL, stream=True)
        response.raise_for_status()

        with open(settings.MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Model downloaded successfully.")

    # ==================================================
    # TEXT PREPROCESSING
    # ==================================================
    def _preprocess_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # ==================================================
    # ENGLISH RATIO CALCULATION
    # ==================================================
    def _english_word_ratio(self, text: str) -> float:

        all_words = text.split()

        if not all_words:
            return 0.0

        english_valid = 0

        for word in all_words:
            latin_word = re.sub(r"[^a-zA-Z']", "", word).lower()

            if latin_word:
                if zipf_frequency(latin_word, "en") > settings.ZIPF_FREQUENCY_THRESHOLD:
                    english_valid += 1

        ratio = english_valid / len(all_words)

        logger.info(
            f"English words: {english_valid}, "
            f"Total words: {len(all_words)}, "
            f"Ratio: {ratio:.3f}"
        )

        return ratio

    # ==================================================
    # MAIN DETECTION
    # ==================================================
    def detect(self, text: str, top_n: int = None):

        if not self.model:
            raise RuntimeError("Model not loaded.")

        top_n = top_n or settings.DEFAULT_TOP_N

        cleaned_text = self._preprocess_text(text)

        labels, scores = self.model.predict(cleaned_text, k=top_n)

        predictions = []

        for label, score in zip(labels, scores):
            language_code = label.replace("__label__", "")

            language_name = self.LANGUAGE_NAME_MAP.get(
                language_code,
                language_code
            )

            predictions.append({
                "language_code": language_code,
                "language_name": language_name,
                "confidence": float(score)
            })

        # ==================================================
        # HINGLISH OVERRIDE LOGIC
        # ==================================================
        if settings.HINGLISH_ENABLED and predictions:

            top_language = predictions[0]["language_code"]
            total_words = len(cleaned_text.split())

            if top_language == "en" and total_words >= settings.MIN_WORDS_FOR_CHECK:

                english_ratio = self._english_word_ratio(cleaned_text)

                if english_ratio < settings.ENGLISH_WORD_THRESHOLD:
                    logger.info("Overriding detection to Hinglish.")

                    predictions.insert(0, {
                        "language_code": "hinglish",
                        "language_name": "Hinglish",
                        "confidence": predictions[0]["confidence"]
                    })

        return predictions


# Singleton instance
detector = LanguageDetector()