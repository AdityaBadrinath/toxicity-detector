import re
import string
from langdetect import detect, LangDetectException, DetectorFactory
import nltk
from nltk.corpus import stopwords

# Make language detection deterministic
DetectorFactory.seed = 0

# Download stopwords if not already installed
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# English Stopwords
ENGLISH_STOPWORDS = set(stopwords.words("english"))

# Hindi Stopwords (compact clean set)
HINDI_STOPWORDS = set([
    "अंदर", "अधिक", "अपना", "अपनी", "अपने", "अभी", "आगे", "आप",
    "इस", "इसके", "इसी", "उस", "उसके", "उसी", "उसे",
    "का", "के", "को", "कि", "क्या", "किसी", "कौन",
    "जब", "जहाँ", "जा", "तो", "था", "थी", "थे",
    "दिया", "नहीं", "बहुत", "भी", "में",
    "यदि", "यह", "यहाँ", "ये", "रहा", "रहे",
    "लिए", "वह", "वहाँ", "वे", "हैं", "हो",
    "होता", "होती", "होते", "होना", "हूँ", "है",
    "मैं", "मुझे", "तुम्हारा", "तुम्हारी",
    "उनका", "उनकी", "उनके", "कहाँ", "कैसे"
])


# ---------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------

def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions and hashtag symbol (keep hashtag word)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


# ---------------------------------------------------
# STOPWORD REMOVAL
# ---------------------------------------------------

def remove_stopwords(text):
    """Removes English and Hindi stopwords."""
    words = text.split()
    filtered_words = [
        word for word in words
        if word not in ENGLISH_STOPWORDS
        and word not in HINDI_STOPWORDS
    ]
    return " ".join(filtered_words)


# ---------------------------------------------------
# LANGUAGE DETECTION (STABLE VERSION)
# ---------------------------------------------------

def detect_language(text):
    """
    Detects language.
    Forces English for very short text to avoid random results.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "unknown"

    # Fix for short text (langdetect unreliable)
    if len(text.split()) < 3:
        return "EN"

    try:
        lang = detect(text)
        return lang.upper()
    except LangDetectException:
        return "EN"


# ---------------------------------------------------
# FULL PREPROCESSING PIPELINE
# ---------------------------------------------------

def preprocess_pipeline(text):
    """
    Complete preprocessing pipeline.
    Used mainly for TF-IDF baseline.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text