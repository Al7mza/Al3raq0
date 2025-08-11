#!/usr/bin/env python3
"""
Telegram bot that:
- Admin can add reference images via /add (attached as photo or by replying /add to a photo).
- Any user can send a photo; the bot extracts yellow Chinese text using OpenCV + Cloud OCR,
  compares it with stored entries (FuzzyWuzzy), and returns the best matched reference image
  if similarity >= 80, otherwise replies that no match was found.

OCR backend:
- Cloud OCR (OCR.space). Requires an API key. This avoids heavy dependencies like PyTorch/Tesseract.

Dependencies:
- python-telegram-bot >= 20
- opencv-python (or opencv-python-headless)
- requests
- fuzzywuzzy (optional: python-Levenshtein for speed)

Run:
- Set BOT_TOKEN and ADMIN_ID below, then:  python3 bot.py
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import re
import jieba
from deep_translator import GoogleTranslator
from hanziconv import HanziConv
from fuzzywuzzy import fuzz
from telegram import Update
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram import InputMediaPhoto
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    filters,
)
import base64

# =====================
# Configuration
# =====================
# Place your bot token and admin user id here
BOT_TOKEN: str = "8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY"
# Replace with your numeric Telegram user ID
ADMIN_ID: int = 123456789

# Moderators: comma-separated IDs via env, includes ADMIN by default
MODERATOR_IDS: set = {ADMIN_ID}
_env_mods = os.environ.get("MODERATOR_IDS", "").strip()
if _env_mods:
    try:
        for part in _env_mods.split(','):
            part = part.strip()
            if part:
                MODERATOR_IDS.add(int(part))
    except Exception:
        pass

# Cloud OCR config (OCR.space)
OCR_API_KEY: str = os.environ.get("OCR_API_KEY", "K86394806188957")
OCR_API_ENDPOINT: str = "https://api.ocr.space/parse/image"
OCR_TIMEOUT_SECONDS: int = 30

# Preprocessing toggles
OCR_VARIANTS: List[str] = [
    "original",
    "gray_otsu",
    "clahe_v",
    "bilateral_otsu",
    "adaptive",
    "canny",
    "morph_close",
    "sobelx",
    "scharr",
]

# Yellow detection HSV bounds (tunable via env)
YELLOW_LOWER_H = int(os.environ.get("YELLOW_LOWER_H", "15"))
YELLOW_LOWER_S = int(os.environ.get("YELLOW_LOWER_S", "80"))
YELLOW_LOWER_V = int(os.environ.get("YELLOW_LOWER_V", "80"))
YELLOW_UPPER_H = int(os.environ.get("YELLOW_UPPER_H", "40"))
YELLOW_UPPER_S = int(os.environ.get("YELLOW_UPPER_S", "255"))
YELLOW_UPPER_V = int(os.environ.get("YELLOW_UPPER_V", "255"))
BOTTOM_RATIO = float(os.environ.get("BOTTOM_RATIO", "0.3"))  # bottom 30% band

# Red detection HSV ranges (wrap-around)
RED_LOWER1_H = int(os.environ.get("RED_LOWER1_H", "0"))
RED_LOWER1_S = int(os.environ.get("RED_LOER1_S", "80")) if os.environ.get("RED_LOER1_S") else int(os.environ.get("RED_LOWER1_S", "80"))
RED_LOWER1_V = int(os.environ.get("RED_LOWER1_V", "80"))
RED_UPPER1_H = int(os.environ.get("RED_UPPER1_H", "10"))
RED_UPPER1_S = int(os.environ.get("RED_UPPER1_S", "255"))
RED_UPPER1_V = int(os.environ.get("RED_UPPER1_V", "255"))
RED_LOWER2_H = int(os.environ.get("RED_LOWER2_H", "170"))
RED_LOWER2_S = int(os.environ.get("RED_LOWER2_S", "80"))
RED_LOWER2_V = int(os.environ.get("RED_LOWER2_V", "80"))
RED_UPPER2_H = int(os.environ.get("RED_UPPER2_H", "180"))
RED_UPPER2_S = int(os.environ.get("RED_UPPER2_S", "255"))
RED_UPPER2_V = int(os.environ.get("RED_UPPER2_V", "255"))

# Storage paths
IMAGES_DIR: str = "images"
DB_PATH: str = "database.json"
TEXT_REPLIES_DB_PATH: str = "text_replies.json"
QUERIES_DIR: str = "queries"

# Similarity thresholds and weights
SIMILARITY_THRESHOLD: int = int(os.environ.get("SIMILARITY_THRESHOLD", "80"))
WEIGHT_CHAR_RATIO: float = float(os.environ.get("WEIGHT_CHAR_RATIO", "0.35"))
WEIGHT_PARTIAL_RATIO: float = float(os.environ.get("WEIGHT_PARTIAL_RATIO", "0.25"))
WEIGHT_TOKEN_SET: float = float(os.environ.get("WEIGHT_TOKEN_SET", "0.25"))
WEIGHT_PARTIAL_TOKEN_SET: float = float(os.environ.get("WEIGHT_PARTIAL_TOKEN_SET", "0.1"))
WEIGHT_NGRAM_JACCARD: float = float(os.environ.get("WEIGHT_NGRAM_JACCARD", "0.05"))
NGRAM_N: int = int(os.environ.get("NGRAM_N", "2"))
MAX_OCR_CALLS: int = int(os.environ.get("MAX_OCR_CALLS", "24"))

# Overlap acceptance
MIN_TOKEN_OVERLAP: int = int(os.environ.get("MIN_TOKEN_OVERLAP", "1"))
MIN_SIMILARITY_FALLBACK: int = int(os.environ.get("MIN_SIMILARITY_FALLBACK", "70"))
MIN_LCS_ACCEPT: int = int(os.environ.get("MIN_LCS_ACCEPT", "2"))
CENTER_RATIO: float = float(os.environ.get("CENTER_RATIO", "0.5"))

# Translation settings
ENABLE_TRANSLATION: bool = os.environ.get("ENABLE_TRANSLATION", "1") == "1"
TRANSLATE_TARGET_LANG: str = os.environ.get("TRANSLATE_TARGET_LANG", "en")

# Baidu OCR settings
BAIDU_API_KEY: str = os.environ.get("BAIDU_OCR_API_KEY", "")
BAIDU_SECRET_KEY: str = os.environ.get("BAIDU_OCR_SECRET_KEY", "")
_baidu_token: Dict[str, Any] = {"token": None, "expires_at": 0.0}

# Google Vision OCR settings
GOOGLE_VISION_API_KEY: str = os.environ.get("GOOGLE_VISION_API_KEY", "AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k")

# Initialize logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =====================
# Utilities
# =====================

def ensure_storage() -> None:
    if not os.path.isdir(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)
    if not os.path.isdir(QUERIES_DIR):
        os.makedirs(QUERIES_DIR, exist_ok=True)
    if not os.path.isfile(DB_PATH):
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    if not os.path.isfile(TEXT_REPLIES_DB_PATH):
        with open(TEXT_REPLIES_DB_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_database() -> List[Dict[str, Any]]:
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_database(db: List[Dict[str, Any]]) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def load_text_replies() -> List[Dict[str, Any]]:
    try:
        with open(TEXT_REPLIES_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_text_replies(db: List[Dict[str, Any]]) -> None:
    with open(TEXT_REPLIES_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def to_simplified(text: str) -> str:
    try:
        return HanziConv.toSimplified(text)
    except Exception:
        return text


def normalize_text(text: str) -> str:
    """Convert to simplified Chinese, lowercase, keep CJK, letters, digits."""
    text = to_simplified(text)
    text = text.lower()
    normalized_chars: List[str] = []
    for ch in text:
        if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum():
            normalized_chars.append(ch)
    return "".join(normalized_chars)


def tokenize_chinese(text: str) -> List[str]:
    # Use jieba to tokenize; keep tokens that are CJK or alphanumeric
    tokens = jieba.lcut(text)
    filtered: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in tok) or any(c.isalnum() for c in tok):
            filtered.append(tok)
    return filtered


def tokens_to_string(tokens: List[str]) -> str:
    # Join tokens with spaces for token-based fuzzy matching
    return " ".join(tokens)

def get_tail_tokens(text: str, k: int = 2) -> List[str]:
    """Return the last k meaningful tokens from Chinese/alpha-numeric text."""
    if not text:
        return []
    simple = to_simplified(text)
    toks = [t.strip() for t in jieba.lcut(simple) if t and (any("\u4e00" <= ch <= "\u9fff" for ch in t) or any(c.isalnum() for c in t))]
    if not toks:
        # fallback to last k CJK sequences
        cjk_seq = re.findall(r"[\u4e00-\u9fff]+", simple)
        toks = cjk_seq
    return toks[-k:] if len(toks) >= k else toks


def ngram_set(text: str, n: int = 2) -> set:
    if n <= 0:
        n = 2
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def longest_common_substring_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    # DP over shorter dimension to control memory if needed
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        curr = [0]
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                val = prev[j - 1] + 1
            else:
                val = 0
            curr.append(val)
            if val > best:
                best = val
        prev = curr
    return best


def build_token_set(text: str) -> set:
    """Build a token set combining jieba tokens and character n-grams of normalized text."""
    simple = to_simplified(text)
    norm = normalize_text(simple)
    words = set(tokenize_chinese(simple))
    grams = ngram_set(norm, NGRAM_N)
    return {t for t in (words | grams) if t}


def yellow_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([YELLOW_LOWER_H, YELLOW_LOWER_S, YELLOW_LOWER_V])
    upper = np.array([YELLOW_UPPER_H, YELLOW_UPPER_S, YELLOW_UPPER_V])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def red_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Build a mask for red regions using two HSV ranges (wrap-around)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([RED_LOWER1_H, RED_LOWER1_S, RED_LOWER1_V])
    upper1 = np.array([RED_UPPER1_H, RED_UPPER1_S, RED_UPPER1_V])
    lower2 = np.array([RED_LOWER2_H, RED_LOWER2_S, RED_LOWER2_V])
    upper2 = np.array([RED_UPPER2_H, RED_UPPER2_S, RED_UPPER2_V])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def find_rois_from_mask(image_bgr: np.ndarray, mask: np.ndarray, min_area: int = 200, max_count: int = 10) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: List[np.ndarray] = []
    h, w = mask.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area:
            continue
        pad = max(2, int(0.05 * max(cw, ch)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        roi = image_bgr[y0:y1, x0:x1]
        if roi.size > 0:
            rois.append(roi)
    rois.sort(key=lambda r: r.shape[0] * r.shape[1], reverse=True)
    return rois[:max_count]


def to_thresh(image_bgr_or_gray: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY) if len(image_bgr_or_gray.shape) == 3 else image_bgr_or_gray
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th


def preprocess_variants_full(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessing variants of the full image for OCR."""
    variants: List[np.ndarray] = []

    if "original" in OCR_VARIANTS:
        variants.append(image_bgr)

    if "gray_otsu" in OCR_VARIANTS:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th)
        # inverted
        variants.append(cv2.bitwise_not(th))
        # 90-degree rotations on thresholded image
        variants.append(cv2.rotate(th, cv2.ROTATE_90_CLOCKWISE))
        variants.append(cv2.rotate(th, cv2.ROTATE_90_COUNTERCLOCKWISE))
        # upscale thresholded image for small text
        th_up = cv2.resize(th, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        variants.append(th_up)

    if "clahe_v" in OCR_VARIANTS:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v2 = clahe.apply(v)
        hsv2 = cv2.merge([h, s, v2])
        img2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, th2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th2)

    if "bilateral_otsu" in OCR_VARIANTS:
        bf = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        gray3 = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
        _, th3 = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th3)

    if "adaptive" in OCR_VARIANTS:
        gray4 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        th4 = cv2.adaptiveThreshold(
            gray4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
        variants.append(th4)

    if "canny" in OCR_VARIANTS:
        gray5 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray5, 80, 160)
        # thicken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        variants.append(edges)

    if "morph_close" in OCR_VARIANTS:
        gray6 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, th6 = cv2.threshold(gray6, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(th6, cv2.MORPH_CLOSE, kernel, iterations=1)
        variants.append(closed)

    if "sobelx" in OCR_VARIANTS:
        gray7 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray7, cv2.CV_16S, 1, 0)
        sx = cv2.convertScaleAbs(sx)
        _, th7 = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th7)

    if "scharr" in OCR_VARIANTS:
        gray8 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sch = cv2.Scharr(gray8, cv2.CV_16S, 1, 0)
        sch = cv2.convertScaleAbs(sch)
        _, th8 = cv2.threshold(sch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th8)

    return variants

def scan_tiles(image_bgr: np.ndarray, rows: int = 2, cols: int = 2) -> List[np.ndarray]:
    """Split image into tiles and produce simple thresholded variants to improve recall."""
    h, w = image_bgr.shape[:2]
    tile_h = max(1, h // rows)
    tile_w = max(1, w // cols)
    variants: List[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            y0 = r * tile_h
            x0 = c * tile_w
            y1 = h if r == rows - 1 else (r + 1) * tile_h
            x1 = w if c == cols - 1 else (c + 1) * tile_w
            tile = image_bgr[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            variants.append(th)
    return variants


def ocr_with_cloud(image_np: np.ndarray) -> str:
    """Try Google Vision first, then Baidu, then OCR.space (engine 2->1)."""
    success, encoded = cv2.imencode(
        ".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not success:
        return ""

    # 1) Google Vision OCR
    gcv_text = ""
    if GOOGLE_VISION_API_KEY:
        gcv_text = ocr_with_google_bytes(encoded.tobytes())
        if count_cjk(gcv_text) >= 2:
            return gcv_text

    # 2) Baidu OCR
    baidu_text = ""
    if BAIDU_API_KEY and BAIDU_SECRET_KEY:
        baidu_text = ocr_with_baidu_bytes(encoded.tobytes())
        if count_cjk(baidu_text) >= 2:
            return baidu_text

    # 3) OCR.space
    headers = {"apikey": OCR_API_KEY}
    files = {"file": ("image.jpg", encoded.tobytes(), "image/jpeg")}

    def call_engine(engine: int) -> str:
        data = {
            "language": "chs+eng",
            "isOverlayRequired": "false",
            "OCREngine": str(engine),
            "scale": "true",
            "detectOrientation": "true",
        }
        try:
            resp = requests.post(
                OCR_API_ENDPOINT,
                headers=headers,
                data=data,
                files=files,
                timeout=OCR_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            logger.error("Cloud OCR request failed (engine %s): %s", engine, e)
            return ""

        try:
            if payload.get("IsErroredOnProcessing"):
                logger.warning("OCR error (engine %s): %s", engine, payload.get("ErrorMessage"))
                return ""
            results = payload.get("ParsedResults", [])
            texts: List[str] = []
            for r in results:
                t = r.get("ParsedText", "")
                if t:
                    texts.append(t)
            return " ".join(s.strip() for s in texts if s.strip())
        except Exception as e:
            logger.error("Cloud OCR parse failed (engine %s): %s", engine, e)
            return ""

    text2 = call_engine(2)
    if count_cjk(text2) >= 2:
        return text2
    text1 = call_engine(1) if not text2 else text2
    return text1

# Google Vision OCR via REST API
def ocr_with_google_bytes(jpeg_bytes: bytes) -> str:
    try:
        img_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        req = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {
                        "languageHints": ["zh", "zh-Hans", "zh-Hant", "en"]
                    },
                }
            ]
        }
        resp = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
            json=req,
            timeout=OCR_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        anns = (data.get("responses") or [{}])[0]
        text = ""
        if "fullTextAnnotation" in anns and anns["fullTextAnnotation"].get("text"):
            text = anns["fullTextAnnotation"]["text"]
        elif "textAnnotations" in anns and anns["textAnnotations"]:
            text = anns["textAnnotations"][0].get("description", "")
        return text.strip()
    except Exception as e:
        logger.warning("Google Vision OCR error: %s", e)
        return ""

def baidu_get_token() -> Optional[str]:
    now = time.time()
    if _baidu_token["token"] and now < _baidu_token["expires_at"]:
        return _baidu_token["token"]
    try:
        resp = requests.get(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": BAIDU_API_KEY,
                "client_secret": BAIDU_SECRET_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        expires_in = float(data.get("expires_in", 3600))
        if token:
            _baidu_token["token"] = token
            _baidu_token["expires_at"] = now + max(600.0, expires_in - 120.0)
            return token
    except Exception as e:
        logger.warning("Baidu token fetch failed: %s", e)
    return None

def ocr_with_baidu_bytes(jpeg_bytes: bytes) -> str:
    token = baidu_get_token()
    if not token:
        return ""
    img_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    # Try accurate_basic first, then general_basic
    for endpoint in [
        "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
        "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",
    ]:
        try:
            resp = requests.post(
                f"{endpoint}?access_token={token}",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "image": img_b64,
                    "language_type": "CHN_ENG",
                    "detect_direction": "true",
                    "probability": "true",
                },
                timeout=OCR_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            if "words_result" in data:
                words = [w.get("words", "") for w in data.get("words_result", [])]
                text = " ".join(w.strip() for w in words if w.strip())
                if text:
                    return text
        except Exception as e:
            logger.warning("Baidu OCR error: %s", e)
            continue
    return ""


def extract_text_from_image(image_path: str) -> str:
    """Extract Chinese text by running OCR on the full image with multiple preprocessing variants."""
    image = cv2.imread(image_path)
    if image is None:
        return ""

    candidates: List[str] = []
    calls = 0
    for variant in preprocess_variants_full(image):
        if calls >= MAX_OCR_CALLS:
            break
        text = ocr_with_cloud(variant)
        calls += 1
        if text:
            candidates.append(text)
            if count_cjk(text) >= 3:
                break

    # If still weak, scan tiles to capture smaller/localized text
    if calls < MAX_OCR_CALLS and (not candidates or count_cjk(max(candidates, key=lambda s: count_cjk(s))) < 2):
        for tvar in scan_tiles(image, rows=2, cols=2):
            if calls >= MAX_OCR_CALLS:
                break
            ttext = ocr_with_cloud(tvar)
            calls += 1
            if ttext:
                candidates.append(ttext)

    if not candidates:
        return ""

    best = max(candidates, key=lambda s: (count_cjk(s), len(s)))
    return best

def extract_center_text(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        return ""
    h, w = image.shape[:2]
    ch = max(1, int(h * CENTER_RATIO))
    cw = max(1, int(w * CENTER_RATIO))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    center = image[y0:y0 + ch, x0:x0 + cw]
    candidates: List[str] = []
    calls = 0
    for variant in preprocess_variants_full(center):
        if calls >= MAX_OCR_CALLS:
            break
        text = ocr_with_cloud(variant)
        calls += 1
        if text:
            candidates.append(text)
            if count_cjk(text) >= 3:
                break
    if not candidates:
        return ""
    return max(candidates, key=lambda s: (count_cjk(s), len(s)))


def extract_yellow_text(image_path: str) -> str:
    """Extract Chinese text focusing on yellow content across the whole image (ROIs + masked)."""
    image = cv2.imread(image_path)
    if image is None:
        return ""
    mask = yellow_mask_bgr(image)
    candidates: List[str] = []
    calls = 0
    # ROI-based
    for roi in find_rois_from_mask(image, mask):
        if calls >= MAX_OCR_CALLS:
            break
        t = ocr_with_cloud(to_thresh(roi))
        calls += 1
        if not t and calls < MAX_OCR_CALLS:
            t = ocr_with_cloud(roi)
            calls += 1
        if t:
            candidates.append(t)
            if count_cjk(t) >= 3:
                break
    # Full masked
    masked = cv2.bitwise_and(image, image, mask=mask)
    t_mask = None
    if calls < MAX_OCR_CALLS:
        t_mask = ocr_with_cloud(to_thresh(masked))
        calls += 1
    if not t_mask and calls < MAX_OCR_CALLS:
        t_mask = ocr_with_cloud(masked)
        calls += 1
    if t_mask:
        candidates.append(t_mask)
        if count_cjk(t_mask) < 3 and calls < MAX_OCR_CALLS:
            # Try rotated masked threshold
            masked_th = to_thresh(masked)
            rot1 = cv2.rotate(masked_th, cv2.ROTATE_90_CLOCKWISE)
            rot2 = cv2.rotate(masked_th, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for rot in (rot1, rot2):
                if calls >= MAX_OCR_CALLS:
                    break
                tr = ocr_with_cloud(rot)
                calls += 1
                if tr:
                    candidates.append(tr)
                    if count_cjk(tr) >= 3:
                        break

    if not candidates:
        return ""
    return max(candidates, key=lambda s: (count_cjk(s), len(s)))

def extract_red_text(image_path: str) -> str:
    """Extract Chinese text focusing on red content across the whole image (ROIs + masked)."""
    image = cv2.imread(image_path)
    if image is None:
        return ""
    mask = red_mask_bgr(image)
    candidates: List[str] = []
    calls = 0
    # ROI-based
    for roi in find_rois_from_mask(image, mask):
        if calls >= MAX_OCR_CALLS:
            break
        t = ocr_with_cloud(to_thresh(roi))
        calls += 1
        if not t and calls < MAX_OCR_CALLS:
            t = ocr_with_cloud(roi)
            calls += 1
        if t:
            candidates.append(t)
            if count_cjk(t) >= 3:
                break
    # Full masked
    masked = cv2.bitwise_and(image, image, mask=mask)
    t_mask = None
    if calls < MAX_OCR_CALLS:
        t_mask = ocr_with_cloud(to_thresh(masked))
        calls += 1
    if not t_mask and calls < MAX_OCR_CALLS:
        t_mask = ocr_with_cloud(masked)
        calls += 1
    if t_mask:
        candidates.append(t_mask)
        if count_cjk(t_mask) < 3 and calls < MAX_OCR_CALLS:
            masked_th = to_thresh(masked)
            rot1 = cv2.rotate(masked_th, cv2.ROTATE_90_CLOCKWISE)
            rot2 = cv2.rotate(masked_th, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for rot in (rot1, rot2):
                if calls >= MAX_OCR_CALLS:
                    break
                tr = ocr_with_cloud(rot)
                calls += 1
                if tr:
                    candidates.append(tr)
                    if count_cjk(tr) >= 3:
                        break

    if not candidates:
        return ""
    return max(candidates, key=lambda s: (count_cjk(s), len(s)))


def extract_bottom_yellow_text(image_path: str) -> str:
    """Extract Chinese text from bottom band focusing on yellow (admin references often at bottom)."""
    image = cv2.imread(image_path)
    if image is None:
        return ""
    h, w = image.shape[:2]
    band_top = int(h * (1.0 - max(0.05, min(0.9, BOTTOM_RATIO))))
    band = image[band_top:h, 0:w]
    # Reuse yellow pipeline on the band
    tmp_file = None
    try:
        # Avoid writing; process directly
        mask = yellow_mask_bgr(band)
        candidates: List[str] = []
        for roi in find_rois_from_mask(band, mask):
            t = ocr_with_cloud(to_thresh(roi)) or ocr_with_cloud(roi)
            if t:
                candidates.append(t)
        masked = cv2.bitwise_and(band, band, mask=mask)
        t_band = ocr_with_cloud(to_thresh(masked)) or ocr_with_cloud(masked)
        if t_band:
            candidates.append(t_band)
        if not candidates:
            return ""
        def cjk_count(s: str) -> int:
            return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
        return max(candidates, key=lambda s: (cjk_count(s), len(s)))
    finally:
        pass


def extract_text_for_add(image_path: str) -> str:
    """Admin add: prefer red-text OCR, fallback to robust full-image OCR."""
    text_red = extract_red_text(image_path)
    if text_red:
        return text_red
    return extract_text_from_image(image_path)


def extract_text_for_query(image_path: str) -> str:
    """User query: focus center first, then full-image OCR (color-agnostic)."""
    text = extract_center_text(image_path)
    if text:
        return text
    return extract_text_from_image(image_path)


def compute_similarity(query_text: str, candidate_text: str) -> int:
    """Compute a weighted similarity score (0-100) combining multiple fuzzy metrics and n-gram Jaccard."""
    q_simpl = to_simplified(query_text)
    c_simpl = to_simplified(candidate_text)

    q_norm = normalize_text(q_simpl)
    c_norm = normalize_text(c_simpl)

    # Character-level metrics
    char_ratio = fuzz.ratio(q_norm, c_norm)
    char_partial = fuzz.partial_ratio(q_norm, c_norm)

    # Token-level metrics
    q_tokens = tokens_to_string(tokenize_chinese(q_simpl))
    c_tokens = tokens_to_string(tokenize_chinese(c_simpl))
    token_set = fuzz.token_set_ratio(q_tokens, c_tokens)
    token_partial_set = fuzz.partial_token_set_ratio(q_tokens, c_tokens)

    # N-gram Jaccard over normalized string
    q_ngrams = ngram_set(q_norm, NGRAM_N)
    c_ngrams = ngram_set(c_norm, NGRAM_N)
    jac = jaccard(q_ngrams, c_ngrams)
    jac_score = int(round(jac * 100))

    # Weighted score
    weighted = (
        WEIGHT_CHAR_RATIO * char_ratio
        + WEIGHT_PARTIAL_RATIO * char_partial
        + WEIGHT_TOKEN_SET * token_set
        + WEIGHT_PARTIAL_TOKEN_SET * token_partial_set
        + WEIGHT_NGRAM_JACCARD * jac_score
    )

    # Clamp and return as int
    score = int(round(min(100.0, max(0.0, weighted))))
    return score

# Translation cache to reduce repeated calls
_translate_cache: Dict[Tuple[str, str], str] = {}

def translate_text(text: str, target_lang: str = TRANSLATE_TARGET_LANG) -> str:
    if not ENABLE_TRANSLATION or not text:
        return ""
    key = (text, target_lang)
    if key in _translate_cache:
        return _translate_cache[key]
    try:
        tr = GoogleTranslator(source='auto', target=target_lang).translate(text)
        if tr:
            _translate_cache[key] = tr
            return tr
    except Exception:
        return ""
    return ""


def add_entry_to_db(image_path: str, text: str) -> None:
    db = load_database()
    norm_text = normalize_text(text)
    tokens = sorted(list(build_token_set(text))) if text else []
    tail_tokens = get_tail_tokens(text, k=2)
    entry = {
        "image_path": image_path,
        "text": text,
        "norm_text": norm_text,
        "tokens": tokens,
        "tail_tokens": tail_tokens,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    db.append(entry)
    save_database(db)


def add_text_reply_to_db(trigger_text: str, reply_image_path: str) -> None:
    db = load_text_replies()
    norm_text = normalize_text(trigger_text)
    tokens = sorted(list(build_token_set(trigger_text))) if trigger_text else []
    tail_tokens = get_tail_tokens(trigger_text, k=2)
    entry = {
        "text": trigger_text,
        "norm_text": norm_text,
        "tokens": tokens,
        "tail_tokens": tail_tokens,
        "reply_image_path": reply_image_path,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    db.append(entry)
    save_text_replies(db)


def find_best_match(query_text: str) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Return (best_entry, best_similarity_score, best_token_overlap).
    Accept if token overlap >= MIN_TOKEN_OVERLAP or similarity >= SIMILARITY_THRESHOLD.
    """
    db = load_database()
    if not db:
        return None, 0, 0

    query_tokens = build_token_set(query_text)
    query_norm = normalize_text(query_text)

    best_entry: Optional[Dict[str, Any]] = None
    best_sim_score: int = -1
    best_tok_overlap: int = -1
    best_overall_score: float = -1.0
    best_lcs_len: int = 0

    for entry in db:
        candidate_text = entry.get("text", "") or ""
        candidate_norm = entry.get("norm_text") or normalize_text(candidate_text)
        entry_tokens = set(entry.get("tokens") or build_token_set(candidate_text))
        entry_tail = set(entry.get("tail_tokens") or get_tail_tokens(candidate_text, k=2))

        # token overlap
        overlap = len(query_tokens & entry_tokens)

        # weighted similarity
        sim_score = compute_similarity(query_text, candidate_text)

        # LCS on normalized strings
        lcs_len = longest_common_substring_length(query_norm, candidate_norm)

        # Translation-assisted similarity (English)
        qe = translate_text(query_text)
        ce = translate_text(candidate_text)
        sim_en = compute_similarity(qe, ce) if qe and ce else 0

        # overall ranking score prioritizing overlap then similarity
        # Tail match bonus if query tail matches entry tail
        query_tail = set(get_tail_tokens(query_text, k=2))
        tail_overlap = len(query_tail & entry_tail)
        overall = tail_overlap * 2000 + overlap * 300 + lcs_len * 80 + max(sim_score, sim_en)

        if overall > best_overall_score:
            best_overall_score = overall
            best_entry = entry
            best_sim_score = max(sim_score, sim_en)
            best_tok_overlap = overlap
            best_lcs_len = lcs_len

    # Stash best lcs in sim score lower bits? Keep interface: return sim and overlap
    return best_entry, best_sim_score if best_entry else 0, max(0, best_tok_overlap)


def find_best_text_reply(query_text: str) -> Tuple[Optional[Dict[str, Any]], int, int]:
    """Search text_replies for the best match. Return (entry, sim_score, token_overlap)."""
    db = load_text_replies()
    if not db:
        return None, 0, 0
    query_tokens = build_token_set(query_text)

    best_entry: Optional[Dict[str, Any]] = None
    best_sim_score: int = -1
    best_tok_overlap: int = -1
    best_overall_score: float = -1.0

    for entry in db:
        candidate_text = entry.get("text", "") or ""
        entry_tokens = set(entry.get("tokens") or build_token_set(candidate_text))
        entry_tail = set(entry.get("tail_tokens") or get_tail_tokens(candidate_text, k=2))
        overlap = len(query_tokens & entry_tokens)
        sim_score = compute_similarity(query_text, candidate_text)
        # approximate LCS on normalized strings
        candidate_norm = normalize_text(candidate_text)
        lcs_len = longest_common_substring_length(normalize_text(query_text), candidate_norm)
        qe = translate_text(query_text)
        ce = translate_text(candidate_text)
        sim_en = compute_similarity(qe, ce) if qe and ce else 0
        query_tail = set(get_tail_tokens(query_text, k=2))
        tail_overlap = len(query_tail & entry_tail)
        overall = tail_overlap * 2000 + overlap * 300 + lcs_len * 80 + max(sim_score, sim_en)
        if overall > best_overall_score:
            best_overall_score = overall
            best_entry = entry
            best_sim_score = max(sim_score, sim_en)
            best_tok_overlap = overlap
    return best_entry, best_sim_score if best_entry else 0, max(0, best_tok_overlap)


async def download_photo_as_file(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    dest_dir: str,
    photo_file_id: str,
    fallback_suffix: str = "",
) -> Optional[str]:
    """Download a Telegram photo file_id to dest_dir, return local file path."""
    try:
        telegram_file = await context.bot.get_file(photo_file_id)
        base_name = f"{int(time.time())}_{photo_file_id[:16]}{fallback_suffix}.jpg"
        local_path = os.path.join(dest_dir, base_name)
        await telegram_file.download_to_drive(custom_path=local_path)
        return local_path
    except Exception as e:
        logger.exception("Failed to download photo: %s", e)
        return None


def is_admin(user_id: Optional[int]) -> bool:
    return user_id is not None and int(user_id) == int(ADMIN_ID)


def is_moderator(user_id: Optional[int]) -> bool:
    return user_id is not None and int(user_id) in MODERATOR_IDS


# =====================
# Handlers
# =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a photo with Chinese text. Admins can add reference images using /add or $+."
    )


# ============== Moderator UI (Add Text → Reply Image) ==============
MANAGE_ADD_TEXT, MANAGE_WAIT_TEXT, MANAGE_WAIT_PHOTO = range(3)

async def manage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if not is_moderator(user.id):
        await update.message.reply_text("Only moderators can use this.")
        return ConversationHandler.END
    kb = [
        [InlineKeyboardButton(text="Add Text Reply", callback_data="add_text_reply")],
        [InlineKeyboardButton(text="Cancel", callback_data="cancel_manage")],
    ]
    await update.message.reply_text(
        "Moderator panel:", reply_markup=InlineKeyboardMarkup(kb)
    )
    return MANAGE_ADD_TEXT

async def manage_cb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    user = update.effective_user
    await query.answer()
    if not is_moderator(user.id):
        await query.edit_message_text("Only moderators can use this.")
        return ConversationHandler.END
    if query.data == "add_text_reply":
        await query.edit_message_text("Send the Chinese trigger text to match (yellow or any text from customer images).")
        return MANAGE_WAIT_TEXT
    if query.data == "cancel_manage":
        await query.edit_message_text("Cancelled.")
        return ConversationHandler.END
    return ConversationHandler.END

async def manage_capture_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if not is_moderator(user.id):
        await update.message.reply_text("Only moderators can use this.")
        return ConversationHandler.END
    trigger_text = (update.message.text or "").strip()
    if not trigger_text:
        await update.message.reply_text("Please send non-empty text.")
        return MANAGE_WAIT_TEXT
    context.user_data["trigger_text"] = trigger_text
    await update.message.reply_text("Now send the reply image (as photo or image document).")
    return MANAGE_WAIT_PHOTO

async def manage_capture_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if not is_moderator(user.id):
        await update.message.reply_text("Only moderators can use this.")
        return ConversationHandler.END
    trigger_text = context.user_data.get("trigger_text", "").strip()
    if not trigger_text:
        await update.message.reply_text("No trigger text found. Start again with /manage.")
        return ConversationHandler.END

    local_path: Optional[str] = None
    if update.message.photo:
        largest = update.message.photo[-1]
        local_path = await download_photo_as_file(update, context, IMAGES_DIR, largest.file_id, fallback_suffix="_reply")
    elif update.message.document:
        doc = update.message.document
        if (doc.mime_type or "").lower().startswith("image/"):
            try:
                telegram_file = await context.bot.get_file(doc.file_id)
                local_path = os.path.join(IMAGES_DIR, f"{int(time.time())}_{doc.file_unique_id}_reply")
                ext = ".jpg"
                mime = (doc.mime_type or "").lower()
                if "/png" in mime:
                    ext = ".png"
                elif "/jpeg" in mime or "/jpg" in mime:
                    ext = ".jpg"
                local_path = local_path + ext
                await telegram_file.download_to_drive(custom_path=local_path)
            except Exception as e:
                logger.exception("Failed to download reply document: %s", e)
                local_path = None
    if not local_path:
        await update.message.reply_text("Please send a valid image.")
        return MANAGE_WAIT_PHOTO

    add_text_reply_to_db(trigger_text, local_path)
    await update.message.reply_text("Saved text reply mapping.")
    context.user_data.pop("trigger_text", None)
    return ConversationHandler.END

async def manage_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.pop("trigger_text", None)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END

# Shortcut command to jump straight into adding mapping
async def addmapping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_moderator(user.id):
        await update.message.reply_text("Only moderators can use this.")
        return
    kb = [[InlineKeyboardButton(text="Add Text Reply", callback_data="add_text_reply")]]
    await update.message.reply_text("Click to add a text→image mapping:", reply_markup=InlineKeyboardMarkup(kb))


async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Admin-only. Usage options:
    - Reply to a photo with /add
    - Send a photo with caption '/add'
    """
    user = update.effective_user
    if not is_admin(user.id):
        await update.message.reply_text("Only the administrator can use /add.")
        return

    message = update.effective_message

    # Case 1: Invoked as a reply to a photo
    if message.reply_to_message and message.reply_to_message.photo:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        largest = message.reply_to_message.photo[-1]
        local_path = await download_photo_as_file(
            update, context, IMAGES_DIR, largest.file_id, fallback_suffix="_add"
        )
        if not local_path:
            await message.reply_text("Failed to download the photo.")
            return
        try:
            text = extract_text_for_add(local_path)
        except Exception as e:
            logger.exception("Add OCR failed: %s", e)
            text = ""
        add_entry_to_db(local_path, text)
        await message.reply_text("Added image to database.")
        return

    # Case 2: If /add was sent standalone, instruct admin how to use it
    await message.reply_text(
        "Please either: 1) Send a photo with caption '/add', or 2) Reply to a photo with /add."
    )


async def photo_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Route photo messages:
    - If admin and caption starts with /add -> treat as add
    - Else -> treat as user search
    """
    message = update.effective_message
    user = update.effective_user
    if not message or not message.photo:
        return

    caption = (message.caption or "").strip()
    caption_lower = caption.lower()

    if caption_lower == "$+" or caption_lower.startswith("/add"):
        if not is_admin(user.id):
            await message.reply_text("Only the administrator can use /add.")
            return
        # Admin add flow with caption
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        largest = message.photo[-1]
        local_path = await download_photo_as_file(
            update, context, IMAGES_DIR, largest.file_id, fallback_suffix="_add"
        )
        if not local_path:
            await message.reply_text("Failed to download the photo.")
            return
        try:
            text = extract_text_for_add(local_path)
        except Exception as e:
            logger.exception("Add OCR failed: %s", e)
            text = ""
        add_entry_to_db(local_path, text)
        await message.reply_text("Added image to database.")
        return

    # User query flow
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    largest = message.photo[-1]
    tmp_path = await download_photo_as_file(update, context, QUERIES_DIR, largest.file_id, fallback_suffix="_query")
    if not tmp_path:
        await message.reply_text("Failed to download the photo.")
        return

    try:
        # Extract yellow (gold) and full
        query_yellow = extract_yellow_text(tmp_path)
        query_full = extract_text_for_query(tmp_path)
        # Prefer yellow tail for matching, but pass full text into find_best_* which computes tails too
        query_text = query_yellow if query_yellow else query_full
    except Exception as e:
        logger.exception("Query OCR failed: %s", e)
        await message.reply_text("Sorry, OCR failed. Please try another image or later.")
        return
    # First try text-reply mappings
    best_reply_entry, reply_score, reply_overlap = find_best_text_reply(query_text)
    if best_reply_entry and (reply_overlap >= MIN_TOKEN_OVERLAP or reply_score >= SIMILARITY_THRESHOLD):
        reply_path = best_reply_entry.get("reply_image_path")
        if reply_path and os.path.isfile(reply_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                media = []
                media.append(InputMediaPhoto(open(reply_path, "rb"), caption=f"Reply {'overlap' if reply_overlap>=MIN_TOKEN_OVERLAP else str(reply_score)+'%'}"))
                media.append(InputMediaPhoto(open(tmp_path, "rb")))
                await update.message.reply_media_group(media=media)
            except Exception as e:
                logger.exception("Failed to send reply image: %s", e)
        else:
            logger.warning("Reply image missing at path: %s", reply_path)
            await update.message.reply_text("Sorry, reply image missing.")
        return
    # Else fallback to image DB
    best_entry, best_score, best_overlap = find_best_match(query_text)

    logger.info("Query text: %s | Best score: %s | Overlap: %s", query_text, best_score, best_overlap)

    # Accept if center/full text shares 2+ char substring, or token overlap, or similarity thresholds
    lcs_len_best = longest_common_substring_length(normalize_text(query_text), normalize_text(best_entry.get("text", "") if best_entry else "")) if best_entry else 0
    if best_entry and (lcs_len_best >= MIN_LCS_ACCEPT or best_overlap >= MIN_TOKEN_OVERLAP or best_score >= SIMILARITY_THRESHOLD or (best_score >= MIN_SIMILARITY_FALLBACK and best_overlap >= 0)):
        image_path = best_entry.get("image_path")
        if image_path and os.path.isfile(image_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                media = []
                media.append(InputMediaPhoto(open(image_path, "rb"), caption=f"Match {'overlap' if best_overlap>=MIN_TOKEN_OVERLAP else str(best_score)+'%'}"))
                media.append(InputMediaPhoto(open(tmp_path, "rb")))
                await update.message.reply_media_group(media=media)
            except Exception as e:
                logger.exception("Failed to send matched image: %s", e)
                await update.message.reply_text("Sorry, failed to send the matched image.")
        else:
            await update.message.reply_text("Matched entry found but image file is missing.")
    else:
        await update.message.reply_text("Sorry, no match found.")


async def document_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle images sent as documents (e.g., PNG/JPEG as files)."""
    message = update.effective_message
    doc = message.document
    if not doc:
        return

    mime = (doc.mime_type or "").lower()
    if not (mime.startswith("image/")):
        return

    user = update.effective_user
    caption = (message.caption or "").strip()
    caption_lower = caption.lower()

    if caption_lower == "$+" or caption_lower.startswith("/add"):
        if not is_admin(user.id):
            await message.reply_text("Only the administrator can use /add.")
            return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
        try:
            telegram_file = await context.bot.get_file(doc.file_id)
            local_path = os.path.join(IMAGES_DIR, f"{int(time.time())}_{doc.file_unique_id}_add")
            # Try to infer extension from mime
            ext = ".jpg"
            if "/png" in mime:
                ext = ".png"
            elif "/jpeg" in mime or "/jpg" in mime:
                ext = ".jpg"
            local_path = local_path + ext
            await telegram_file.download_to_drive(custom_path=local_path)
        except Exception as e:
            logger.exception("Failed to download document: %s", e)
            await message.reply_text("Failed to download the image document.")
            return

        try:
            text = extract_text_for_add(local_path)
        except Exception as e:
            logger.exception("Add OCR failed: %s", e)
            text = ""
        add_entry_to_db(local_path, text)
        await message.reply_text("Added image to database.")
        return

    # User query flow for document image
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        telegram_file = await context.bot.get_file(doc.file_id)
        local_path = os.path.join(QUERIES_DIR, f"{int(time.time())}_{doc.file_unique_id}_query")
        ext = ".jpg"
        mime = (doc.mime_type or "").lower()
        if "/png" in mime:
            ext = ".png"
        elif "/jpeg" in mime or "/jpg" in mime:
            ext = ".jpg"
        local_path = local_path + ext
        await telegram_file.download_to_drive(custom_path=local_path)
    except Exception as e:
        logger.exception("Failed to download document: %s", e)
        await message.reply_text("Failed to download the image document.")
        return

    try:
        query_yellow = extract_yellow_text(local_path)
        query_full = extract_text_for_query(local_path)
        query_text = query_yellow if query_yellow else query_full
    except Exception as e:
        logger.exception("Query OCR failed: %s", e)
        await message.reply_text("Sorry, OCR failed. Please try another image or later.")
        return
    best_reply_entry, reply_score, reply_overlap = find_best_text_reply(query_text)
    if best_reply_entry and (reply_overlap >= MIN_TOKEN_OVERLAP or reply_score >= SIMILARITY_THRESHOLD):
        reply_path = best_reply_entry.get("reply_image_path")
        if reply_path and os.path.isfile(reply_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                media = []
                media.append(InputMediaPhoto(open(reply_path, "rb"), caption=f"Reply {'overlap' if reply_overlap>=MIN_TOKEN_OVERLAP else str(reply_score)+'%'}"))
                media.append(InputMediaPhoto(open(local_path, "rb")))
                await update.message.reply_media_group(media=media)
            except Exception as e:
                logger.exception("Failed to send reply image: %s", e)
        else:
            logger.warning("Reply image missing at path: %s", reply_path)
            await message.reply_text("Sorry, reply image missing.")
        return
    best_entry, best_score, best_overlap = find_best_match(query_text)

    lcs_len_best = longest_common_substring_length(normalize_text(query_text), normalize_text(best_entry.get("text", "") if best_entry else "")) if best_entry else 0
    if best_entry and (lcs_len_best >= MIN_LCS_ACCEPT or best_overlap >= MIN_TOKEN_OVERLAP or best_score >= SIMILARITY_THRESHOLD or (best_score >= MIN_SIMILARITY_FALLBACK and best_overlap >= 0)):
        image_path = best_entry.get("image_path")
        if image_path and os.path.isfile(image_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                media = []
                media.append(InputMediaPhoto(open(image_path, "rb"), caption=f"Match {'overlap' if best_overlap>=MIN_TOKEN_OVERLAP else str(best_score)+'%'}"))
                media.append(InputMediaPhoto(open(local_path, "rb")))
                await update.message.reply_media_group(media=media)
            except Exception as e:
                logger.exception("Failed to send matched image: %s", e)
                await message.reply_text("Sorry, failed to send the matched image.")
        else:
            await message.reply_text("Matched entry found but image file is missing.")
    else:
        await message.reply_text("Sorry, no match found.")


# "$+" text command handler (moderators): reply to a photo with $+ to add it
async def add_dollar_plus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    message = update.effective_message
    if not is_moderator(user.id):
        await message.reply_text("Only moderators can use this.")
        return
    if not message.reply_to_message or not message.reply_to_message.photo:
        await message.reply_text("Reply $+ to a photo to add it.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    largest = message.reply_to_message.photo[-1]
    local_path = await download_photo_as_file(update, context, IMAGES_DIR, largest.file_id, fallback_suffix="_add")
    if not local_path:
        await message.reply_text("Failed to download the photo.")
        return
    try:
        text = extract_text_for_add(local_path)
    except Exception as e:
        logger.exception("Add OCR failed: %s", e)
        text = ""
    add_entry_to_db(local_path, text)
    await message.reply_text("Added image to database.")


# Admin utility to rebuild missing/weak texts in DB
async def rebuild_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_admin(user.id):
        await update.message.reply_text("Only the administrator can use /rebuild.")
        return
    await update.message.reply_text("Rebuilding OCR text for stored images. This may take a while...")
    db = load_database()
    changed = 0
    for entry in db:
        text = entry.get("text", "")
        if count_cjk(text) >= 2:
            # also refresh derived fields
            entry["norm_text"] = normalize_text(text)
            entry["tokens"] = sorted(list(build_token_set(text)))
            continue
        image_path = entry.get("image_path")
        if image_path and os.path.isfile(image_path):
            try:
                new_text = extract_text_for_add(image_path)
                entry["text"] = new_text
                entry["norm_text"] = normalize_text(new_text)
                entry["tokens"] = sorted(list(build_token_set(new_text)))
                changed += 1
            except Exception as e:
                logger.exception("Rebuild OCR failed for %s: %s", image_path, e)
    save_database(db)
    await update.message.reply_text(f"Rebuild complete. Updated {changed} entries.")

# Refresh at startup for entries with missing derived fields
def refresh_database_texts_on_startup() -> None:
    db = load_database()
    updated = False
    for entry in db:
        text = entry.get("text", "")
        if "norm_text" not in entry:
            entry["norm_text"] = normalize_text(text)
            updated = True
        if "tokens" not in entry:
            entry["tokens"] = sorted(list(build_token_set(text))) if text else []
            updated = True
        if "tail_tokens" not in entry:
            entry["tail_tokens"] = get_tail_tokens(text, k=2)
            updated = True
    if updated:
        save_database(db)


# =====================
# App bootstrap
# =====================

def main() -> None:
    ensure_storage()
    refresh_database_texts_on_startup()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("rebuild", rebuild_db))
    # Moderator manage flow
    manage_conv = ConversationHandler(
        entry_points=[CommandHandler("manage", manage)],
        states={
            MANAGE_ADD_TEXT: [CallbackQueryHandler(manage_cb)],
            MANAGE_WAIT_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, manage_capture_text)],
            MANAGE_WAIT_PHOTO: [MessageHandler((filters.PHOTO | filters.Document.IMAGE) & ~filters.COMMAND, manage_capture_photo)],
        },
        fallbacks=[CommandHandler("cancel", manage_cancel)],
        name="manage_conv",
        persistent=False,
    )
    app.add_handler(manage_conv)
    app.add_handler(CommandHandler("addmapping", addmapping))
    app.add_handler(CommandHandler("add", add_command))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^\$\+$"), add_dollar_plus))

    # Photos with or without caption
    app.add_handler(MessageHandler(filters.PHOTO, photo_router))

    # Images sent as documents
    app.add_handler(MessageHandler(filters.Document.ALL, document_router))

    logger.info("Bot is starting... Using OCR backend: cloud (OCR.space)")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()