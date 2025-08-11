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
from hanziconv import HanziConv
from fuzzywuzzy import fuzz
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =====================
# Configuration
# =====================
# Place your bot token and admin user id here
BOT_TOKEN: str = "8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY"
# Replace with your numeric Telegram user ID
ADMIN_ID: int = 123456789

# Cloud OCR config (OCR.space)
OCR_API_KEY: str = os.environ.get("OCR_API_KEY", "K88956884888957")
OCR_API_ENDPOINT: str = "https://api.ocr.space/parse/image"
OCR_TIMEOUT_SECONDS: int = 30

# Preprocessing toggles
OCR_VARIANTS: List[str] = [
    "original",
    "gray_otsu",
    "clahe_v",
    "bilateral_otsu",
    "adaptive",
]

# Storage paths
IMAGES_DIR: str = "images"
DB_PATH: str = "database.json"

# Similarity thresholds and weights
SIMILARITY_THRESHOLD: int = int(os.environ.get("SIMILARITY_THRESHOLD", "80"))
WEIGHT_CHAR_RATIO: float = float(os.environ.get("WEIGHT_CHAR_RATIO", "0.35"))
WEIGHT_PARTIAL_RATIO: float = float(os.environ.get("WEIGHT_PARTIAL_RATIO", "0.25"))
WEIGHT_TOKEN_SET: float = float(os.environ.get("WEIGHT_TOKEN_SET", "0.25"))
WEIGHT_PARTIAL_TOKEN_SET: float = float(os.environ.get("WEIGHT_PARTIAL_TOKEN_SET", "0.1"))
WEIGHT_NGRAM_JACCARD: float = float(os.environ.get("WEIGHT_NGRAM_JACCARD", "0.05"))
NGRAM_N: int = int(os.environ.get("NGRAM_N", "2"))

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
    if not os.path.isfile(DB_PATH):
        with open(DB_PATH, "w", encoding="utf-8") as f:
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


def preprocess_variants_full(image_bgr: np.ndarray) -> List[np.ndarray]:
    """Generate multiple preprocessing variants of the full image for OCR."""
    variants: List[np.ndarray] = []

    if "original" in OCR_VARIANTS:
        variants.append(image_bgr)

    if "gray_otsu" in OCR_VARIANTS:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        variants.append(th)

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

    return variants


def ocr_with_cloud(image_np: np.ndarray) -> str:
    """Call OCR.space API to extract Chinese Simplified text from the provided image array."""
    success, encoded = cv2.imencode(
        ".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not success:
        return ""
    data = {
        "language": "chs",
        "isOverlayRequired": "false",
        "OCREngine": "2",
        "scale": "true",
        "detectOrientation": "true",
    }
    headers = {"apikey": OCR_API_KEY}
    files = {
        "file": ("image.jpg", encoded.tobytes(), "image/jpeg"),
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
        logger.error("Cloud OCR request failed: %s", e)
        return ""

    try:
        if payload.get("IsErroredOnProcessing"):
            logger.warning("OCR error: %s", payload.get("ErrorMessage"))
            return ""
        results = payload.get("ParsedResults", [])
        texts: List[str] = []
        for r in results:
            t = r.get("ParsedText", "")
            if t:
                texts.append(t)
        combined = " ".join(s.strip() for s in texts if s.strip())
        return combined
    except Exception as e:
        logger.error("Cloud OCR parse failed: %s", e)
        return ""


def extract_text_from_image(image_path: str) -> str:
    """Extract Chinese text by running OCR on the full image with multiple preprocessing variants."""
    image = cv2.imread(image_path)
    if image is None:
        return ""

    candidates: List[str] = []
    for variant in preprocess_variants_full(image):
        text = ocr_with_cloud(variant)
        if text:
            candidates.append(text)

    if not candidates:
        return ""

    def cjk_count(s: str) -> int:
        return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")

    best = max(candidates, key=lambda s: (cjk_count(s), len(s)))
    return best


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


def add_entry_to_db(image_path: str, text: str) -> None:
    db = load_database()
    entry = {
        "image_path": image_path,
        "text": text,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    db.append(entry)
    save_database(db)


def find_best_match(query_text: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (best_entry, best_score) using multi-metric weighted similarity."""
    db = load_database()
    if not db:
        return None, 0

    best_entry: Optional[Dict[str, Any]] = None
    best_score: int = -1

    for entry in db:
        candidate_text = entry.get("text", "") or ""
        score = compute_similarity(query_text, candidate_text)
        if score > best_score:
            best_score = score
            best_entry = entry

    return best_entry, best_score if best_entry else 0


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


# =====================
# Handlers
# =====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send a photo containing yellow Chinese text. Admin can add reference images using /add."
    )


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
        text = extract_text_from_image(local_path)
        add_entry_to_db(local_path, text)
        await message.reply_text(
            f"Added image to database. Extracted text: {text or '(none)'}"
        )
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

    caption = (message.caption or "").strip().lower()

    if caption.startswith("/add"):
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
        text = extract_text_from_image(local_path)
        add_entry_to_db(local_path, text)
        await message.reply_text(
            f"Added image to database. Extracted text: {text or '(none)'}"
        )
        return

    # User query flow
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    largest = message.photo[-1]
    tmp_path = await download_photo_as_file(update, context, IMAGES_DIR, largest.file_id, fallback_suffix="_query")
    if not tmp_path:
        await message.reply_text("Failed to download the photo.")
        return

    query_text = extract_text_from_image(tmp_path)
    best_entry, best_score = find_best_match(query_text)

    logger.info("Cloud OCR | Query text: %s | Best score: %s", query_text, best_score)

    if best_entry and best_score >= SIMILARITY_THRESHOLD:
        image_path = best_entry.get("image_path")
        if image_path and os.path.isfile(image_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                with open(image_path, "rb") as f:
                    await message.reply_photo(photo=f, caption=f"Match {best_score}%")
            except Exception as e:
                logger.exception("Failed to send matched image: %s", e)
                await message.reply_text("Sorry, failed to send the matched image.")
        else:
            await message.reply_text("Matched entry found but image file is missing.")
    else:
        await message.reply_text("Sorry, no match found.")


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
    caption = (message.caption or "").strip().lower()

    if caption.startswith("/add"):
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

        text = extract_text_from_image(local_path)
        add_entry_to_db(local_path, text)
        await message.reply_text(
            f"Added image to database. Extracted text: {text or '(none)'}"
        )
        return

    # User query flow for document image
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        telegram_file = await context.bot.get_file(doc.file_id)
        local_path = os.path.join(IMAGES_DIR, f"{int(time.time())}_{doc.file_unique_id}_query")
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

    query_text = extract_text_from_image(local_path)
    best_entry, best_score = find_best_match(query_text)

    if best_entry and best_score >= SIMILARITY_THRESHOLD:
        image_path = best_entry.get("image_path")
        if image_path and os.path.isfile(image_path):
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            try:
                with open(image_path, "rb") as f:
                    await message.reply_photo(photo=f, caption=f"Match {best_score}%")
            except Exception as e:
                logger.exception("Failed to send matched image: %s", e)
                await message.reply_text("Sorry, failed to send the matched image.")
        else:
            await message.reply_text("Matched entry found but image file is missing.")
    else:
        await message.reply_text("Sorry, no match found.")


# =====================
# App bootstrap
# =====================

def main() -> None:
    ensure_storage()

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_command))

    # Photos with or without caption
    app.add_handler(MessageHandler(filters.PHOTO, photo_router))

    # Images sent as documents
    app.add_handler(MessageHandler(filters.Document.ALL, document_router))

    logger.info("Bot is starting... Using OCR backend: cloud (OCR.space)")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()