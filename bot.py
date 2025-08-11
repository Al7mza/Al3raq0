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

# Storage paths
IMAGES_DIR: str = "images"
DB_PATH: str = "database.json"

# Similarity threshold
SIMILARITY_THRESHOLD: int = 80

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


def yellow_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Return a cleaned mask isolating yellow regions in a BGR image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Yellow range in HSV (tune if needed)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Morphological operations to remove noise and strengthen text strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def ocr_with_cloud(image_np: np.ndarray) -> str:
    """Call OCR.space API to extract Chinese Simplified text from the provided image array."""
    # Encode to JPEG to reduce size
    success, encoded = cv2.imencode(
        ".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not success:
        return ""
    data = {
        "language": "chs",  # Chinese Simplified
        "isOverlayRequired": "false",
        "OCREngine": "2",
        "scale": "true",
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
    """Extract Chinese text by isolating yellow regions and running Cloud OCR."""
    image = cv2.imread(image_path)
    if image is None:
        return ""

    mask = yellow_mask_bgr(image)
    masked = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return ocr_with_cloud(thresh)


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
    """Return (best_entry, best_score)."""
    db = load_database()
    if not db:
        return None, 0

    best_entry: Optional[Dict[str, Any]] = None
    best_score: int = -1
    for entry in db:
        candidate_text = entry.get("text", "") or ""
        score = fuzz.ratio(query_text, candidate_text)
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