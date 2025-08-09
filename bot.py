import asyncio
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image
import pytesseract
from rapidfuzz import fuzz

from telegram import Update, InputFile
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------------------------
# Configuration & Setup
# ---------------------------
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ADMIN_USER_IDS_ENV = os.getenv("ADMIN_USER_IDS", "").strip()
ADMIN_USER_IDS = set()
if ADMIN_USER_IDS_ENV:
    for part in ADMIN_USER_IDS_ENV.split(","):
        part = part.strip()
        if part.isdigit():
            ADMIN_USER_IDS.add(int(part))

DATA_DIR = os.path.join("/workspace", "bot_data")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference_images")
DB_PATH = os.path.join(DATA_DIR, "db.sqlite3")

os.makedirs(REFERENCE_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------
# Database Helpers
# ---------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

INSERT_REFERENCE_SQL = """
INSERT INTO references (file_id, file_path, text, created_at)
VALUES (?, ?, ?, ?)
"""

SELECT_ALL_REFERENCES_SQL = """
SELECT id, file_id, file_path, text, created_at FROM references ORDER BY id ASC
"""

DELETE_ALL_REFERENCES_SQL = "DELETE FROM references"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------------------------
# OCR and Text Processing
# ---------------------------

# Ensure Tesseract language packs are installed: chi_sim, chi_tra
TESS_LANG = os.getenv("TESS_LANG", "chi_sim+chi_tra")

# Compile regex to keep only CJK, Latin letters and digits
_ALLOWED_CHARS_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = _ALLOWED_CHARS_RE.sub("", text)
    return text


def ocr_image(image_path: str) -> str:
    # Use Tesseract to extract Chinese/English text
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.exception("Failed to open image for OCR: %s", e)
        return ""
    try:
        raw_text = pytesseract.image_to_string(image, lang=TESS_LANG)
    except Exception as e:
        logger.exception("Tesseract OCR failed: %s", e)
        raw_text = ""
    normalized = normalize_text(raw_text)
    return normalized


# ---------------------------
# Reference Management
# ---------------------------

def add_reference(file_id: str, file_path: str, text: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            INSERT_REFERENCE_SQL,
            (file_id, file_path, text, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def load_references() -> List[sqlite3.Row]:
    conn = get_db_connection()
    try:
        rows = conn.execute(SELECT_ALL_REFERENCES_SQL).fetchall()
        return rows
    finally:
        conn.close()


def clear_references() -> int:
    conn = get_db_connection()
    try:
        cur = conn.execute(DELETE_ALL_REFERENCES_SQL)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---------------------------
# Matching Logic
# ---------------------------

def find_best_match(query_text: str, references: List[sqlite3.Row]) -> Tuple[Optional[sqlite3.Row], float]:
    best_row = None
    best_score = -1.0
    for row in references:
        ref_text = row["text"] or ""
        score = fuzz.ratio(query_text, ref_text) / 100.0
        if score > best_score:
            best_score = score
            best_row = row
    return best_row, best_score


# ---------------------------
# Telegram Handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    if user_id in ADMIN_USER_IDS:
        await update.message.reply_text(
            "Admin mode:\n- Send reference photos containing Chinese text to store them.\n- /listrefs to see count.\n- /clearrefs to delete all."
        )
    else:
        await update.message.reply_text(
            "Send me a photo with Chinese text. I will try to match it against the admin's references and return the closest match (>=90%)."
        )


async def listrefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    if user_id not in ADMIN_USER_IDS:
        return
    refs = load_references()
    await update.message.reply_text(f"Stored references: {len(refs)}")


async def clearrefs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    if user_id not in ADMIN_USER_IDS:
        return
    n = clear_references()
    await update.message.reply_text(f"Cleared {n} references.")


async def handle_admin_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if user_id not in ADMIN_USER_IDS:
        return

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    # Pick highest resolution photo
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    timestamp = int(time.time() * 1000)
    local_path = os.path.join(REFERENCE_DIR, f"ref_{timestamp}.jpg")
    await file.download_to_drive(local_path)

    text = ocr_image(local_path)
    add_reference(photo.file_id, local_path, text)

    short_text = (text[:60] + "...") if len(text) > 60 else text
    await update.message.reply_text(
        f"Reference saved. OCR (normalized): {short_text if short_text else '[no text recognized]'}"
    )


async def handle_user_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    user_id = update.effective_user.id if update.effective_user else None
    if user_id in ADMIN_USER_IDS:
        return  # Admin photos are handled by handle_admin_photo

    await update.message.chat.send_action(ChatAction.TYPING)

    # Download user's image to a temp path
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    tmp_path = os.path.join(DATA_DIR, f"tmp_{int(time.time()*1000)}.jpg")
    await file.download_to_drive(tmp_path)

    try:
        query_text = ocr_image(tmp_path)
        if not query_text:
            await update.message.reply_text("I couldn't read any text from the image.")
            return
        references = load_references()
        if not references:
            await update.message.reply_text("No references available yet. Please try again later.")
            return

        best_row, best_score = find_best_match(query_text, references)
        if best_row is not None and best_score >= 0.90:
            # Prefer reusing Telegram file_id to avoid reupload
            await update.message.reply_photo(
                photo=best_row["file_id"],
                caption=f"Best match: {best_score*100:.1f}%",
            )
        else:
            await update.message.reply_text("No close match found (>=90%).")
    finally:
        # Cleanup temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in environment.")

    init_db()

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("listrefs", listrefs))
    app.add_handler(CommandHandler("clearrefs", clearrefs))

    # Admin photos -> saved as references
    app.add_handler(MessageHandler(filters.PHOTO & filters.User(user_id=list(ADMIN_USER_IDS)), handle_admin_photo))

    # User photos -> matching
    app.add_handler(MessageHandler(filters.PHOTO & ~filters.User(user_id=list(ADMIN_USER_IDS)), handle_user_photo))

    return app


def main() -> None:
    app = build_application()
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass