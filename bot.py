import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
import yt_dlp
from telegram import InputFile, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

# User-provided bot token (embedded as requested)
TELEGRAM_BOT_TOKEN = "8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY"

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Basic URL regex and allowed hosts
URL_REGEX = re.compile(r"https?://[^\s]+", re.IGNORECASE)
ALLOWED_HOST_KEYWORDS = (
    "tiktok.com",
    "douyin.com",
    "iesdouyin.com",
    "vm.tiktok.com",
    "v.douyin.com",
)

# Try to upload large files; Telegram bot API supports large uploads now, but errors if too big
# We'll still warn if extremely large to avoid wasteful attempts.
HARD_WARN_BYTES = 1_800 * 1024 * 1024  # ~1.8 GB


def is_allowed_link(url: str) -> bool:
    lowered = url.lower()
    return any(host in lowered for host in ALLOWED_HOST_KEYWORDS)


def extract_first_url(text: str) -> Optional[str]:
    if not text:
        return None
    match = URL_REGEX.search(text)
    if match:
        return match.group(0)
    return None


def resolve_redirects(url: str, timeout: int = 12) -> str:
    try:
        # Resolve common short share links (vm.tiktok.com, v.douyin.com, etc.)
        resp = requests.get(url, allow_redirects=True, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        })
        return resp.url if resp.url else url
    except Exception:
        return url


def _download_with_ytdlp(url: str, download_dir: Path) -> Tuple[Path, dict]:
    """Blocking download via yt-dlp. Returns (file_path, info_dict)."""
    ydl_opts = {
        "outtmpl": str(download_dir / "%(title).180B.%(ext)s"),
        # Prefer highest quality video+audio; fallback to best single stream
        "format": "bv*+ba/b/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Merge to mp4 when needed
        "merge_output_format": "mp4",
        # Occasionally helpful for TikTok/Douyin
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.tiktok.com/",
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp returns filename via prepare_filename; ensure we get the actual final file
        filename = ydl.prepare_filename(info)

    file_path = Path(filename)
    if not file_path.exists():
        # Try locating any resulting mp4 in dir as a fallback
        candidates = sorted(download_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("Downloaded file not found.")
        file_path = candidates[0]

    return file_path, info


async def download_video(url: str) -> Tuple[Path, dict]:
    resolved = await asyncio.to_thread(resolve_redirects, url)
    download_dir_obj = tempfile.TemporaryDirectory(prefix="ttdl_")
    download_dir = Path(download_dir_obj.name)

    try:
        file_path, info = await asyncio.to_thread(_download_with_ytdlp, resolved, download_dir)
        # Attach the tempdir object to file_path so it can be cleaned later
        file_path.tempdir_obj = download_dir_obj  # type: ignore[attr-defined]
        return file_path, info
    except Exception as e:
        # Ensure temp dir is cleaned on failure
        try:
            download_dir_obj.cleanup()
        except Exception:
            pass
        raise e


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a TikTok or Douyin link, and I will download the video in high quality."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just paste a TikTok (tiktok.com) or Douyin (douyin.com) link here. I will fetch the best-quality video."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    url = extract_first_url(message.text)
    if not url or not is_allowed_link(url):
        return

    await message.reply_chat_action(action=ChatAction.TYPING)
    notice = await message.reply_text("Downloading your video... This may take a few seconds.")

    file_path: Optional[Path] = None
    try:
        file_path, info = await download_video(url)
        file_size = file_path.stat().st_size

        if file_size >= HARD_WARN_BYTES:
            await notice.edit_text(
                (
                    "Video is very large; attempting upload (may fail due to Telegram limits).\n"
                    f"Size: {file_size / (1024 * 1024):.1f} MB"
                )
            )

        caption = info.get("title") or "TikTok/Douyin video"
        await message.reply_chat_action(action=ChatAction.UPLOAD_VIDEO)
        try:
            with file_path.open("rb") as f:
                await message.reply_video(
                    video=InputFile(f, filename=file_path.name),
                    caption=caption[:1024],
                    supports_streaming=True,
                )
            await notice.delete()
        except TelegramError as te:
            text = str(te)
            if "Request Entity Too Large" in text or "too large" in text.lower():
                await notice.edit_text(
                    (
                        "Downloaded, but the file is too large to upload via Telegram bot.\n"
                        f"File size: {file_size / (1024 * 1024):.1f} MB"
                    )
                )
            else:
                raise

    except yt_dlp.utils.DownloadError:
        logger.exception("yt-dlp download error")
        await notice.edit_text("Failed to download this link. It may be unavailable or blocked.")
    except Exception:
        logger.exception("Unexpected error")
        await notice.edit_text("An unexpected error occurred while processing your link.")
    finally:
        # Cleanup temp directory
        try:
            if file_path is not None:
                tempdir_obj = getattr(file_path, "tempdir_obj", None)
                if tempdir_obj is not None:
                    tempdir_obj.cleanup()
        except Exception:
            pass


def main() -> None:
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("Telegram bot token not configured.")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        pass