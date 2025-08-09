## Telegram TikTok/Douyin Downloader Bot

This bot downloads high-quality videos from TikTok (global) and Douyin (Chinese TikTok) and sends them back in Telegram.

### What’s included
- `bot.py`: Telegram bot source code using python-telegram-bot v21 and yt-dlp
- `app.py`: Entrypoint compatible with hosts expecting `/home/container/app.py`
- `requirements.txt`: Dependencies

### Prerequisites
- Python 3.10+
- ffmpeg (required by yt-dlp to merge streams)

Install ffmpeg on Debian/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run (local)
```bash
python bot.py
```

### Run (hosts expecting app.py)
Some hosting panels run `/home/container/app.py` by default. This repo includes `app.py` which forwards to the bot.
```bash
python app.py
```

Send a TikTok or Douyin URL to the bot in a private chat.

### Optional: Pass cookies for higher success rate
Some regions and accounts require cookies to access TikTok/Douyin. Provide cookies to yt-dlp in Netscape format:
1. Export cookies for `tiktok.com` and/or `douyin.com` using a browser extension (e.g., "Get cookies.txt" for Chrome/Firefox).
2. Save as `cookies.txt` in the same directory as `bot.py` (or set `COOKIES_FILE` env var to an absolute path).
3. On some panels, place it at `/home/container/cookies.txt`.

The bot auto-detects the file at:
- `./cookies.txt`
- value of `COOKIES_FILE` env var
- `/home/container/cookies.txt`

### Notes
- The bot attempts to fetch the best available quality and merges streams.
- If a shortened link is used (e.g., `vm.tiktok.com`, `v.douyin.com`), the bot resolves it automatically.
- If you see "Failed to download this link...", try adding `cookies.txt` as above.