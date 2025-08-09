## Telegram TikTok/Douyin Downloader Bot

This bot downloads high-quality videos from TikTok (global) and Douyin (Chinese TikTok) and sends them back in Telegram.

### What’s included
- `bot.py`: Telegram bot source code using python-telegram-bot v20 and yt-dlp
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

### Run
```bash
python bot.py
```

Send a TikTok or Douyin URL to the bot in a private chat.

### Notes
- The bot attempts to fetch the best available quality.
- Telegram bots have a file size limit. If a video exceeds the upload limit, the bot will notify you instead of uploading the file.
- If a share link is shortened (e.g., `vm.tiktok.com` or `v.douyin.com`), the bot resolves it automatically.