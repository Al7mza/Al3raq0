# Telegram Image-Text Matcher Bot

This bot lets an Admin upload reference images containing Chinese text. Users can then send an image, the bot performs OCR (Chinese) and returns the closest matching reference image if similarity >= 90%.

## Requirements

- Linux with `tesseract-ocr` and Chinese language packs installed
- Python 3.10+

## Install system dependencies

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
```

## Install Python dependencies

```bash
pip install -r requirements.txt
```

## Configure environment

Set environment variables:

- `TELEGRAM_BOT_TOKEN`: your bot token from BotFather
- `ADMIN_USER_IDS`: comma-separated numeric Telegram user IDs who can upload references
- Optional: `TESS_LANG` (default `chi_sim+chi_tra`)

For example, create a `.env` file in project root:

```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
ADMIN_USER_IDS=111111111,222222222
TESS_LANG=chi_sim+chi_tra
```

## Run the bot

```bash
python bot.py
```

## Usage

- Admins: send photos to the bot; it will OCR and store them as references. Use `/listrefs` to see count, `/clearrefs` to delete all.
- Users: send a photo with Chinese text. If a >=90% text match is found, you'll receive the matching reference image.

Data is stored under `/workspace/bot_data/` with images in `reference_images/` and a SQLite database `db.sqlite3`.