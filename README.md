# 🎯 Treasure Map Bot

A Telegram bot that helps players find matching treasure maps using OCR and text similarity matching.

## ✨ Features

- **OCR Text Extraction**: Automatically extracts text from images using Tesseract
- **Multi-language Support**: Chinese (Simplified) + English
- **Smart Text Matching**: Fuzzy text matching with configurable similarity threshold
- **Admin Panel**: Upload, delete, and manage reference images
- **User-friendly**: Simple image upload to find matches

## 🚀 Quick Start

### 1. Setup
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup (installs dependencies)
./setup.sh
```

### 2. Configuration
Edit `.env` file:
```env
BOT_TOKEN=your_bot_token_here
ADMIN_USER_IDS=your_telegram_user_id
```

### 3. Test
```bash
python3 test_bot.py
```

### 4. Run
```bash
python3 bot.py
```

## 🔧 **Python 3.13+ Compatibility**

If you're getting build errors with Python 3.13+, use the simplified version:

```bash
# Install only essential packages
pip3 install python-telegram-bot python-dotenv Pillow

# Run simplified bot (Python 3.13+ compatible)
python3 bot-simple.py
```

**See `INSTALLATION_GUIDE.md` for detailed troubleshooting!**

## 📋 Commands

### Admin Commands
- `/add` - Upload reference images (reply to image)
- `/delete` - Remove reference images
- `/list` - Show all stored images
- `/setthreshold <value>` - Set similarity threshold (1-100%)

### User Commands
- Send any image to find matching treasure maps
- `/help` - Get detailed help
- `/start` - Welcome message

## 🔧 Requirements

- Python 3.8+
- Tesseract OCR
- System: Ubuntu/Debian (for setup script)

## 📁 Files

- `bot.py` - Main bot application
- `bot-simple.py` - **Simplified bot for Python 3.13+**
- `config.py` - Configuration settings
- `database.py` - Database operations
- `ocr_handler.py` - OCR text extraction
- `text_matcher.py` - Text similarity matching
- `setup.sh` - Automated setup script
- `test_bot.py` - Component testing
- `requirements.txt` - Python dependencies
- `requirements-python313.txt` - **Alternative dependencies for Python 3.13+**
- `requirements-minimal.txt` - **Minimal dependencies**
- `INSTALLATION_GUIDE.md` - **Troubleshooting guide**

## 🎯 How It Works

1. **Admin uploads reference images** using `/add` command
2. **Bot extracts text** from images using OCR
3. **Users send treasure map images** to find matches
4. **Bot compares text similarity** and returns best match
5. **Similarity threshold** determines match quality

## 🔍 Similarity Algorithm

Combines multiple text matching approaches:
- Fuzzy string matching (Levenshtein distance)
- Token-based similarity
- Keyword overlap analysis
- Configurable weights for optimal results

## 🚨 **Troubleshooting**

### Build Errors (Python 3.13+)
```bash
# Use simplified bot instead
python3 bot-simple.py

# Or install with pre-compiled wheels
pip3 install --only-binary=all python-telegram-bot python-dotenv Pillow
```

### Missing Dependencies
```bash
# Try alternative requirements
pip3 install -r requirements-python313.txt

# Or minimal installation
pip3 install -r requirements-minimal.txt
```

### Full Troubleshooting Guide
See `INSTALLATION_GUIDE.md` for step-by-step solutions!

## 📞 Support

For issues or questions:
1. Check `INSTALLATION_GUIDE.md`
2. Run: `python3 test_bot.py`
3. Try the simplified bot: `python3 bot-simple.py`

## 📝 License

This project is open source and available under the MIT License.