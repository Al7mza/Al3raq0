#!/bin/bash

# Treasure Map Bot Setup Script
# This script helps you set up the bot on Ubuntu/Debian systems

echo "🎯 Treasure Map Bot Setup Script"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please don't run this script as root. Run it as a regular user."
    exit 1
fi

# Check Python version
echo "🔍 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Installing..."
    sudo apt-get install -y python3-pip
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

# Verify Tesseract installation
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR installed successfully"
    echo "Available languages:"
    tesseract --list-langs
else
    echo "❌ Failed to install Tesseract OCR"
    exit 1
fi

# Install Python dependencies
echo ""
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully"
else
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "⚙️ Creating configuration file..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "📝 Please edit .env file with your bot configuration:"
    echo "   - BOT_TOKEN: Get from @BotFather on Telegram"
    echo "   - ADMIN_USER_IDS: Your Telegram user ID"
    echo ""
    echo "To get your user ID, send /start to @userinfobot"
else
    echo "✅ .env file already exists"
fi

# Create uploads directory
echo ""
echo "📁 Creating uploads directory..."
mkdir -p uploads
echo "✅ Uploads directory created"

# Set permissions
echo ""
echo "🔐 Setting file permissions..."
chmod +x bot.py
chmod 755 uploads
echo "✅ Permissions set"

# Final instructions
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your bot configuration"
echo "2. Run the bot: python3 bot.py"
echo "3. Test with /start command in Telegram"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo "Happy treasure hunting! 🗺️✨"