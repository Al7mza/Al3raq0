#!/bin/bash

# Treasure Map Bot Setup Script
# This script installs all dependencies and sets up the bot

echo "🎯 Setting up Treasure Map Bot..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip3 is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ Python 3 and pip3 found"

# Install system dependencies (Ubuntu/Debian)
echo "📦 Installing system dependencies..."

# Update package list
sudo apt-get update

# Install Tesseract OCR and language packs
sudo apt-get install -y tesseract-ocr
sudo apt-get install -y tesseract-ocr-chi-sim  # Chinese simplified
sudo apt-get install -y tesseract-ocr-eng       # English

# Install OpenCV dependencies
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

echo "✅ System dependencies installed"

# Install Python dependencies
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
    echo "📝 Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "⚠️  Please edit .env file with your bot token and admin user ID"
    else
        echo "❌ .env.example not found. Please create .env file manually"
    fi
fi

# Create uploads directory
echo "📁 Creating uploads directory..."
mkdir -p uploads

# Set executable permissions
chmod +x bot.py
chmod +x setup.sh

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your bot token and admin user ID"
echo "2. Get your Telegram user ID by sending /start to @userinfobot"
echo "3. Test the bot: python3 test_bot.py"
echo "4. Run the bot: python3 bot.py"
echo ""
echo "📚 For help, see README.md or run: python3 demo.py"