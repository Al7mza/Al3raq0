#!/bin/bash

# Treasure Map Bot Setup Script
# This script installs all dependencies and sets up the bot

echo "🎯 Setting up Treasure Map Bot..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

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

# Try different installation methods based on Python version
echo "🐍 Installing Python dependencies..."

# Method 1: Try main requirements
echo "🔍 Trying main requirements.txt..."
if pip3 install -r requirements.txt; then
    echo "✅ Main requirements installed successfully"
else
    echo "⚠️  Main requirements failed, trying alternative method..."
    
    # Method 2: Try alternative requirements for Python 3.13+
    if [ -f "requirements-python313.txt" ]; then
        echo "🔍 Trying Python 3.13+ compatible requirements..."
        if pip3 install -r requirements-python313.txt; then
            echo "✅ Alternative requirements installed successfully"
        else
            echo "⚠️  Alternative requirements failed, trying minimal installation..."
            
            # Method 3: Install packages one by one
            echo "🔍 Installing packages individually..."
            pip3 install python-telegram-bot
            pip3 install Pillow
            pip3 install opencv-python-headless
            pip3 install numpy
            pip3 install pytesseract
            pip3 install python-dotenv
            
            # Try fuzzy matching packages
            if ! pip3 install fuzzywuzzy python-Levenshtein; then
                echo "⚠️  Installing rapidfuzz as alternative..."
                pip3 install rapidfuzz
            fi
            
            echo "✅ Individual packages installed"
        fi
    else
        echo "❌ Alternative requirements file not found"
        exit 1
    fi
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
echo ""
echo "🔧 If you encounter issues:"
echo "- Try running: python3 test_bot.py"
echo "- Check Python version compatibility"
echo "- Install missing packages manually if needed"