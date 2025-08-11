#!/bin/bash
echo "🚀 Setting up Treasure Map Bot..."

# Check Python version
echo "🔍 Checking Python version..."
python3 --version

# Check pip
echo "🔍 Checking pip..."
pip3 --version

# Install telegram bot
echo "📦 Installing python-telegram-bot..."
pip3 install python-telegram-bot

# Create uploads folder
echo "📁 Creating uploads folder..."
mkdir -p uploads

# Make bot executable
echo "🔧 Making bot executable..."
chmod +x treasure_map_bot.py

echo "✅ Setup complete!"
echo "🚀 Run your bot: python3 treasure_map_bot.py"