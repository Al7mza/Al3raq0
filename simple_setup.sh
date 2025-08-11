#!/bin/bash
echo "🚀 Setting up Simple Treasure Map Bot..."

# Check Python
echo "🔍 Checking Python..."
python3 --version

# Install telegram bot
echo "📦 Installing python-telegram-bot..."
pip3 install python-telegram-bot

# Create uploads folder
echo "📁 Creating uploads folder..."
mkdir -p uploads

# Make bot executable
echo "🔧 Making bot executable..."
chmod +x simple_bot.py

echo "✅ Setup complete!"
echo "🚀 Run your bot: python3 simple_bot.py"