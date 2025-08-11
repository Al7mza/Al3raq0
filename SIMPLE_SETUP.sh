#!/bin/bash
echo "🚀 Setting up ULTIMATE SIMPLE BOT..."

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
chmod +x ULTIMATE_SIMPLE_BOT.py

echo "✅ Setup complete!"
echo "🚀 Run your bot: python3 ULTIMATE_SIMPLE_BOT.py"