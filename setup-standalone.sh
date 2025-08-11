#!/bin/bash

# Standalone Bot Setup Script
# This script sets up the bot with minimal dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Standalone Treasure Map Bot...${NC}"

# Check Python version
echo -e "${YELLOW}🔍 Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✅ Python version: $PYTHON_VERSION${NC}"

# Check if pip is available
echo -e "${YELLOW}🔍 Checking pip...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  pip3 not found, trying pip...${NC}"
    if ! command -v pip &> /dev/null; then
        echo -e "${YELLOW}⚠️  pip not found, installing pip...${NC}"
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user
        rm get-pip.py
    fi
    PIP_CMD="pip"
else
    PIP_CMD="pip3"
fi

echo -e "${GREEN}✅ Using: $PIP_CMD${NC}"

# Install python-telegram-bot
echo -e "${YELLOW}🔍 Installing python-telegram-bot...${NC}"
if $PIP_CMD install python-telegram-bot; then
    echo -e "${GREEN}✅ python-telegram-bot installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Trying alternative installation method...${NC}"
    if $PIP_CMD install --user python-telegram-bot; then
        echo -e "${GREEN}✅ python-telegram-bot installed successfully (user mode)${NC}"
    else
        echo -e "${YELLOW}⚠️  Trying with --break-system-packages...${NC}"
        if $PIP_CMD install --break-system-packages python-telegram-bot; then
            echo -e "${GREEN}✅ python-telegram-bot installed successfully${NC}"
        else
            echo -e "${YELLOW}⚠️  Installation failed, but you can try running the bot anyway${NC}"
        fi
    fi
fi

# Create uploads directory
echo -e "${YELLOW}🔍 Creating uploads directory...${NC}"
mkdir -p uploads
echo -e "${GREEN}✅ Uploads directory created${NC}"

# Make bot file executable
echo -e "${YELLOW}🔍 Making bot file executable...${NC}"
chmod +x bot-standalone.py
echo -e "${GREEN}✅ Bot file is executable${NC}"

# Show final instructions
echo -e "${BLUE}🎉 Setup Complete!${NC}"
echo ""
echo -e "${GREEN}📋 To run your bot:${NC}"
echo -e "${YELLOW}   python3 bot-standalone.py${NC}"
echo ""
echo -e "${GREEN}📋 Your bot is already configured with:${NC}"
echo -e "${YELLOW}   🤖 Bot Token: 8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY${NC}"
echo -e "${YELLOW}   👑 Admin ID: 1694244496${NC}"
echo -e "${YELLOW}   🔑 Google API: AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k${NC}"
echo ""
echo -e "${GREEN}📋 If you get import errors, try:${NC}"
echo -e "${YELLOW}   pip3 install --user python-telegram-bot${NC}"
echo ""
echo -e "${BLUE}🚀 Your bot is ready to run!${NC}"