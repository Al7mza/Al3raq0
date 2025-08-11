#!/bin/bash

# Treasure Map Bot Startup Script
# This script starts the bot with proper error handling and monitoring

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOT_SCRIPT="bot-robust.py"
MONITOR_SCRIPT="monitor_bot.py"
LOG_FILE="bot.log"
PID_FILE="bot.pid"
MONITOR_PID_FILE="monitor.pid"

echo -e "${BLUE}🚀 Starting Treasure Map Bot...${NC}"

# Function to check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed or not in PATH${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✅ Python version: $PYTHON_VERSION${NC}"
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check if required packages are installed
    if ! python3 -c "import telegram" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  python-telegram-bot not found. Installing...${NC}"
        pip3 install python-telegram-bot python-dotenv
    fi
    
    if ! python3 -c "import requests" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  requests not found. Installing...${NC}"
        pip3 install requests
    fi
    
    echo -e "${GREEN}✅ Dependencies checked${NC}"
}

# Function to stop existing bot processes
stop_existing_bot() {
    echo -e "${BLUE}🛑 Stopping existing bot processes...${NC}"
    
    # Stop bot process
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping bot process (PID: $PID)...${NC}"
            kill "$PID"
            sleep 2
        fi
        rm -f "$PID_FILE"
    fi
    
    # Stop monitor process
    if [ -f "$MONITOR_PID_FILE" ]; then
        MONITOR_PID=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$MONITOR_PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping monitor process (PID: $MONITOR_PID)...${NC}"
            kill "$MONITOR_PID"
            sleep 2
        fi
        rm -f "$MONITOR_PID_FILE"
    fi
    
    # Kill any remaining bot processes
    pkill -f "$BOT_SCRIPT" 2>/dev/null || true
    pkill -f "$MONITOR_SCRIPT" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Existing processes stopped${NC}"
}

# Function to start the bot
start_bot() {
    echo -e "${BLUE}🤖 Starting bot...${NC}"
    
    # Check if bot script exists
    if [ ! -f "$BOT_SCRIPT" ]; then
        echo -e "${RED}❌ Bot script not found: $BOT_SCRIPT${NC}"
        exit 1
    fi
    
    # Start bot in background
    nohup python3 "$BOT_SCRIPT" > "$LOG_FILE" 2>&1 &
    BOT_PID=$!
    
    # Save PID
    echo "$BOT_PID" > "$PID_FILE"
    
    echo -e "${GREEN}✅ Bot started with PID: $BOT_PID${NC}"
    
    # Wait a moment for bot to initialize
    sleep 3
    
    # Check if bot is running
    if kill -0 "$BOT_PID" 2>/dev/null; then
        echo -e "${GREEN}✅ Bot is running successfully${NC}"
    else
        echo -e "${RED}❌ Bot failed to start${NC}"
        exit 1
    fi
}

# Function to start monitoring
start_monitoring() {
    echo -e "${BLUE}📊 Starting bot monitoring...${NC}"
    
    # Check if monitor script exists
    if [ ! -f "$MONITOR_SCRIPT" ]; then
        echo -e "${YELLOW}⚠️  Monitor script not found. Skipping monitoring.${NC}"
        return
    fi
    
    # Start monitor in background
    nohup python3 "$MONITOR_SCRIPT" > "monitor.log" 2>&1 &
    MONITOR_PID=$!
    
    # Save monitor PID
    echo "$MONITOR_PID" > "$MONITOR_PID_FILE"
    
    echo -e "${GREEN}✅ Monitor started with PID: $MONITOR_PID${NC}"
}

# Function to show status
show_status() {
    echo -e "${BLUE}📋 Bot Status:${NC}"
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${GREEN}✅ Bot is running (PID: $PID)${NC}"
        else
            echo -e "${RED}❌ Bot process not running${NC}"
        fi
    else
        echo -e "${RED}❌ No bot PID file found${NC}"
    fi
    
    if [ -f "$MONITOR_PID_FILE" ]; then
        MONITOR_PID=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$MONITOR_PID" 2>/dev/null; then
            echo -e "${GREEN}✅ Monitor is running (PID: $MONITOR_PID)${NC}"
        else
            echo -e "${RED}❌ Monitor process not running${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  No monitor PID file found${NC}"
    fi
    
    # Show recent log entries
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}📝 Recent bot logs:${NC}"
        tail -5 "$LOG_FILE" | while read line; do
            echo -e "  $line"
        done
    fi
}

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    stop_existing_bot
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo -e "${BLUE}🎯 Treasure Map Bot Startup Script${NC}"
    echo -e "${BLUE}================================${NC}"
    
    # Check Python
    check_python
    
    # Check dependencies
    check_dependencies
    
    # Stop existing processes
    stop_existing_bot
    
    # Start bot
    start_bot
    
    # Start monitoring
    start_monitoring
    
    # Show status
    echo ""
    show_status
    
    echo ""
    echo -e "${GREEN}🎉 Bot startup complete!${NC}"
    echo -e "${BLUE}📖 Use 'tail -f $LOG_FILE' to monitor logs${NC}"
    echo -e "${BLUE}🛑 Use './stop_bot.sh' to stop the bot${NC}"
    echo -e "${BLUE}📊 Use './status_bot.sh' to check status${NC}"
    
    # Keep script running to maintain processes
    echo -e "${YELLOW}⏳ Press Ctrl+C to stop the bot${NC}"
    
    # Wait for interrupt
    while true; do
        sleep 10
        
        # Check if bot is still running
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "${RED}❌ Bot process died unexpectedly${NC}"
                cleanup
            fi
        fi
    done
}

# Run main function
main "$@"