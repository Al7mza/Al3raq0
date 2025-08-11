#!/bin/bash

# Treasure Map Bot Stop Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BOT_SCRIPT="bot-robust.py"
MONITOR_SCRIPT="monitor_bot.py"
PID_FILE="bot.pid"
MONITOR_PID_FILE="monitor.pid"

echo -e "${BLUE}🛑 Stopping Treasure Map Bot...${NC}"

# Function to stop processes
stop_processes() {
    # Stop bot process
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping bot process (PID: $PID)...${NC}"
            kill "$PID"
            sleep 2
            
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${YELLOW}Force killing bot process...${NC}"
                kill -9 "$PID"
            fi
        fi
        rm -f "$PID_FILE"
        echo -e "${GREEN}✅ Bot process stopped${NC}"
    else
        echo -e "${YELLOW}No bot PID file found${NC}"
    fi
    
    # Stop monitor process
    if [ -f "$MONITOR_PID_FILE" ]; then
        MONITOR_PID=$(cat "$MONITOR_PID_FILE")
        if kill -0 "$MONITOR_PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping monitor process (PID: $MONITOR_PID)...${NC}"
            kill "$MONITOR_PID"
            sleep 2
            
            # Force kill if still running
            if kill -0 "$MONITOR_PID" 2>/dev/null; then
                echo -e "${YELLOW}Force killing monitor process...${NC}"
                kill -9 "$MONITOR_PID"
            fi
        fi
        rm -f "$MONITOR_PID_FILE"
        echo -e "${GREEN}✅ Monitor process stopped${NC}"
    else
        echo -e "${YELLOW}No monitor PID file found${NC}"
    fi
    
    # Kill any remaining processes
    echo -e "${BLUE}Checking for remaining processes...${NC}"
    
    # Kill bot processes
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        echo -e "${YELLOW}Killing remaining bot processes...${NC}"
        pkill -f "$BOT_SCRIPT"
        sleep 1
        pkill -9 -f "$BOT_SCRIPT" 2>/dev/null || true
    fi
    
    # Kill monitor processes
    if pgrep -f "$MONITOR_SCRIPT" > /dev/null; then
        echo -e "${YELLOW}Killing remaining monitor processes...${NC}"
        pkill -f "$MONITOR_SCRIPT"
        sleep 1
        pkill -9 -f "$MONITOR_SCRIPT" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ All processes stopped${NC}"
}

# Function to show final status
show_final_status() {
    echo -e "${BLUE}📋 Final Status Check:${NC}"
    
    # Check if any bot processes are still running
    if pgrep -f "$BOT_SCRIPT" > /dev/null; then
        echo -e "${RED}❌ Bot processes still running:${NC}"
        pgrep -f "$BOT_SCRIPT" | while read pid; do
            echo -e "  PID: $pid"
        done
    else
        echo -e "${GREEN}✅ No bot processes running${NC}"
    fi
    
    # Check if any monitor processes are still running
    if pgrep -f "$MONITOR_SCRIPT" > /dev/null; then
        echo -e "${RED}❌ Monitor processes still running:${NC}"
        pgrep -f "$MONITOR_SCRIPT" | while read pid; do
            echo -e "  PID: $pid"
        done
    else
        echo -e "${GREEN}✅ No monitor processes running${NC}"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}🎯 Treasure Map Bot Stop Script${NC}"
    echo -e "${BLUE}==============================${NC}"
    
    # Stop all processes
    stop_processes
    
    # Show final status
    echo ""
    show_final_status
    
    echo ""
    echo -e "${GREEN}🎉 Bot stopped successfully!${NC}"
    echo -e "${BLUE}🚀 Use './start_bot.sh' to start the bot again${NC}"
}

# Run main function
main "$@"