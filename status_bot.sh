#!/bin/bash

# Treasure Map Bot Status Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BOT_SCRIPT="bot-robust.py"
MONITOR_SCRIPT="monitor_bot.py"
LOG_FILE="bot.log"
MONITOR_LOG_FILE="monitor.log"
PID_FILE="bot.pid"
MONITOR_PID_FILE="monitor.pid"
BOT_TOKEN="8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY"

echo -e "${BLUE}📊 Treasure Map Bot Status Report${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to check process status
check_process_status() {
    local name="$1"
    local pid_file="$2"
    local script="$3"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${GREEN}✅ $name is running (PID: $pid)${NC}"
            
            # Get process info
            local process_info=$(ps -p "$pid" -o pid,ppid,cmd,etime,pcpu,pmem --no-headers 2>/dev/null)
            if [ -n "$process_info" ]; then
                echo -e "${CYAN}  📋 Process Info: $process_info${NC}"
            fi
            
            return 0
        else
            echo -e "${RED}❌ $name PID file exists but process not running${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  $name PID file not found${NC}"
        return 1
    fi
}

# Function to check Telegram API
check_telegram_api() {
    echo -e "${BLUE}🔍 Checking Telegram API...${NC}"
    
    if command -v curl >/dev/null 2>&1; then
        local response=$(curl -s "https://api.telegram.org/bot$BOT_TOKEN/getMe" 2>/dev/null)
        if [ -n "$response" ]; then
            local username=$(echo "$response" | grep -o '"username":"[^"]*"' | cut -d'"' -f4)
            if [ -n "$username" ]; then
                echo -e "${GREEN}✅ Telegram API responding - Bot: @$username${NC}"
                return 0
            else
                echo -e "${RED}❌ Telegram API error: $response${NC}"
                return 1
            fi
        else
            echo -e "${RED}❌ Telegram API not responding${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  curl not available, skipping API check${NC}"
        return 1
    fi
}

# Function to check log files
check_log_files() {
    echo -e "${BLUE}📝 Checking log files...${NC}"
    
    # Check bot log
    if [ -f "$LOG_FILE" ]; then
        local log_size=$(du -h "$LOG_FILE" | cut -f1)
        local log_lines=$(wc -l < "$LOG_FILE")
        local last_modified=$(stat -c %y "$LOG_FILE" 2>/dev/null || stat -f %Sm "$LOG_FILE" 2>/dev/null)
        
        echo -e "${GREEN}✅ Bot log: $LOG_FILE${NC}"
        echo -e "${CYAN}  📏 Size: $log_size, Lines: $log_lines${NC}"
        echo -e "${CYAN}  🕒 Last modified: $last_modified${NC}"
        
        # Show recent entries
        if [ $log_lines -gt 0 ]; then
            echo -e "${CYAN}  📖 Recent entries:${NC}"
            tail -3 "$LOG_FILE" | while read line; do
                echo -e "    $line"
            done
        fi
    else
        echo -e "${RED}❌ Bot log file not found: $LOG_FILE${NC}"
    fi
    
    # Check monitor log
    if [ -f "$MONITOR_LOG_FILE" ]; then
        local monitor_log_size=$(du -h "$MONITOR_LOG_FILE" | cut -f1)
        local monitor_log_lines=$(wc -l < "$MONITOR_LOG_FILE")
        local monitor_last_modified=$(stat -c %y "$MONITOR_LOG_FILE" 2>/dev/null || stat -f %Sm "$MONITOR_LOG_FILE" 2>/dev/null)
        
        echo -e "${GREEN}✅ Monitor log: $MONITOR_LOG_FILE${NC}"
        echo -e "${CYAN}  📏 Size: $monitor_log_size, Lines: $monitor_log_lines${NC}"
        echo -e "${CYAN}  🕒 Last modified: $monitor_last_modified${NC}"
        
        # Show recent entries
        if [ $monitor_log_lines -gt 0 ]; then
            echo -e "${CYAN}  📖 Recent entries:${NC}"
            tail -3 "$MONITOR_LOG_FILE" | while read line; do
                echo -e "    $line"
            done
        fi
    else
        echo -e "${YELLOW}⚠️  Monitor log file not found: $MONITOR_LOG_FILE${NC}"
    fi
}

# Function to check system resources
check_system_resources() {
    echo -e "${BLUE}💻 Checking system resources...${NC}"
    
    # Check disk space
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}')
    local disk_available=$(df -h . | tail -1 | awk '{print $4}')
    echo -e "${CYAN}  💾 Disk usage: $disk_usage (${disk_available} available)${NC}"
    
    # Check memory usage
    if command -v free >/dev/null 2>&1; then
        local memory_info=$(free -h | grep Mem)
        local memory_total=$(echo "$memory_info" | awk '{print $2}')
        local memory_used=$(echo "$memory_info" | awk '{print $3}')
        local memory_free=$(echo "$memory_info" | awk '{print $4}')
        echo -e "${CYAN}  🧠 Memory: $memory_used/$memory_total used, $memory_free free${NC}"
    fi
    
    # Check CPU load
    if command -v uptime >/dev/null 2>&1; then
        local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        echo -e "${CYAN}  ⚡ Load average: $load_avg${NC}"
    fi
}

# Function to check database
check_database() {
    echo -e "${BLUE}🗄️  Checking database...${NC}"
    
    local db_file="treasure_map.db"
    if [ -f "$db_file" ]; then
        local db_size=$(du -h "$db_file" | cut -f1)
        local db_modified=$(stat -c %y "$db_file" 2>/dev/null || stat -f %Sm "$db_file" 2>/dev/null)
        
        echo -e "${GREEN}✅ Database: $db_file${NC}"
        echo -e "${CYAN}  📏 Size: $db_size${NC}"
        echo -e "${CYAN}  🕒 Last modified: $db_modified${NC}"
        
        # Try to check database integrity
        if command -v sqlite3 >/dev/null 2>&1; then
            local table_count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
            echo -e "${CYAN}  🗃️  Tables: $table_count${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Database file not found: $db_file${NC}"
    fi
}

# Function to check uploads folder
check_uploads_folder() {
    echo -e "${BLUE}📁 Checking uploads folder...${NC}"
    
    local uploads_folder="uploads"
    if [ -d "$uploads_folder" ]; then
        local file_count=$(find "$uploads_folder" -type f | wc -l)
        local folder_size=$(du -sh "$uploads_folder" 2>/dev/null | cut -f1 || echo "unknown")
        
        echo -e "${GREEN}✅ Uploads folder: $uploads_folder${NC}"
        echo -e "${CYAN}  📏 Size: $folder_size${NC}"
        echo -e "${CYAN}  📄 Files: $file_count${NC}"
        
        # Show recent files
        if [ $file_count -gt 0 ]; then
            echo -e "${CYAN}  📖 Recent files:${NC}"
            find "$uploads_folder" -type f -printf "%T@ %p\n" | sort -nr | head -3 | while read timestamp file; do
                local filename=$(basename "$file")
                local size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "unknown")
                echo -e "    $filename ($size)"
            done
        fi
    else
        echo -e "${YELLOW}⚠️  Uploads folder not found: $uploads_folder${NC}"
    fi
}

# Function to show overall status
show_overall_status() {
    echo ""
    echo -e "${BLUE}📊 Overall Status Summary${NC}"
    echo -e "${BLUE}========================${NC}"
    
    local bot_status=0
    local monitor_status=0
    local api_status=0
    
    # Check bot process
    if check_process_status "Bot" "$PID_FILE" "$BOT_SCRIPT" >/dev/null; then
        bot_status=1
    fi
    
    # Check monitor process
    if check_process_status "Monitor" "$MONITOR_PID_FILE" "$MONITOR_SCRIPT" >/dev/null; then
        monitor_status=1
    fi
    
    # Check Telegram API
    if check_telegram_api >/dev/null; then
        api_status=1
    fi
    
    # Calculate overall health
    local total_checks=3
    local passed_checks=$((bot_status + monitor_status + api_status))
    local health_percentage=$((passed_checks * 100 / total_checks))
    
    echo ""
    if [ $health_percentage -eq 100 ]; then
        echo -e "${GREEN}🎉 Bot Status: EXCELLENT (100%)${NC}"
    elif [ $health_percentage -ge 66 ]; then
        echo -e "${GREEN}✅ Bot Status: GOOD (${health_percentage}%)${NC}"
    elif [ $health_percentage -ge 33 ]; then
        echo -e "${YELLOW}⚠️  Bot Status: FAIR (${health_percentage}%)${NC}"
    else
        echo -e "${RED}❌ Bot Status: POOR (${health_percentage}%)${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}🔧 Quick Actions:${NC}"
    echo -e "${CYAN}  🚀 Start bot: ./start_bot.sh${NC}"
    echo -e "${CYAN}  🛑 Stop bot: ./stop_bot.sh${NC}"
    echo -e "${CYAN}  📖 View logs: tail -f $LOG_FILE${NC}"
    echo -e "${CYAN}  📊 Monitor logs: tail -f $MONITOR_LOG_FILE${NC}"
}

# Main execution
main() {
    # Check bot process
    echo -e "${BLUE}🤖 Bot Process Status:${NC}"
    check_process_status "Bot" "$PID_FILE" "$BOT_SCRIPT"
    echo ""
    
    # Check monitor process
    echo -e "${BLUE}📊 Monitor Process Status:${NC}"
    check_process_status "Monitor" "$MONITOR_PID_FILE" "$MONITOR_SCRIPT"
    echo ""
    
    # Check Telegram API
    check_telegram_api
    echo ""
    
    # Check log files
    check_log_files
    echo ""
    
    # Check database
    check_database
    echo ""
    
    # Check uploads folder
    check_uploads_folder
    echo ""
    
    # Check system resources
    check_system_resources
    echo ""
    
    # Show overall status
    show_overall_status
}

# Run main function
main "$@"