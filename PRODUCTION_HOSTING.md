# 🚀 Production Hosting Guide for Treasure Map Bot

## ⚠️ **Problem Solved: Bot Hanging After 15+ Minutes**

This guide addresses the issue where your bot appears to be running but doesn't respond properly, showing "Silly Development" messages and consuming resources without working.

## 🔧 **Root Causes & Solutions**

### **Problem 1: Bot Process Hanging**
- **Cause**: Bot gets stuck in infinite loops or network timeouts
- **Solution**: Robust error handling with timeouts and automatic restarts

### **Problem 2: Memory Leaks**
- **Cause**: Resources not properly cleaned up
- **Solution**: Proper cleanup, garbage collection, and resource limits

### **Problem 3: Network Issues**
- **Cause**: Telegram API connection problems
- **Solution**: Connection monitoring and automatic reconnection

### **Problem 4: Process Management**
- **Cause**: No proper process supervision
- **Solution**: Systemd service and monitoring scripts

## 🎯 **Quick Fix: Use the Robust Bot**

### **Step 1: Install Dependencies**
```bash
# Install only essential packages (avoids build issues)
pip3 install python-telegram-bot python-dotenv requests

# Or use the robust requirements file
pip3 install -r requirements-robust.txt
```

### **Step 2: Start the Robust Bot**
```bash
# Make scripts executable
chmod +x *.sh

# Start the bot with monitoring
./start_bot.sh
```

### **Step 3: Check Status**
```bash
# Check if bot is working
./status_bot.sh

# View live logs
tail -f bot.log
```

## 🏗️ **Production Setup Options**

### **Option 1: Simple Script Management (Recommended)**
```bash
# Start bot
./start_bot.sh

# Check status
./status_bot.sh

# Stop bot
./stop_bot.sh
```

### **Option 2: Systemd Service**
```bash
# Copy service file
sudo cp treasure-map-bot.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable treasure-map-bot
sudo systemctl start treasure-map-bot

# Check status
sudo systemctl status treasure-map-bot

# View logs
sudo journalctl -u treasure-map-bot -f
```

### **Option 3: Screen/Tmux Session**
```bash
# Create screen session
screen -S treasure-bot

# Start bot
python3 bot-robust.py

# Detach: Ctrl+A, then D
# Reattach: screen -r treasure-bot
```

## 📊 **Monitoring & Troubleshooting**

### **Real-time Monitoring**
```bash
# Watch bot logs
tail -f bot.log

# Watch monitor logs
tail -f monitor.log

# Check process status
ps aux | grep bot-robust
```

### **Health Checks**
```bash
# Check bot health
./status_bot.sh

# Test Telegram API
curl "https://api.telegram.org/bot$BOT_TOKEN/getMe"

# Check system resources
htop
df -h
free -h
```

### **Common Issues & Fixes**

#### **Issue: Bot Not Responding**
```bash
# Check if process is running
./status_bot.sh

# Restart if needed
./stop_bot.sh
./start_bot.sh
```

#### **Issue: High Memory Usage**
```bash
# Check memory usage
ps aux | grep bot-robust

# Restart bot to clear memory
./stop_bot.sh && sleep 5 && ./start_bot.sh
```

#### **Issue: Log Files Too Large**
```bash
# Rotate logs
mv bot.log bot.log.old
mv monitor.log monitor.log.old

# Restart bot
./stop_bot.sh && ./start_bot.sh
```

## 🔄 **Automatic Restart Strategies**

### **1. Monitor Script (Built-in)**
- Checks bot health every 60 seconds
- Automatically restarts if bot fails
- Limits restarts to 5 per hour
- Logs all actions

### **2. Systemd Service**
- Automatic restart on failure
- Restart delay: 10 seconds
- Maximum restarts: unlimited
- Proper logging to journal

### **3. Cron Job (Alternative)**
```bash
# Add to crontab: check every 5 minutes
*/5 * * * * /workspace/status_bot.sh | grep -q "Bot Status: POOR" && /workspace/stop_bot.sh && sleep 10 && /workspace/start_bot.sh
```

## 📈 **Performance Optimization**

### **Resource Limits**
```bash
# Set memory limit (512MB)
ulimit -v 524288

# Set CPU limit (50%)
# Already configured in systemd service
```

### **Database Optimization**
```bash
# Optimize SQLite database
sqlite3 treasure_map.db "VACUUM;"
sqlite3 treasure_map.db "ANALYZE;"
```

### **Log Rotation**
```bash
# Create logrotate config
cat > /etc/logrotate.d/treasure-bot << EOF
/workspace/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF
```

## 🚨 **Emergency Procedures**

### **Bot Completely Unresponsive**
```bash
# Force stop all processes
pkill -9 -f bot-robust
pkill -9 -f monitor_bot

# Clear PID files
rm -f *.pid

# Restart fresh
./start_bot.sh
```

### **High Resource Usage**
```bash
# Check what's consuming resources
top -p $(pgrep -f bot-robust)

# Kill problematic processes
kill -9 $(pgrep -f bot-robust)

# Restart with monitoring
./start_bot.sh
```

### **Database Corruption**
```bash
# Backup current database
cp treasure_map.db treasure_map.db.backup

# Remove corrupted database
rm treasure_map.db

# Restart bot (will create new database)
./stop_bot.sh && ./start_bot.sh
```

## 📋 **Daily Maintenance**

### **Morning Check**
```bash
# Check overnight status
./status_bot.sh

# Review logs for errors
grep -i error bot.log | tail -10

# Check system resources
df -h && free -h
```

### **Weekly Maintenance**
```bash
# Rotate logs
./stop_bot.sh
mv *.log *.log.old
./start_bot.sh

# Optimize database
sqlite3 treasure_map.db "VACUUM; ANALYZE;"

# Check for updates
pip3 list --outdated
```

## 🎯 **Success Indicators**

### **✅ Bot is Working Properly When:**
- `/status` command responds within 2 seconds
- Images are processed within 30 seconds
- Log file shows regular activity
- Memory usage stays under 100MB
- CPU usage stays under 10%

### **❌ Bot Needs Attention When:**
- Commands don't respond
- Log file stops updating
- Memory usage exceeds 200MB
- CPU usage spikes above 50%
- Error messages in logs

## 🔗 **Quick Reference Commands**

```bash
# Start bot
./start_bot.sh

# Stop bot
./stop_bot.sh

# Check status
./status_bot.sh

# View logs
tail -f bot.log

# Monitor logs
tail -f monitor.log

# Restart bot
./stop_bot.sh && sleep 5 && ./start_bot.sh

# Emergency restart
pkill -9 -f bot-robust && ./start_bot.sh
```

## 🎉 **You're All Set!**

With these improvements, your bot should:
- ✅ **Run continuously without hanging**
- ✅ **Automatically restart on failures**
- ✅ **Provide clear status information**
- ✅ **Use minimal system resources**
- ✅ **Log all activities for debugging**

**Start with `./start_bot.sh` and use `./status_bot.sh` to monitor!** 🚀