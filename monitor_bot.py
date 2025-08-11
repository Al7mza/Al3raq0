#!/usr/bin/env python3
"""
Bot Monitoring Script
Checks bot health and restarts if needed
"""

import os
import time
import subprocess
import logging
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = "8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY"
CHECK_INTERVAL = 60  # Check every 60 seconds
MAX_RESTARTS = 5  # Maximum restarts per hour
RESTART_COOLDOWN = 3600  # 1 hour cooldown between restart attempts

class BotMonitor:
    def __init__(self):
        self.last_restart = 0
        self.restart_count = 0
        self.hour_start = time.time()
    
    def check_bot_health(self):
        """Check if bot is responding to Telegram API"""
        try:
            # Try to get bot info from Telegram API
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    logger.info(f"Bot is healthy: {data['result']['username']}")
                    return True
                else:
                    logger.warning(f"Bot API error: {data}")
                    return False
            else:
                logger.error(f"HTTP error: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking bot: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking bot: {e}")
            return False
    
    def check_bot_process(self):
        """Check if bot process is running"""
        try:
            # Check if bot process is running
            result = subprocess.run(
                ['pgrep', '-f', 'bot-robust.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                logger.info(f"Bot process running with PIDs: {pids}")
                return True
            else:
                logger.warning("Bot process not found")
                return False
                
        except Exception as e:
            logger.error(f"Error checking bot process: {e}")
            return False
    
    def check_log_file(self):
        """Check if bot log file is being updated"""
        try:
            log_file = 'bot.log'
            if not os.path.exists(log_file):
                logger.warning("Bot log file not found")
                return False
            
            # Check if log file was updated in last 5 minutes
            stat = os.stat(log_file)
            if time.time() - stat.st_mtime > 300:  # 5 minutes
                logger.warning("Bot log file not updated recently")
                return False
            
            logger.info("Bot log file is being updated")
            return True
            
        except Exception as e:
            logger.error(f"Error checking log file: {e}")
            return False
    
    def restart_bot(self):
        """Restart the bot"""
        current_time = time.time()
        
        # Check restart limits
        if current_time - self.hour_start > 3600:  # New hour
            self.restart_count = 0
            self.hour_start = current_time
        
        if self.restart_count >= MAX_RESTARTS:
            logger.error(f"Maximum restarts ({MAX_RESTARTS}) reached this hour. Waiting for cooldown.")
            return False
        
        if current_time - self.last_restart < RESTART_COOLDOWN:
            logger.info("Restart cooldown active. Skipping restart.")
            return False
        
        try:
            logger.info("Restarting bot...")
            
            # Stop existing bot process
            subprocess.run(['pkill', '-f', 'bot-robust.py'], capture_output=True)
            time.sleep(5)  # Wait for process to stop
            
            # Start bot again
            subprocess.Popen([
                '/usr/local/bin/python3', 
                '/workspace/bot-robust.py'
            ], cwd='/workspace')
            
            self.last_restart = current_time
            self.restart_count += 1
            
            logger.info(f"Bot restarted. Restart count this hour: {self.restart_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error restarting bot: {e}")
            return False
    
    def run_monitoring(self):
        """Main monitoring loop"""
        logger.info("Starting bot monitoring...")
        
        while True:
            try:
                # Check bot health
                bot_healthy = self.check_bot_health()
                process_running = self.check_bot_process()
                log_updating = self.check_log_file()
                
                # Determine overall health
                overall_healthy = bot_healthy and process_running and log_updating
                
                if overall_healthy:
                    logger.info("✅ Bot is healthy and running")
                else:
                    logger.warning("⚠️ Bot health issues detected:")
                    if not bot_healthy:
                        logger.warning("  - Telegram API not responding")
                    if not process_running:
                        logger.warning("  - Bot process not running")
                    if not log_updating:
                        logger.warning("  - Log file not updating")
                    
                    # Attempt restart
                    if self.restart_bot():
                        logger.info("Bot restart initiated")
                    else:
                        logger.error("Bot restart failed or skipped")
                
                # Wait before next check
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitoring: {e}")
                time.sleep(CHECK_INTERVAL)

def main():
    monitor = BotMonitor()
    monitor.run_monitoring()

if __name__ == "__main__":
    main()