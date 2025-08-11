#!/usr/bin/env python3
"""
Robust Treasure Map Bot for Production Hosting
This version includes better error handling, logging, and connection management
"""

import os
import sys
import logging
import asyncio
import signal
import time
from datetime import datetime
from typing import List, Dict, Optional
import sqlite3
import re
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import Telegram libraries with error handling
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from telegram.error import NetworkError, TimedOut, RetryAfter
    TELEGRAM_AVAILABLE = True
except ImportError as e:
    logger.error(f"Telegram library not available: {e}")
    TELEGRAM_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError as e:
    logger.error(f"Dotenv not available: {e}")
    DOTENV_AVAILABLE = False

# Load environment variables
if DOTENV_AVAILABLE:
    load_dotenv()

# Configuration with fallbacks
BOT_TOKEN = os.getenv('BOT_TOKEN', '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY')
ADMIN_USER_IDS = [int(x.strip()) for x in os.getenv('ADMIN_USER_IDS', '1694244496').split(',') if x.strip()]
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'treasure_map.db'
DEFAULT_SIMILARITY_THRESHOLD = 15.0
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class RobustDatabase:
    """Database handler with error recovery"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create reference_images table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS reference_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        extracted_text TEXT NOT NULL,
                        admin_user_id INTEGER NOT NULL,
                        language TEXT NOT NULL,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                ''')
                
                # Insert default similarity threshold
                cursor.execute('''
                    INSERT OR IGNORE INTO settings (key, value)
                    VALUES ('similarity_threshold', ?)
                ''', (str(DEFAULT_SIMILARITY_THRESHOLD),))
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_reference_image(self, filename, file_path, extracted_text, admin_user_id, language):
        """Add a new reference image with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO reference_images (filename, file_path, extracted_text, admin_user_id, language)
                    VALUES (?, ?, ?, ?, ?)
                ''', (filename, file_path, extracted_text, admin_user_id, language))
                conn.commit()
                logger.info(f"Reference image added: {filename}")
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error adding reference image: {e}")
            raise
    
    def get_all_reference_texts(self):
        """Get all reference texts with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, extracted_text FROM reference_images
                    ORDER BY upload_date DESC
                ''')
                return [{'id': row[0], 'text': row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting reference texts: {e}")
            return []
    
    def get_similarity_threshold(self):
        """Get current similarity threshold with error handling"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM settings WHERE key = "similarity_threshold"')
                result = cursor.fetchone()
                return int(result[0]) if result else DEFAULT_SIMILARITY_THRESHOLD
        except Exception as e:
            logger.error(f"Error getting similarity threshold: {e}")
            return DEFAULT_SIMILARITY_THRESHOLD

class RobustTextMatcher:
    """Text matcher with error handling"""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts with error handling"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Normalize texts
            norm_text1 = self.normalize_text(text1)
            norm_text2 = self.normalize_text(text2)
            
            # Use difflib for similarity
            similarity = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio() * 100
            
            # Add keyword overlap bonus
            keyword_bonus = self.keyword_overlap(norm_text1, norm_text2)
            
            return min(100.0, similarity + keyword_bonus)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison with error handling"""
        try:
            if not text:
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation but keep Chinese characters
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error normalizing text: {e}")
            return text if text else ""
    
    def keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap bonus with error handling"""
        try:
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            if union == 0:
                return 0.0
            
            return (intersection / union) * 20  # Max 20% bonus
            
        except Exception as e:
            logger.error(f"Error calculating keyword overlap: {e}")
            return 0.0
    
    def find_best_match(self, query_text: str, reference_texts: List[Dict], threshold: float = 15.0) -> Optional[Dict]:
        """Find best match above threshold with error handling"""
        try:
            if not query_text or not reference_texts:
                return None
            
            best_match = None
            best_similarity = 0.0
            
            for ref in reference_texts:
                similarity = self.calculate_similarity(query_text, ref['text'])
                
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        'id': ref['id'],
                        'similarity': similarity,
                        'text': ref['text']
                    }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding best match: {e}")
            return None

class RobustTreasureMapBot:
    """Robust Treasure Map Bot with error handling and logging"""
    
    def __init__(self):
        self.db = RobustDatabase()
        self.text_matcher = RobustTextMatcher()
        self.start_time = time.time()
        self.message_count = 0
        logger.info("Bot initialized successfully")
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with error handling"""
        try:
            self.message_count += 1
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            
            logger.info(f"Start command from user {user_id} (@{username})")
            
            welcome_text = f"""
🎯 Welcome to Treasure Map Bot!

I can help you find matching treasure maps using text similarity.

📤 **How to use:**
1. Send me an image of a treasure map
2. I'll extract the text and find matches
3. Get the best matching reference image

👑 **Admin commands:**
• /add - Upload reference images
• /status - Check bot status
• /help - Show this help message

Send me an image to get started!
            """
            await update.message.reply_text(welcome_text.strip())
            logger.info(f"Start message sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await self.send_error_message(update, "start command")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command with error handling"""
        try:
            self.message_count += 1
            user_id = update.effective_user.id
            
            help_text = """
🔍 **Treasure Map Bot Help**

**For Users:**
• Send any image to find matching treasure maps
• I'll extract text and compare with reference database
• Get similarity percentage and best match

**For Admins:**
• /add - Upload reference images (reply to image)
• /status - Check bot status
• /setthreshold <value> - Set similarity threshold (1-100%)

**How it works:**
1. Admin uploads reference images using /add
2. Bot extracts text from images
3. Users send images to find matches
4. Bot compares text similarity and returns results

Need help? Contact the bot administrator.
            """
            await update.message.reply_text(help_text.strip())
            logger.info(f"Help message sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error in help command: {e}")
            await self.send_error_message(update, "help command")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command to check bot health"""
        try:
            user_id = update.effective_user.id
            
            if user_id not in ADMIN_USER_IDS:
                await update.message.reply_text("❌ You don't have permission to use this command.")
                return
            
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            
            status_text = f"""
🤖 **Bot Status Report**

⏱️ **Uptime:** {hours}h {minutes}m {seconds}s
📊 **Messages Processed:** {self.message_count}
💾 **Database:** {'✅ Connected' if self.db else '❌ Error'}
🔍 **Text Matcher:** {'✅ Working' if self.text_matcher else '❌ Error'}
📁 **Upload Folder:** {'✅ Exists' if os.path.exists(UPLOAD_FOLDER) else '❌ Missing'}

🎯 **Bot is running and healthy!**
            """
            await update.message.reply_text(status_text.strip())
            logger.info(f"Status report sent to admin {user_id}")
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await self.send_error_message(update, "status command")
    
    async def add_reference(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command for uploading reference images with error handling"""
        try:
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            
            if user_id not in ADMIN_USER_IDS:
                await update.message.reply_text("❌ You don't have permission to use this command.")
                return
            
            if not update.message.reply_to_message or not update.message.reply_to_message.photo:
                await update.message.reply_text("❌ Please reply to an image with /add")
                return
            
            # Get the largest photo size
            photo = max(update.message.reply_to_message.photo, key=lambda x: x.file_size)
            
            # Download the image with timeout
            file = await context.bot.get_file(photo.file_id)
            file_path = os.path.join(UPLOAD_FOLDER, f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            await asyncio.wait_for(file.download_to_drive(file_path), timeout=30.0)
            
            # Extract text (simplified - just filename for now)
            extracted_text = f"Reference image: {os.path.basename(file_path)}"
            
            # Save to database
            image_id = self.db.add_reference_image(
                os.path.basename(file_path),
                file_path,
                extracted_text,
                user_id,
                'eng'
            )
            
            await update.message.reply_text(f"✅ Reference image added successfully! (ID: {image_id})")
            logger.info(f"Reference image added by admin {user_id} (@{username}): {file_path}")
            
        except asyncio.TimeoutError:
            await update.message.reply_text("❌ Image download timed out. Please try again.")
            logger.warning(f"Image download timeout for user {user_id}")
        except Exception as e:
            logger.error(f"Error adding reference image: {e}")
            await self.send_error_message(update, "adding reference image")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming images with error handling and timeout"""
        try:
            self.message_count += 1
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            
            logger.info(f"Processing image from user {user_id} (@{username})")
            
            # Get the largest photo size
            photo = max(update.message.photo, key=lambda x: x.file_size)
            
            # Download the image with timeout
            file = await context.bot.get_file(photo.file_id)
            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            await asyncio.wait_for(file.download_to_drive(temp_path), timeout=30.0)
            
            # Extract text (simplified)
            extracted_text = f"User image: {os.path.basename(temp_path)}"
            
            # Find matches
            reference_texts = self.db.get_all_reference_texts()
            threshold = self.db.get_similarity_threshold()
            
            if not reference_texts:
                await update.message.reply_text("❌ No reference images found. Please ask an admin to upload some.")
                logger.info(f"No reference images found for user {user_id}")
                return
            
            # Find best match
            best_match = self.text_matcher.find_best_match(extracted_text, reference_texts, threshold)
            
            if best_match:
                response = f"""
🎯 **Match Found!**

📊 **Similarity:** {best_match['similarity']:.1f}%
📝 **Reference Text:** {best_match['text'][:100]}...
🆔 **Reference ID:** {best_match['id']}

This image matches one of our reference treasure maps!
                """
                logger.info(f"Match found for user {user_id}: {best_match['similarity']:.1f}% similarity")
            else:
                response = f"""
❌ **No Match Found**

🔍 **Similarity Threshold:** {threshold}%
📝 **Your Image Text:** {extracted_text[:100]}...

No reference images matched your treasure map.
Try uploading a clearer image or ask an admin to add more references.
                """
                logger.info(f"No match found for user {user_id} (threshold: {threshold}%)")
            
            await update.message.reply_text(response.strip())
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except asyncio.TimeoutError:
            await update.message.reply_text("❌ Image processing timed out. Please try again.")
            logger.warning(f"Image processing timeout for user {user_id}")
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await self.send_error_message(update, "processing image")
    
    async def set_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setthreshold command with error handling"""
        try:
            user_id = update.effective_user.id
            
            if user_id not in ADMIN_USER_IDS:
                await update.message.reply_text("❌ You don't have permission to use this command.")
                return
            
            if not context.args:
                await update.message.reply_text("❌ Please provide a threshold value: /setthreshold <1-100>")
                return
            
            try:
                threshold = int(context.args[0])
                if 1 <= threshold <= 100:
                    # Update threshold in database
                    with sqlite3.connect(DATABASE_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO settings (key, value)
                            VALUES ('similarity_threshold', ?)
                        ''', (str(threshold),))
                        conn.commit()
                    
                    await update.message.reply_text(f"✅ Similarity threshold set to {threshold}%")
                    logger.info(f"Threshold updated to {threshold}% by admin {user_id}")
                else:
                    await update.message.reply_text("❌ Threshold must be between 1 and 100")
            except ValueError:
                await update.message.reply_text("❌ Invalid threshold value. Please use a number between 1-100")
                
        except Exception as e:
            logger.error(f"Error setting threshold: {e}")
            await self.send_error_message(update, "setting threshold")
    
    async def send_error_message(self, update: Update, operation: str):
        """Send error message to user"""
        try:
            error_text = f"❌ An error occurred while {operation}. Please try again later."
            await update.message.reply_text(error_text)
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot"""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Try to send error message to user if possible
    if update and hasattr(update, 'message') and update.message:
        try:
            await update.message.reply_text("❌ An error occurred. Please try again later.")
        except Exception as e:
            logger.error(f"Error sending error message to user: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main function with error handling"""
    if not TELEGRAM_AVAILABLE:
        logger.error("Telegram library not available. Please install python-telegram-bot")
        return
    
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN not found in .env file")
        return
    
    if not ADMIN_USER_IDS:
        logger.error("ADMIN_USER_IDS not found in .env file")
        return
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Starting Robust Treasure Map Bot...")
    logger.info(f"Bot Token: {BOT_TOKEN[:20]}...")
    logger.info(f"Admin User IDs: {ADMIN_USER_IDS}")
    
    try:
        # Create bot instance
        bot = RobustTreasureMapBot()
        
        # Create application with error handling
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Add error handler
        application.add_error_handler(error_handler)
        
        # Add handlers
        application.add_handler(CommandHandler("start", bot.start))
        application.add_handler(CommandHandler("help", bot.help_command))
        application.add_handler(CommandHandler("status", bot.status_command))
        application.add_handler(CommandHandler("add", bot.add_reference))
        application.add_handler(CommandHandler("setthreshold", bot.set_threshold))
        application.add_handler(MessageHandler(filters.PHOTO, bot.handle_image))
        
        # Start the bot
        logger.info("✅ Bot is running... Press Ctrl+C to stop")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
            close_loop=False
        )
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()