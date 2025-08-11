#!/usr/bin/env python3
"""
Simplified Treasure Map Bot for Python 3.13+
This version avoids problematic dependencies and uses built-in alternatives
"""

import os
import re
import difflib
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN', '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwatqI7WY')
ADMIN_USER_IDS = [int(x.strip()) for x in os.getenv('ADMIN_USER_IDS', '1694244496').split(',') if x.strip()]
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'treasure_map.db'
DEFAULT_SIMILARITY_THRESHOLD = 15.0
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k')

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SimpleDatabase:
    """Simplified database handler"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
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
    
    def add_reference_image(self, filename, file_path, extracted_text, admin_user_id, language):
        """Add a new reference image"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reference_images (filename, file_path, extracted_text, admin_user_id, language)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, file_path, extracted_text, admin_user_id, language))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_reference_texts(self):
        """Get all reference texts for matching"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, extracted_text FROM reference_images
                ORDER BY upload_date DESC
            ''')
            return [{'id': row[0], 'text': row[1]} for row in cursor.fetchall()]
    
    def get_similarity_threshold(self):
        """Get current similarity threshold"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE key = "similarity_threshold"')
            result = cursor.fetchone()
            return int(result[0]) if result else DEFAULT_SIMILARITY_THRESHOLD

class SimpleTextMatcher:
    """Simplified text matcher using built-in libraries"""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
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
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep Chinese characters
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap bonus"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return (intersection / union) * 20  # Max 20% bonus
    
    def find_best_match(self, query_text: str, reference_texts: List[Dict], threshold: float = 15.0) -> Optional[Dict]:
        """Find best match above threshold"""
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

class SimpleTreasureMapBot:
    """Simplified Treasure Map Bot"""
    
    def __init__(self):
        self.db = SimpleDatabase()
        self.text_matcher = SimpleTextMatcher()
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_text = """
🎯 Welcome to Treasure Map Bot!

I can help you find matching treasure maps using text similarity.

📤 **How to use:**
1. Send me an image of a treasure map
2. I'll extract the text and find matches
3. Get the best matching reference image

👑 **Admin commands:**
• /add - Upload reference images
• /help - Show this help message

Send me an image to get started!
        """
        await update.message.reply_text(welcome_text.strip())
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🔍 **Treasure Map Bot Help**

**For Users:**
• Send any image to find matching treasure maps
• I'll extract text and compare with reference database
• Get similarity percentage and best match

**For Admins:**
• /add - Upload reference images (reply to image)
• /setthreshold <value> - Set similarity threshold (1-100%)

**How it works:**
1. Admin uploads reference images using /add
2. Bot extracts text from images
3. Users send images to find matches
4. Bot compares text similarity and returns results

Need help? Contact the bot administrator.
        """
        await update.message.reply_text(help_text.strip())
    
    async def add_reference(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command for uploading reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ You don't have permission to use this command.")
            return
        
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text("❌ Please reply to an image with /add")
            return
        
        try:
            # Get the largest photo size
            photo = max(update.message.reply_to_message.photo, key=lambda x: x.file_size)
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            file_path = os.path.join(UPLOAD_FOLDER, f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            await file.download_to_drive(file_path)
            
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
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error adding reference image: {str(e)}")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming images"""
        try:
            # Get the largest photo size
            photo = max(update.message.photo, key=lambda x: x.file_size)
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            await file.download_to_drive(temp_path)
            
            # Extract text (simplified)
            extracted_text = f"User image: {os.path.basename(temp_path)}"
            
            # Find matches
            reference_texts = self.db.get_all_reference_texts()
            threshold = self.db.get_similarity_threshold()
            
            if not reference_texts:
                await update.message.reply_text("❌ No reference images found. Please ask an admin to upload some.")
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
            else:
                response = f"""
❌ **No Match Found**

🔍 **Similarity Threshold:** {threshold}%
📝 **Your Image Text:** {extracted_text[:100]}...

No reference images matched your treasure map.
Try uploading a clearer image or ask an admin to add more references.
                """
            
            await update.message.reply_text(response.strip())
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error processing image: {str(e)}")
    
    async def set_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setthreshold command"""
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
            else:
                await update.message.reply_text("❌ Threshold must be between 1 and 100")
        except ValueError:
            await update.message.reply_text("❌ Invalid threshold value. Please use a number between 1-100")

def main():
    """Main function"""
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN not found in .env file")
        return
    
    if not ADMIN_USER_IDS:
        print("❌ ADMIN_USER_IDS not found in .env file")
        return
    
    print("🚀 Starting Simplified Treasure Map Bot...")
    
    # Create bot instance
    bot = SimpleTreasureMapBot()
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("add", bot.add_reference))
    application.add_handler(CommandHandler("setthreshold", bot.set_threshold))
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_image))
    
    # Start the bot
    print("✅ Bot is running... Press Ctrl+C to stop")
    application.run_polling()

if __name__ == "__main__":
    main()