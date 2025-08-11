#!/usr/bin/env python3
"""
Treasure Map Bot - Complete Working Version
A Telegram bot for storing and searching treasure map images
"""
import os
import sqlite3
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY'
ADMIN_USER_IDS = [1694244496]
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'treasure_map.db'
GOOGLE_API_KEY = 'AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Database:
    """Database management class"""
    
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS treasure_maps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        username TEXT,
                        first_name TEXT,
                        file_path TEXT NOT NULL,
                        file_size INTEGER,
                        text_content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        query TEXT NOT NULL,
                        results_count INTEGER,
                        searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def save_treasure_map(self, user_id: int, username: str, first_name: str, 
                          file_path: str, file_size: int, text_content: str) -> bool:
        """Save a treasure map to database"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO treasure_maps (user_id, username, first_name, file_path, file_size, text_content)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, username, first_name, file_path, file_size, text_content))
                conn.commit()
                logger.info(f"Saved treasure map for user {user_id}")
                return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False
    
    def get_all_treasure_maps(self) -> List[tuple]:
        """Get all treasure maps"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM treasure_maps 
                    ORDER BY created_at DESC
                ''')
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return []
    
    def search_treasure_maps(self, query: str) -> List[tuple]:
        """Search treasure maps by text content"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM treasure_maps 
                    WHERE text_content LIKE ? OR username LIKE ? OR first_name LIKE ?
                    ORDER BY created_at DESC
                ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_user_maps(self, user_id: int) -> List[tuple]:
        """Get maps uploaded by specific user"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM treasure_maps 
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                ''', (user_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"User maps fetch error: {e}")
            return []
    
    def save_search_history(self, user_id: int, query: str, results_count: int) -> bool:
        """Save search history"""
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO search_history (user_id, query, results_count)
                    VALUES (?, ?, ?)
                ''', (user_id, query, results_count))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Search history save error: {e}")
            return False

class TextMatcher:
    """Text matching and similarity calculation"""
    
    def __init__(self):
        self.similarity_threshold = 15.0
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using difflib"""
        try:
            import difflib
            similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
            return similarity * 100
        except ImportError:
            # Fallback: simple word matching
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return (len(intersection) / len(union)) * 100
    
    def find_similar_maps(self, query: str, all_maps: List[tuple], 
                          threshold: Optional[float] = None) -> List[tuple]:
        """Find maps similar to query"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar_maps = []
        for map_data in all_maps:
            if map_data[6]:  # text_content
                similarity = self.calculate_similarity(query, map_data[6])
                if similarity >= threshold:
                    similar_maps.append((map_data, similarity))
        
        # Sort by similarity (highest first)
        similar_maps.sort(key=lambda x: x[1], reverse=True)
        return similar_maps

class TreasureMapBot:
    """Main bot class"""
    
    def __init__(self):
        self.db = Database()
        self.text_matcher = TextMatcher()
        logger.info("Treasure Map Bot initialized")
    
    async def start_command(self, update, context):
        """Handle /start command"""
        user = update.effective_user
        welcome_message = f"""
🌟 **مرحباً {user.first_name}! مرحباً بك في Treasure Map Bot** 🌟

🔍 **ما يمكنك فعله:**
• أرسل صورة خريطة الكنز
• ابحث في الخرائط المحفوظة
• استعرض جميع الخرائط
• اعرض خرائطك الخاصة

📤 **أرسل صورة الآن لتبدأ!**

💡 **الأوامر المتاحة:**
/start - رسالة الترحيب
/search <نص> - البحث في الخرائط
/list - عرض جميع الخرائط
/mymaps - عرض خرائطك
/help - المساعدة
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update, context):
        """Handle /help command"""
        help_text = """
🔧 **مساعدة Treasure Map Bot**

📤 **رفع الخرائط:**
• أرسل صورة خريطة الكنز
• سيتم حفظها تلقائياً
• يمكن البحث فيها لاحقاً

🔍 **البحث:**
/search <نص البحث>
مثال: /search كنز ذهب

📋 **عرض الخرائط:**
/list - عرض جميع الخرائط
/mymaps - عرض خرائطك الخاصة

❓ **مشاكل شائعة:**
• تأكد من أن الصورة واضحة
• استخدم نصوص بحث محددة
• الصور الكبيرة قد تستغرق وقتاً أطول

💡 **نصائح:**
• اكتب وصفاً واضحاً للصورة
• استخدم كلمات مفتاحية محددة
• احفظ الخرائط المهمة
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def search_command(self, update, context):
        """Handle /search command"""
        if not context.args:
            await update.message.reply_text("🔍 **استخدم:** /search <نص البحث>\nمثال: /search كنز ذهب")
            return
        
        query = ' '.join(context.args)
        user = update.effective_user
        
        await update.message.reply_text(f"🔍 **البحث عن:** {query}")
        
        try:
            all_maps = self.db.get_all_treasure_maps()
            if not all_maps:
                await update.message.reply_text("❌ **لا توجد خرائط محفوظة بعد**")
                return
            
            # Use text matcher for similarity
            similar_maps = self.text_matcher.find_similar_maps(query, all_maps)
            
            if not similar_maps:
                await update.message.reply_text("❌ **لم يتم العثور على نتائج مشابهة**")
                return
            
            # Save search history
            self.db.save_search_history(user.id, query, len(similar_maps))
            
            # Show top results
            results_text = f"🔍 **نتائج البحث عن:** {query}\n\n"
            for i, (map_data, similarity) in enumerate(similar_maps[:5], 1):
                username = map_data[2] or map_data[3] or "مستخدم"
                created_at = map_data[7]
                text_content = map_data[6] or "لا يوجد وصف"
                
                results_text += f"**{i}. {username}** ({similarity:.1f}%)\n"
                results_text += f"📅 {created_at}\n"
                results_text += f"📝 {text_content[:100]}...\n\n"
            
            if len(similar_maps) > 5:
                results_text += f"... و {len(similar_maps) - 5} نتيجة أخرى"
            
            await update.message.reply_text(results_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            await update.message.reply_text("❌ **حدث خطأ أثناء البحث**")
    
    async def list_command(self, update, context):
        """Handle /list command"""
        try:
            all_maps = self.db.get_all_treasure_maps()
            
            if not all_maps:
                await update.message.reply_text("📋 **لا توجد خرائط محفوظة بعد**")
                return
            
            list_text = f"📋 **إجمالي الخرائط:** {len(all_maps)}\n\n"
            
            for i, map_data in enumerate(all_maps[:10], 1):
                username = map_data[2] or map_data[3] or "مستخدم"
                created_at = map_data[7]
                text_content = map_data[6] or "لا يوجد وصف"
                file_size = map_data[5] or 0
                
                list_text += f"**{i}. {username}**\n"
                list_text += f"📅 {created_at}\n"
                list_text += f"📁 {file_size // 1024}KB\n"
                list_text += f"📝 {text_content[:80]}...\n\n"
            
            if len(all_maps) > 10:
                list_text += f"... و {len(all_maps) - 10} خرائط أخرى"
            
            await update.message.reply_text(list_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"List error: {e}")
            await update.message.reply_text("❌ **حدث خطأ أثناء عرض القائمة**")
    
    async def mymaps_command(self, update, context):
        """Handle /mymaps command"""
        user = update.effective_user
        
        try:
            user_maps = self.db.get_user_maps(user.id)
            
            if not user_maps:
                await update.message.reply_text("📋 **لم تقم برفع أي خرائط بعد**")
                return
            
            mymaps_text = f"📋 **خرائطك:** {len(user_maps)}\n\n"
            
            for i, map_data in enumerate(user_maps[:10], 1):
                created_at = map_data[7]
                text_content = map_data[6] or "لا يوجد وصف"
                file_size = map_data[5] or 0
                
                mymaps_text += f"**{i}.**\n"
                mymaps_text += f"📅 {created_at}\n"
                mymaps_text += f"📁 {file_size // 1024}KB\n"
                mymaps_text += f"📝 {text_content[:80]}...\n\n"
            
            if len(user_maps) > 10:
                mymaps_text += f"... و {len(user_maps) - 10} خرائط أخرى"
            
            await update.message.reply_text(mymaps_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"MyMaps error: {e}")
            await update.message.reply_text("❌ **حدث خطأ أثناء عرض خرائطك**")
    
    async def handle_photo(self, update, context):
        """Handle photo uploads"""
        user = update.effective_user
        photo = update.message.photo[-1]  # Get highest quality photo
        
        # Check file size
        if photo.file_size > MAX_FILE_SIZE:
            await update.message.reply_text("❌ **الملف كبير جداً! الحد الأقصى 10MB**")
            return
        
        try:
            # Download photo
            file = await context.bot.get_file(photo.file_id)
            file_path = os.path.join(
                UPLOAD_FOLDER, 
                f"treasure_map_{user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            
            await file.download_to_drive(file_path)
            
            # Extract text from image (simplified OCR simulation)
            text_content = await self.extract_text_from_image(file_path, user)
            
            # Save to database
            if self.db.save_treasure_map(
                user.id, user.username, user.first_name, 
                file_path, photo.file_size, text_content
            ):
                success_message = f"""
✅ **تم حفظ خريطة الكنز بنجاح!**

👤 **المستخدم:** {user.first_name}
📁 **الملف:** {os.path.basename(file_path)}
📏 **الحجم:** {photo.file_size // 1024}KB
📝 **النص المستخرج:** {text_content[:100]}...

🔍 **يمكنك البحث فيها باستخدام:**
/search <نص البحث>
                """
                await update.message.reply_text(success_message, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ **فشل في حفظ الخريطة**")
                
        except Exception as e:
            logger.error(f"Photo handling error: {e}")
            await update.message.reply_text("❌ **حدث خطأ أثناء معالجة الصورة**")
    
    async def extract_text_from_image(self, image_path: str, user) -> str:
        """Extract text from image (simplified OCR simulation)"""
        try:
            # This is a simplified OCR simulation
            # In a real implementation, you would use Tesseract or Google Vision API
            
            filename = os.path.basename(image_path)
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            # Generate descriptive text
            text_content = f"خريطة كنز - {user.first_name} - {current_time} - {filename}"
            
            # Add some random treasure-related keywords for better search
            treasure_keywords = ["كنز", "ذهب", "معدن", "أحجار كريمة", "عملات", "مجوهرات"]
            import random
            text_content += f" - {random.choice(treasure_keywords)}"
            
            return text_content
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return f"خريطة كنز - {user.first_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    async def handle_text(self, update, context):
        """Handle text messages"""
        user = update.effective_user
        text = update.message.text
        
        if text.startswith('/'):
            return  # Let command handlers deal with it
        
        # If it's not a command, treat as search query
        await self.search_command(update, context)

async def main():
    """Main function to run the bot"""
    try:
        logger.info("Starting Treasure Map Bot...")
        
        # Create bot instance
        bot = TreasureMapBot()
        
        # Import telegram libraries
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
            logger.info("Telegram libraries imported successfully")
        except ImportError as e:
            logger.error(f"Import error: {e}")
            print("❌ Error: python-telegram-bot not installed")
            print("🔧 Install with: pip3 install python-telegram-bot")
            return
        
        # Create application
        app = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", bot.start_command))
        app.add_handler(CommandHandler("help", bot.help_command))
        app.add_handler(CommandHandler("search", bot.search_command))
        app.add_handler(CommandHandler("list", bot.list_command))
        app.add_handler(CommandHandler("mymaps", bot.mymaps_command))
        
        # Add message handlers
        app.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
        
        logger.info("Bot handlers configured successfully")
        logger.info("Starting bot polling...")
        
        # Start bot
        await app.run_polling()
        
    except Exception as e:
        logger.error(f"Bot startup error: {e}")
        print(f"❌ Fatal error: {e}")
        print("🔧 Check your bot token and internet connection")

if __name__ == "__main__":
    print("🎯 Treasure Map Bot - Starting...")
    asyncio.run(main())