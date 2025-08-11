#!/usr/bin/env python3
"""
Simple Treasure Map Bot - Easy Hosting Version
Just copy-paste and run!
"""
import os
import sqlite3
import asyncio
import logging
from datetime import datetime

# Simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your bot configuration
BOT_TOKEN = '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY'
ADMIN_USER_IDS = [1694244496]
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'treasure_map.db'

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SimpleDB:
    def __init__(self):
        self.init_db()
    
    def init_db(self):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maps (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    username TEXT,
                    file_path TEXT,
                    text_content TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            print("✅ Database ready!")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    def save_map(self, user_id, username, file_path, text_content):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO maps (user_id, username, file_path, text_content, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, username, file_path, text_content, datetime.now()))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def get_all_maps(self):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM maps ORDER BY created_at DESC')
            result = cursor.fetchall()
            conn.close()
            return result
        except Exception as e:
            print(f"❌ Fetch error: {e}")
            return []

class SimpleBot:
    def __init__(self):
        self.db = SimpleDB()
        print("🤖 Bot initialized!")
    
    async def start_command(self, update, context):
        user = update.effective_user
        message = f"""
🌟 مرحباً {user.first_name}! 
أهلاً بك في Treasure Map Bot

📤 أرسل صورة خريطة الكنز
🔍 ابحث في الخرائط المحفوظة
📋 اعرض جميع الخرائط

الأوامر:
/start - رسالة الترحيب
/search <نص> - البحث
/list - عرض الخرائط
        """
        await update.message.reply_text(message)
    
    async def help_command(self, update, context):
        help_text = """
🔧 مساعدة Treasure Map Bot

📤 رفع الخرائط:
• أرسل صورة خريطة الكنز
• سيتم حفظها تلقائياً

🔍 البحث:
/search <نص البحث>
مثال: /search كنز ذهب

📋 عرض الخرائط:
/list - عرض جميع الخرائط
        """
        await update.message.reply_text(help_text)
    
    async def search_command(self, update, context):
        if not context.args:
            await update.message.reply_text("🔍 استخدم: /search <نص البحث>")
            return
        
        query = ' '.join(context.args)
        await update.message.reply_text(f"🔍 البحث عن: {query}")
        
        try:
            all_maps = self.db.get_all_maps()
            if not all_maps:
                await update.message.reply_text("❌ لا توجد خرائط محفوظة")
                return
            
            # Simple search
            results = []
            for map_data in all_maps:
                if map_data[4] and query.lower() in map_data[4].lower():
                    results.append(map_data)
            
            if not results:
                await update.message.reply_text("❌ لم يتم العثور على نتائج")
                return
            
            # Show results
            results_text = f"🔍 نتائج البحث عن: {query}\n\n"
            for i, map_data in enumerate(results[:5], 1):
                username = map_data[2] or "مستخدم"
                text_content = map_data[4] or "لا يوجد وصف"
                results_text += f"{i}. {username}\n{text_content[:80]}...\n\n"
            
            await update.message.reply_text(results_text)
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            await update.message.reply_text("❌ حدث خطأ أثناء البحث")
    
    async def list_command(self, update, context):
        try:
            all_maps = self.db.get_all_maps()
            
            if not all_maps:
                await update.message.reply_text("📋 لا توجد خرائط محفوظة")
                return
            
            list_text = f"📋 إجمالي الخرائط: {len(all_maps)}\n\n"
            
            for i, map_data in enumerate(all_maps[:10], 1):
                username = map_data[2] or "مستخدم"
                text_content = map_data[4] or "لا يوجد وصف"
                list_text += f"{i}. {username}\n{text_content[:60]}...\n\n"
            
            if len(all_maps) > 10:
                list_text += f"... و {len(all_maps) - 10} خرائط أخرى"
            
            await update.message.reply_text(list_text)
            
        except Exception as e:
            print(f"❌ List error: {e}")
            await update.message.reply_text("❌ حدث خطأ أثناء عرض القائمة")
    
    async def handle_photo(self, update, context):
        user = update.effective_user
        photo = update.message.photo[-1]
        
        try:
            # Download photo
            file = await context.bot.get_file(photo.file_id)
            file_path = os.path.join(UPLOAD_FOLDER, f"map_{user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            
            await file.download_to_drive(file_path)
            
            # Simple text extraction (placeholder)
            text_content = f"خريطة كنز - {user.first_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Save to database
            if self.db.save_map(user.id, user.username, file_path, text_content):
                success_message = f"""
✅ تم حفظ خريطة الكنز!

👤 المستخدم: {user.first_name}
📁 الملف: {os.path.basename(file_path)}
📝 النص: {text_content}

🔍 ابحث فيها: /search <نص>
                """
                await update.message.reply_text(success_message)
            else:
                await update.message.reply_text("❌ فشل في حفظ الخريطة")
                
        except Exception as e:
            print(f"❌ Photo error: {e}")
            await update.message.reply_text("❌ حدث خطأ أثناء معالجة الصورة")
    
    async def handle_text(self, update, context):
        user = update.effective_user
        text = update.message.text
        
        if text.startswith('/'):
            return
        
        # Treat as search query
        await self.search_command(update, context)

async def main():
    """Main function"""
    try:
        print("🚀 Starting Simple Treasure Map Bot...")
        
        # Create bot instance
        bot = SimpleBot()
        
        # Import telegram libraries
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
            print("✅ Telegram libraries imported!")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("🔧 Install with: pip3 install python-telegram-bot")
            return
        
        # Create application
        app = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", bot.start_command))
        app.add_handler(CommandHandler("help", bot.help_command))
        app.add_handler(CommandHandler("search", bot.search_command))
        app.add_handler(CommandHandler("list", bot.list_command))
        
        app.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
        
        print("✅ Bot handlers configured!")
        print("🤖 Starting bot...")
        
        # Start bot
        await app.run_polling()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("🔧 Check your bot token and internet connection")

if __name__ == "__main__":
    print("🎯 Simple Treasure Map Bot - Starting...")
    asyncio.run(main())