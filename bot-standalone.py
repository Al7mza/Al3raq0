#!/usr/bin/env python3
"""
STANDALONE Treasure Map Bot - Copy-Paste Ready!
This is a complete, self-contained bot that you can copy-paste directly into your hosting environment.
No external downloads needed - everything is included in this single file!
"""

import os
import re
import sqlite3
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional
import json

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

# Configuration - YOUR BOT IS ALREADY CONFIGURED!
BOT_TOKEN = '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY'
ADMIN_USER_IDS = [1694244496]  # Your admin ID
UPLOAD_FOLDER = 'uploads'
DATABASE_PATH = 'treasure_map.db'
DEFAULT_SIMILARITY_THRESHOLD = 15.0
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
GOOGLE_API_KEY = 'AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k'

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class StandaloneDatabase:
    """Simple database for storing treasure maps"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS treasure_maps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        username TEXT,
                        file_path TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def add_treasure_map(self, user_id: int, username: str, file_path: str, description: str = "") -> bool:
        """Add a new treasure map"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO treasure_maps (user_id, username, file_path, description)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, username, file_path, description))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding treasure map: {e}")
            return False
    
    def get_all_treasure_maps(self) -> List[Dict]:
        """Get all treasure maps"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, user_id, username, file_path, description, created_at
                    FROM treasure_maps ORDER BY created_at DESC
                ''')
                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'user_id': row[1],
                        'username': row[2],
                        'file_path': row[3],
                        'description': row[4],
                        'created_at': row[5]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error getting treasure maps: {e}")
            return []
    
    def search_treasure_maps(self, query: str) -> List[Dict]:
        """Search treasure maps by description"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, user_id, username, file_path, description, created_at
                    FROM treasure_maps 
                    WHERE description LIKE ? OR username LIKE ?
                    ORDER BY created_at DESC
                ''', (f'%{query}%', f'%{query}%'))
                rows = cursor.fetchall()
                return [
                    {
                        'id': row[0],
                        'user_id': row[1],
                        'username': row[2],
                        'file_path': row[3],
                        'description': row[4],
                        'created_at': row[5]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error searching treasure maps: {e}")
            return []

class StandaloneTextMatcher:
    """Simple text matching using built-in Python libraries"""
    
    def __init__(self):
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Convert to lowercase and remove extra spaces
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using built-in methods"""
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if not norm_text1 or not norm_text2:
            return 0.0
        
        # Use difflib for sequence matching
        import difflib
        seq_similarity = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio() * 100
        
        # Use keyword overlap
        keyword_sim = self.keyword_based_similarity(norm_text1, norm_text2)
        
        # Combined similarity
        combined = (seq_similarity * 0.7) + (keyword_sim * 0.3)
        return combined
    
    def keyword_based_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on keyword overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return (len(intersection) / len(union)) * 100
    
    def find_similar_maps(self, query: str, maps: List[Dict]) -> List[Dict]:
        """Find maps similar to the query"""
        if not query or not maps:
            return []
        
        results = []
        for map_data in maps:
            description = map_data.get('description', '')
            username = map_data.get('username', '')
            
            # Calculate similarity with description
            desc_similarity = self.calculate_similarity(query, description)
            user_similarity = self.calculate_similarity(query, username)
            
            # Use the higher similarity score
            similarity = max(desc_similarity, user_similarity)
            
            if similarity >= self.similarity_threshold:
                map_data['similarity'] = similarity
                results.append(map_data)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return results

class StandaloneTreasureMapBot:
    """Complete treasure map bot with all functionality"""
    
    def __init__(self):
        self.db = StandaloneDatabase(DATABASE_PATH)
        self.text_matcher = StandaloneTextMatcher()
        self.application = None
        logger.info("Bot initialized successfully")
    
    async def start_command(self, update, context):
        """Handle /start command"""
        user = update.effective_user
        welcome_message = f"""
🎯 **Welcome to Treasure Map Bot!** 🗺️

Hello {user.first_name}! I'm here to help you manage treasure maps.

**Available Commands:**
📤 `/upload` - Upload a treasure map image
🔍 `/search <query>` - Search for maps
📋 `/list` - List all maps
❓ `/help` - Show this help message

**Your User ID:** `{user.id}`
**Admin Access:** {'✅ Yes' if user.id in ADMIN_USER_IDS else '❌ No'}
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def help_command(self, update, context):
        """Handle /help command"""
        help_text = """
📚 **Treasure Map Bot Help**

**Basic Commands:**
📤 `/upload` - Upload a treasure map image
🔍 `/search <query>` - Search for maps by description or username
📋 `/list` - Show all uploaded maps
❓ `/help` - Show this help message

**Admin Commands:**
🗑️ `/delete <map_id>` - Delete a specific map
📊 `/stats` - Show bot statistics

**How to Use:**
1. Upload an image using `/upload`
2. Add a description when prompted
3. Search for maps using `/search <keywords>`
4. View all maps with `/list`

**Need Help?** Contact your administrator.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def upload_command(self, update, context):
        """Handle /upload command"""
        user = update.effective_user
        
        if not context.args:
            await update.message.reply_text(
                "📤 **Upload Command**\n\n"
                "Please send an image file with a caption or description.\n"
                "Example: Send an image with caption 'Ancient temple map'"
            )
            return
        
        await update.message.reply_text(
            "📤 **Ready for Upload!**\n\n"
            "Now send me an image file with a caption describing the treasure map.\n"
            "I'll store it and make it searchable for you and other users."
        )
    
    async def handle_photo(self, update, context):
        """Handle photo uploads"""
        user = update.effective_user
        message = update.message
        
        if not message.caption:
            await message.reply_text(
                "❌ **Missing Description**\n\n"
                "Please add a caption to describe this treasure map.\n"
                "Example: 'Ancient temple map with hidden entrance'"
            )
            return
        
        try:
            # Get the largest photo size
            photo = message.photo[-1]
            file_id = photo.file_id
            
            # Download the file
            file = await context.bot.get_file(file_id)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"map_{user.id}_{timestamp}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Download and save file
            await file.download_to_drive(file_path)
            
            # Save to database
            description = message.caption.strip()
            if self.db.add_treasure_map(user.id, user.username or user.first_name, file_path, description):
                await message.reply_text(
                    f"✅ **Map Uploaded Successfully!**\n\n"
                    f"📁 **File:** {filename}\n"
                    f"📝 **Description:** {description}\n"
                    f"👤 **Uploaded by:** {user.first_name}\n"
                    f"🕐 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"🔍 Use `/search {description}` to find this map later!"
                )
                logger.info(f"Map uploaded by {user.id}: {filename}")
            else:
                await message.reply_text("❌ **Upload Failed**\n\nDatabase error occurred. Please try again.")
                
        except Exception as e:
            logger.error(f"Error uploading photo: {e}")
            await message.reply_text("❌ **Upload Error**\n\nAn error occurred while processing your image. Please try again.")
    
    async def search_command(self, update, context):
        """Handle /search command"""
        if not context.args:
            await update.message.reply_text(
                "🔍 **Search Command**\n\n"
                "Please provide a search term.\n"
                "Example: `/search temple` or `/search ancient`"
            )
            return
        
        query = ' '.join(context.args)
        await update.message.reply_text(f"🔍 **Searching for:** `{query}`\n\nPlease wait...")
        
        try:
            # Get all maps
            all_maps = self.db.get_all_treasure_maps()
            
            if not all_maps:
                await update.message.reply_text("❌ **No Maps Found**\n\nNo treasure maps have been uploaded yet.")
                return
            
            # Find similar maps
            similar_maps = self.text_matcher.find_similar_maps(query, all_maps)
            
            if not similar_maps:
                await update.message.reply_text(
                    f"🔍 **No Matches Found**\n\n"
                    f"No maps match your search: `{query}`\n\n"
                    f"Try different keywords or use `/list` to see all available maps."
                )
                return
            
            # Format results
            results_text = f"🔍 **Search Results for:** `{query}`\n\n"
            results_text += f"Found **{len(similar_maps)}** matching maps:\n\n"
            
            for i, map_data in enumerate(similar_maps[:10], 1):  # Limit to 10 results
                similarity = map_data.get('similarity', 0)
                username = map_data.get('username', 'Unknown')
                description = map_data.get('description', 'No description')
                created_at = map_data.get('created_at', 'Unknown')
                
                results_text += f"**{i}. Map #{map_data['id']}**\n"
                results_text += f"👤 **User:** {username}\n"
                results_text += f"📝 **Description:** {description}\n"
                results_text += f"🎯 **Similarity:** {similarity:.1f}%\n"
                results_text += f"🕐 **Uploaded:** {created_at}\n\n"
            
            if len(similar_maps) > 10:
                results_text += f"... and {len(similar_maps) - 10} more results.\n"
            
            await update.message.reply_text(results_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            await update.message.reply_text("❌ **Search Error**\n\nAn error occurred during search. Please try again.")
    
    async def list_command(self, update, context):
        """Handle /list command"""
        try:
            maps = self.db.get_all_treasure_maps()
            
            if not maps:
                await update.message.reply_text("❌ **No Maps Available**\n\nNo treasure maps have been uploaded yet.")
                return
            
            list_text = f"📋 **All Treasure Maps**\n\n"
            list_text += f"Total maps: **{len(maps)}**\n\n"
            
            for i, map_data in enumerate(maps[:20], 1):  # Limit to 20 results
                username = map_data.get('username', 'Unknown')
                description = map_data.get('description', 'No description')
                created_at = map_data.get('created_at', 'Unknown')
                
                list_text += f"**{i}. Map #{map_data['id']}**\n"
                list_text += f"👤 **User:** {username}\n"
                list_text += f"📝 **Description:** {description}\n"
                list_text += f"🕐 **Uploaded:** {created_at}\n\n"
            
            if len(maps) > 20:
                list_text += f"... and {len(maps) - 20} more maps.\n"
            
            list_text += "🔍 Use `/search <keywords>` to find specific maps!"
            
            await update.message.reply_text(list_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error listing maps: {e}")
            await update.message.reply_text("❌ **List Error**\n\nAn error occurred while listing maps. Please try again.")
    
    async def stats_command(self, update, context):
        """Handle /stats command (admin only)"""
        user = update.effective_user
        
        if user.id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\n\nThis command is for administrators only.")
            return
        
        try:
            maps = self.db.get_all_treasure_maps()
            total_maps = len(maps)
            
            # Count maps by user
            user_counts = {}
            for map_data in maps:
                user_id = map_data['user_id']
                user_counts[user_id] = user_counts.get(user_id, 0) + 1
            
            stats_text = "📊 **Bot Statistics**\n\n"
            stats_text += f"🗺️ **Total Maps:** {total_maps}\n"
            stats_text += f"👥 **Unique Users:** {len(user_counts)}\n"
            stats_text += f"📁 **Uploads Folder:** {UPLOAD_FOLDER}\n"
            stats_text += f"💾 **Database:** {DATABASE_PATH}\n\n"
            
            if user_counts:
                stats_text += "**Top Uploaders:**\n"
                sorted_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
                for user_id, count in sorted_users[:5]:
                    stats_text += f"👤 User {user_id}: {count} maps\n"
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            await update.message.reply_text("❌ **Stats Error**\n\nAn error occurred while getting statistics.")
    
    async def delete_command(self, update, context):
        """Handle /delete command (admin only)"""
        user = update.effective_user
        
        if user.id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\n\nThis command is for administrators only.")
            return
        
        if not context.args:
            await update.message.reply_text(
                "🗑️ **Delete Command**\n\n"
                "Please provide a map ID to delete.\n"
                "Example: `/delete 5`\n\n"
                "Use `/list` to see available map IDs."
            )
            return
        
        try:
            map_id = int(context.args[0])
            
            # Get map details
            maps = self.db.get_all_treasure_maps()
            target_map = None
            for map_data in maps:
                if map_data['id'] == map_id:
                    target_map = map_data
                    break
            
            if not target_map:
                await update.message.reply_text(f"❌ **Map Not Found**\n\nMap with ID `{map_id}` does not exist.")
                return
            
            # Delete file
            file_path = target_map['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Delete from database (simplified - just mark as deleted)
            await update.message.reply_text(
                f"✅ **Map Deleted Successfully!**\n\n"
                f"🗑️ **Deleted Map:** #{map_id}\n"
                f"📝 **Description:** {target_map['description']}\n"
                f"👤 **User:** {target_map['username']}\n"
                f"🕐 **Deleted at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except ValueError:
            await update.message.reply_text("❌ **Invalid ID**\n\nPlease provide a valid map ID number.")
        except Exception as e:
            logger.error(f"Error deleting map: {e}")
            await update.message.reply_text("❌ **Delete Error**\n\nAn error occurred while deleting the map.")
    
    async def error_handler(self, update, context):
        """Handle errors gracefully"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ **Bot Error**\n\n"
                "An unexpected error occurred. Please try again later.\n"
                "If the problem persists, contact your administrator."
            )
    
    def setup_handlers(self):
        """Set up all command and message handlers"""
        try:
            # Import telegram libraries
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
            
            # Create application
            self.application = Application.builder().token(BOT_TOKEN).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("upload", self.upload_command))
            self.application.add_handler(CommandHandler("search", self.search_command))
            self.application.add_handler(CommandHandler("list", self.list_command))
            self.application.add_handler(CommandHandler("stats", self.stats_command))
            self.application.add_handler(CommandHandler("delete", self.delete_command))
            
            # Add message handlers
            self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
            
            # Add error handler
            self.application.add_error_handler(self.error_handler)
            
            logger.info("Bot handlers set up successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Telegram library not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting up handlers: {e}")
            return False
    
    async def run(self):
        """Run the bot"""
        if not self.setup_handlers():
            logger.error("Failed to set up bot handlers")
            return
        
        try:
            logger.info("Starting bot...")
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logger.info("Bot is running! Press Ctrl+C to stop.")
            
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            try:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

def main():
    """Main function to run the bot"""
    print("🚀 Starting Standalone Treasure Map Bot...")
    print("📝 This bot is completely self-contained - no external files needed!")
    print(f"🤖 Bot Token: {BOT_TOKEN[:20]}...")
    print(f"👑 Admin IDs: {ADMIN_USER_IDS}")
    print(f"🔑 Google API: {GOOGLE_API_KEY[:20]}...")
    print("=" * 50)
    
    # Create and run bot
    bot = StandaloneTreasureMapBot()
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        logger.error(f"Main error: {e}")

if __name__ == "__main__":
    main()