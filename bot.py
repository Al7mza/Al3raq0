#!/usr/bin/env python3
"""
Treasure Map Bot - Telegram bot for matching treasure map clues
"""

import logging
import os
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode

from config import BOT_TOKEN, ADMIN_USER_IDS, UPLOAD_FOLDER
from database import Database
from ocr_handler import OCRHandler
from text_matcher import TextMatcher

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TreasureMapBot:
    def __init__(self):
        """Initialize the bot with all components"""
        self.db = Database()
        self.ocr_handler = OCRHandler()
        self.text_matcher = TextMatcher()
        
        # Initialize database
        self.db.init_database()
        
        # Create uploads directory if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        logger.info("Treasure Map Bot initialized successfully")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        is_admin = user_id in ADMIN_USER_IDS
        
        if is_admin:
            welcome_text = (
                "🎯 **Welcome to Treasure Map Bot - Admin Mode!**\n\n"
                "**Admin Commands:**\n"
                "• `/add` - Upload reference images (reply to an image)\n"
                "• `/delete` - Remove reference images\n"
                "• `/list` - Show all stored images\n"
                "• `/setthreshold <value>` - Set similarity threshold\n"
                "• `/help` - Detailed help\n\n"
                "**User Commands:**\n"
                "• Send any image to find matching treasure maps\n\n"
                "**Current Settings:**\n"
                f"• Similarity Threshold: {self.db.get_similarity_threshold()}%\n"
                f"• OCR Language: {self.ocr_handler.get_language()}\n"
                f"• Total Reference Images: {len(self.db.get_all_reference_texts())}"
            )
        else:
            welcome_text = (
                "🎯 **Welcome to Treasure Map Bot!**\n\n"
                "**How to use:**\n"
                "Simply send me a treasure map image and I'll find similar ones!\n\n"
                "**Commands:**\n"
                "• `/help` - Get detailed help\n"
                "• `/start` - Show this message\n\n"
                "**Current Settings:**\n"
                f"• Similarity Threshold: {self.db.get_similarity_threshold()}%\n"
                f"• OCR Language: {self.ocr_handler.get_language()}"
            )
        
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_id = update.effective_user.id
        is_admin = user_id in ADMIN_USER_IDS
        
        if is_admin:
            help_text = (
                "🔍 **Treasure Map Bot - Admin Help**\n\n"
                "**Admin Commands:**\n"
                "• `/add` - Upload reference images\n"
                "  - Reply to an image with this command\n"
                "  - Supports multiple languages (Chinese, English)\n"
                "  - Automatically extracts text using OCR\n\n"
                "• `/delete` - Remove reference images\n"
                "  - Shows interactive list of images\n"
                "  - Click to delete specific images\n\n"
                "• `/list` - Show all stored images\n"
                "  - Displays image info and extracted text\n"
                "  - Shows upload dates and admin info\n\n"
                "• `/setthreshold <value>` - Set similarity threshold\n"
                "  - Example: `/setthreshold 20` for 20%\n"
                "  - Range: 1-100%\n\n"
                "**User Commands:**\n"
                "• Send any image to find matching treasure maps\n"
                "• Bot will extract text and find similar references\n"
                "• Returns best match if similarity ≥ threshold\n\n"
                "**Technical Details:**\n"
                "• OCR Engine: Tesseract\n"
                "• Text Matching: Fuzzy + Keyword-based\n"
                "• Database: SQLite\n"
                "• File Support: PNG, JPG, JPEG, GIF, BMP"
            )
        else:
            help_text = (
                "🔍 **Treasure Map Bot - User Help**\n\n"
                "**How it works:**\n"
                "1. Send me a treasure map image\n"
                "2. I'll extract text using OCR\n"
                "3. Compare with stored reference images\n"
                "4. Return the best match if found\n\n"
                "**Supported Formats:**\n"
                "• PNG, JPG, JPEG, GIF, BMP\n"
                "• Max file size: 10MB\n\n"
                "**Languages Supported:**\n"
                "• Chinese (Simplified)\n"
                "• English\n\n"
                "**Commands:**\n"
                "• `/help` - Show this help\n"
                "• `/start` - Welcome message\n\n"
                "**Note:** Only admins can add reference images."
            )
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def add_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command for uploading reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\nOnly administrators can add reference images.")
            return
        
        # Check if this is a reply to an image
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(
                "📸 **How to add reference images:**\n\n"
                "1. Send an image\n"
                "2. Reply to that image with `/add`\n\n"
                "The bot will automatically:\n"
                "• Extract text using OCR\n"
                "• Store the image and text\n"
                "• Make it searchable for users"
            )
            return
        
        await update.message.reply_text("🔄 Processing reference image...")
        await self.process_reference_image(update.message.reply_to_message, user_id)
    
    async def process_reference_image(self, message, admin_user_id):
        """Process and store a reference image"""
        try:
            # Get the largest photo size
            photo = message.photo[-1]
            file_id = photo.file_id
            
            # Download the file
            file = await message.bot.get_file(file_id)
            file_path = os.path.join(UPLOAD_FOLDER, f"ref_{file_id}.jpg")
            
            await file.download_to_drive(file_path)
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                os.remove(file_path)
                await message.reply_text("❌ **File too large**\nMaximum file size is 10MB.")
                return
            
            # Extract text using OCR
            extracted_text = self.ocr_handler.extract_text(file_path, self.ocr_handler.get_language())
            
            if not extracted_text.strip():
                await message.reply_text(
                    "⚠️ **Warning: No text extracted**\n"
                    "The image may not contain readable text or OCR failed.\n"
                    "You can still add it, but users may not find matches."
                )
            
            # Store in database
            filename = f"ref_{file_id}.jpg"
            self.db.add_reference_image(
                filename=filename,
                file_path=file_path,
                extracted_text=extracted_text,
                admin_user_id=admin_user_id,
                language=self.ocr_handler.get_language()
            )
            
            # Send confirmation
            text_preview = extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
            await message.reply_text(
                f"✅ **Reference image added successfully!**\n\n"
                f"**Extracted text:**\n`{text_preview}`\n\n"
                f"**File:** {filename}\n"
                f"**Size:** {file_size / 1024:.1f} KB\n"
                f"**Language:** {self.ocr_handler.get_language()}"
            )
            
            logger.info(f"Admin {admin_user_id} added reference image: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing reference image: {e}")
            await message.reply_text(f"❌ **Error processing image:** {str(e)}")
    
    async def delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /delete command for removing reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\nOnly administrators can delete reference images.")
            return
        
        # Get all reference images for this admin
        images = self.db.get_all_reference_images(user_id)
        
        if not images:
            await update.message.reply_text("📭 **No reference images found**\nUse `/add` to upload some first.")
            return
        
        # Create inline keyboard for deletion
        keyboard = []
        for image in images:
            text_preview = image['extracted_text'][:30] + "..." if len(image['extracted_text']) > 30 else image['extracted_text']
            button_text = f"🗑️ {text_preview}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"delete_{image['id']}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🗑️ **Select image to delete:**\n\n"
            "Click on an image below to delete it:",
            reply_markup=reply_markup
        )
    
    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /list command to show all reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\nOnly administrators can list reference images.")
            return
        
        # Get all reference images for this admin
        images = self.db.get_all_reference_images(user_id)
        
        if not images:
            await update.message.reply_text("📭 **No reference images found**\nUse `/add` to upload some first.")
            return
        
        # Create list message
        message_text = f"📚 **Your Reference Images ({len(images)} total):**\n\n"
        
        for i, image in enumerate(images, 1):
            text_preview = image['extracted_text'][:50] + "..." if len(image['extracted_text']) > 50 else image['extracted_text']
            upload_date = image['upload_date'][:10] if image['upload_date'] else "Unknown"
            
            message_text += (
                f"**{i}. {image['filename']}**\n"
                f"📅 Uploaded: {upload_date}\n"
                f"📝 Text: `{text_preview}`\n\n"
            )
        
        # Split long messages if needed
        if len(message_text) > 4000:
            parts = [message_text[i:i+4000] for i in range(0, len(message_text), 4000)]
            for i, part in enumerate(parts):
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(message_text, parse_mode=ParseMode.MARKDOWN)
    
    async def set_threshold_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setthreshold command to set similarity threshold"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ **Access Denied**\nOnly administrators can change settings.")
            return
        
        if not context.args:
            current_threshold = self.db.get_similarity_threshold()
            await update.message.reply_text(
                f"🎯 **Current Similarity Threshold:** {current_threshold}%\n\n"
                "**Usage:** `/setthreshold <value>`\n"
                "**Example:** `/setthreshold 20` for 20%\n"
                "**Range:** 1-100%"
            )
            return
        
        try:
            threshold = int(context.args[0])
            if threshold < 1 or threshold > 100:
                await update.message.reply_text("❌ **Invalid threshold**\nPlease use a value between 1 and 100.")
                return
            
            # Update threshold in database
            self.db.set_similarity_threshold(threshold)
            
            await update.message.reply_text(
                f"✅ **Similarity threshold updated!**\n"
                f"New threshold: **{threshold}%**\n\n"
                "Users will now need at least this similarity to get matches."
            )
            
            logger.info(f"Admin {user_id} updated similarity threshold to {threshold}%")
            
        except ValueError:
            await update.message.reply_text("❌ **Invalid input**\nPlease use a number (e.g., `/setthreshold 20`).")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming images from users"""
        user_id = update.effective_user.id
        is_admin = user_id in ADMIN_USER_IDS
        
        # Check if this is a reply to /add command (admin only)
        if update.message.reply_to_message and update.message.reply_to_message.text == "/add":
            if is_admin:
                await update.message.reply_text("🔄 Processing reference image...")
                await self.process_reference_image(update.message, user_id)
            else:
                await update.message.reply_text("❌ **Access Denied**\nOnly administrators can add reference images.")
            return
        
        # Regular user image processing
        await update.message.reply_text("🔍 **Processing treasure map image...**")
        await self.process_treasure_map_image(update.message)
    
    async def process_treasure_map_image(self, message):
        """Process user's treasure map image and find matches"""
        try:
            # Get the largest photo size
            photo = message.photo[-1]
            file_id = photo.file_id
            
            # Download the file
            file = await message.bot.get_file(file_id)
            temp_file_path = os.path.join(UPLOAD_FOLDER, f"temp_{file_id}.jpg")
            
            await file.download_to_drive(temp_file_path)
            
            # Extract text from user's image
            user_text = self.ocr_handler.extract_text(temp_file_path, self.ocr_handler.get_language())
            
            if not user_text.strip():
                os.remove(temp_file_path)
                await message.reply_text(
                    "❌ **No text found**\n"
                    "I couldn't extract any readable text from your image.\n"
                    "Please try a clearer image with better text quality."
                )
                return
            
            # Get all reference texts for comparison
            reference_texts = self.db.get_all_reference_texts()
            
            if not reference_texts:
                os.remove(temp_file_path)
                await message.reply_text(
                    "📭 **No reference images available**\n"
                    "Admins haven't uploaded any reference images yet.\n"
                    "Please contact an administrator."
                )
                return
            
            # Find best match
            threshold = self.db.get_similarity_threshold()
            best_match = self.text_matcher.find_best_match(user_text, reference_texts, threshold)
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
            if best_match:
                # Get the reference image file
                reference_image = self.db.get_reference_image_by_id(best_match['id'])
                
                if reference_image and os.path.exists(reference_image['file_path']):
                    # Send the matching reference image
                    with open(reference_image['file_path'], 'rb') as photo_file:
                        await message.reply_photo(
                            photo=photo_file,
                            caption=(
                                f"🎯 **Match Found!**\n\n"
                                f"**Similarity:** {best_match['similarity']:.1f}%\n"
                                f"**Threshold:** {threshold}%\n\n"
                                f"**Your text:**\n`{user_text[:100]}{'...' if len(user_text) > 100 else ''}`\n\n"
                                f"**Reference text:**\n`{reference_image['extracted_text'][:100]}{'...' if len(reference_image['extracted_text']) > 100 else ''}`"
                            ),
                            parse_mode=ParseMode.MARKDOWN
                        )
                    
                    logger.info(f"User {message.from_user.id} found match: {best_match['similarity']:.1f}%")
                else:
                    await message.reply_text("❌ **Error retrieving reference image**\nPlease contact an administrator.")
            else:
                await message.reply_text(
                    f"❌ **No similar treasure map found**\n\n"
                    f"**Your text:**\n`{user_text[:100]}{'...' if len(user_text) > 100 else ''}`\n\n"
                    f"**Similarity threshold:** {threshold}%\n\n"
                    "Try uploading a clearer image or contact an admin to add more reference images."
                )
                
        except Exception as e:
            logger.error(f"Error processing treasure map image: {e}")
            await message.reply_text(f"❌ **Error processing image:** {str(e)}")
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()
        
        if query.data.startswith("delete_"):
            await self.handle_delete_callback(query)
    
    async def handle_delete_callback(self, query):
        """Handle delete confirmation callbacks"""
        try:
            image_id = int(query.data.split("_")[1])
            user_id = query.from_user.id
            
            if user_id not in ADMIN_USER_IDS:
                await query.edit_message_text("❌ **Access Denied**\nOnly administrators can delete images.")
                return
            
            # Delete the image
            success = self.db.delete_reference_image(image_id, user_id)
            
            if success:
                await query.edit_message_text("✅ **Image deleted successfully!**")
                logger.info(f"Admin {user_id} deleted image {image_id}")
            else:
                await query.edit_message_text("❌ **Failed to delete image**\nImage not found or access denied.")
                
        except Exception as e:
            logger.error(f"Error handling delete callback: {e}")
            await query.edit_message_text("❌ **Error deleting image**\nPlease try again.")
    
    def setup_handlers(self, application: Application):
        """Set up all bot handlers"""
        # Command handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("add", self.add_command))
        application.add_handler(CommandHandler("delete", self.delete_command))
        application.add_handler(CommandHandler("list", self.list_command))
        application.add_handler(CommandHandler("setthreshold", self.set_threshold_command))
        
        # Message handlers
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        
        # Callback query handlers
        application.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def run(self):
        """Start the bot"""
        # Create application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Set up handlers
        self.setup_handlers(application)
        
        # Start the bot
        logger.info("Starting Treasure Map Bot...")
        await application.run_polling()

async def main():
    """Main function"""
    bot = TreasureMapBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())