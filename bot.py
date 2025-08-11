import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
import asyncio
from datetime import datetime

from config import BOT_TOKEN, ADMIN_USER_IDS, UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
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
        self.db = Database()
        self.ocr = OCRHandler()
        self.matcher = TextMatcher()
        
        # Initialize bot application
        self.application = Application.builder().token(BOT_TOKEN).build()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all bot handlers"""
        # Admin commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("add", self.add_command))
        self.application.add_handler(CommandHandler("delete", self.delete_command))
        self.application.add_handler(CommandHandler("list", self.list_command))
        self.application.add_handler(CommandHandler("setthreshold", self.set_threshold_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        
        # Handle image uploads
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        
        # Handle callback queries (for delete confirmation)
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        
        if user_id in ADMIN_USER_IDS:
            welcome_text = (
                "🎯 **Treasure Map Bot** - Admin Mode\n\n"
                "Welcome! You have admin privileges.\n\n"
                "**Admin Commands:**\n"
                "• `/add` - Upload reference images\n"
                "• `/delete` - Remove reference images\n"
                "• `/list` - Show all stored images\n"
                "• `/setthreshold <value>` - Set similarity threshold\n"
                "• `/help` - Show help information\n\n"
                "**How it works:**\n"
                "1. Upload reference images with `/add`\n"
                "2. Users send treasure map images\n"
                "3. Bot finds matches and sends reference images\n\n"
                "Current similarity threshold: **{}%**"
            ).format(self.db.get_similarity_threshold())
        else:
            welcome_text = (
                "🎯 **Treasure Map Bot**\n\n"
                "Welcome! I can help you find treasure map references.\n\n"
                "**How to use:**\n"
                "Simply send me an image of your treasure map clue, "
                "and I'll try to find a matching reference image.\n\n"
                "**Features:**\n"
                "• OCR text extraction\n"
                "• Fuzzy text matching\n"
                "• Support for multiple languages\n"
                "• Fast and accurate results"
            )
        
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        user_id = update.effective_user.id
        
        if user_id in ADMIN_USER_IDS:
            help_text = (
                "🔧 **Admin Help**\n\n"
                "**Commands:**\n"
                "• `/add` - Upload reference images\n"
                "  - Reply to an image with this command\n"
                "  - Bot will extract text and store the image\n\n"
                "• `/delete` - Remove reference images\n"
                "  - Shows list of images to delete\n\n"
                "• `/list` - Show all stored images\n"
                "  - Displays all reference images with extracted text\n\n"
                "• `/setthreshold <value>` - Set similarity threshold\n"
                "  - Example: `/setthreshold 20` for 20%\n"
                "  - Default: 15%\n\n"
                "**Tips:**\n"
                "• Upload clear, high-quality reference images\n"
                "• Text should be clearly visible for better OCR\n"
                "• Adjust threshold based on your needs"
            )
        else:
            help_text = (
                "❓ **User Help**\n\n"
                "**How to use:**\n"
                "1. Take a photo of your treasure map clue\n"
                "2. Send it to me\n"
                "3. I'll extract the text and find matches\n"
                "4. If a match is found, I'll send the reference image\n\n"
                "**Tips:**\n"
                "• Ensure text is clearly visible\n"
                "• Good lighting helps with OCR accuracy\n"
                "• Send one image at a time"
            )
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def add_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command for uploading reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ You don't have permission to use this command.")
            return
        
        # Check if this is a reply to an image
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(
                "📸 **How to add a reference image:**\n\n"
                "1. Send an image to the chat\n"
                "2. Reply to that image with `/add`\n\n"
                "The bot will extract the text and store the image for future matching."
            )
            return
        
        await self.process_reference_image(update, context)
    
    async def process_reference_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process and store a reference image"""
        user_id = update.effective_user.id
        message = update.message.reply_to_message
        
        # Get the largest photo size
        photo = max(message.photo, key=lambda p: p.file_size)
        
        # Check file size
        if photo.file_size > MAX_FILE_SIZE:
            await update.message.reply_text(
                f"❌ File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
            )
            return
        
        # Download the image
        status_msg = await update.message.reply_text("⏳ Processing image...")
        
        try:
            file = await context.bot.get_file(photo.file_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ref_{user_id}_{timestamp}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            await file.download_to_drive(file_path)
            
            # Extract text using OCR
            await status_msg.edit_text("🔍 Extracting text with OCR...")
            extracted_text = self.ocr.extract_text(file_path)
            
            if not extracted_text.strip():
                await status_msg.edit_text(
                    "⚠️ **Warning:** No text could be extracted from this image.\n\n"
                    "The image will still be stored, but matching may be less effective.\n"
                    "Consider uploading a clearer image with more visible text."
                )
                extracted_text = "No text extracted"
            
            # Store in database
            image_id = self.db.add_reference_image(
                filename, file_path, extracted_text, user_id
            )
            
            # Show success message
            success_text = (
                "✅ **Reference image added successfully!**\n\n"
                f"**Image ID:** {image_id}\n"
                f"**Filename:** {filename}\n\n"
                "**Extracted Text:**\n"
                f"```\n{extracted_text[:500]}{'...' if len(extracted_text) > 500 else ''}\n```\n\n"
                "The image is now available for treasure map matching."
            )
            
            await status_msg.edit_text(success_text, parse_mode=ParseMode.MARKDOWN)
            
        except Exception as e:
            logger.error(f"Error processing reference image: {e}")
            await status_msg.edit_text(
                "❌ **Error:** Failed to process the image. Please try again."
            )
    
    async def delete_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /delete command for removing reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ You don't have permission to use this command.")
            return
        
        # Get all reference images for this admin
        images = self.db.get_all_reference_images(user_id)
        
        if not images:
            await update.message.reply_text("📭 No reference images found.")
            return
        
        # Create inline keyboard for deletion
        keyboard = []
        for image_id, filename, extracted_text, upload_date in images:
            # Truncate text for display
            display_text = extracted_text[:50] + "..." if len(extracted_text) > 50 else extracted_text
            button_text = f"🗑️ {filename[:20]}... | {display_text}"
            
            keyboard.append([
                InlineKeyboardButton(
                    button_text,
                    callback_data=f"delete_{image_id}"
                )
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🗑️ **Select an image to delete:**\n\n"
            "Click on the image you want to remove:",
            reply_markup=reply_markup
        )
    
    async def list_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /list command to show all reference images"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ You don't have permission to use this command.")
            return
        
        # Get all reference images for this admin
        images = self.db.get_all_reference_images(user_id)
        
        if not images:
            await update.message.reply_text("📭 No reference images found.")
            return
        
        # Create list message
        list_text = f"📋 **Reference Images ({len(images)} total):**\n\n"
        
        for i, (image_id, filename, extracted_text, upload_date) in enumerate(images, 1):
            # Format upload date
            try:
                date_obj = datetime.fromisoformat(upload_date.replace('Z', '+00:00'))
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = upload_date
            
            # Truncate text for display
            display_text = extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
            
            list_text += (
                f"**{i}. Image ID: {image_id}**\n"
                f"📁 {filename}\n"
                f"📅 {formatted_date}\n"
                f"📝 {display_text}\n\n"
            )
        
        # Split message if too long
        if len(list_text) > 4000:
            parts = [list_text[i:i+4000] for i in range(0, len(list_text), 4000)]
            for i, part in enumerate(parts):
                if i == 0:
                    await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
                else:
                    await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(list_text, parse_mode=ParseMode.MARKDOWN)
    
    async def set_threshold_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setthreshold command to set similarity threshold"""
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await update.message.reply_text("❌ You don't have permission to use this command.")
            return
        
        # Check if threshold value is provided
        if not context.args:
            current_threshold = self.db.get_similarity_threshold()
            await update.message.reply_text(
                f"🎯 **Current similarity threshold:** {current_threshold}%\n\n"
                "**Usage:** `/setthreshold <value>`\n"
                "**Example:** `/setthreshold 20` for 20%\n\n"
                "**Recommended range:** 10-25%\n"
                "• Lower values = more matches (but may be less accurate)\n"
                "• Higher values = fewer matches (but more accurate)"
            )
            return
        
        try:
            threshold = int(context.args[0])
            
            if threshold < 1 or threshold > 100:
                await update.message.reply_text(
                    "❌ **Invalid threshold value.**\n\n"
                    "Please use a value between 1 and 100."
                )
                return
            
            # Set the threshold
            self.db.set_similarity_threshold(threshold)
            
            await update.message.reply_text(
                f"✅ **Similarity threshold updated!**\n\n"
                f"New threshold: **{threshold}%**\n\n"
                "This will affect how strictly the bot matches treasure map texts."
            )
            
        except ValueError:
            await update.message.reply_text(
                "❌ **Invalid threshold value.**\n\n"
                "Please use a number (e.g., `/setthreshold 20`)."
            )
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming images from users"""
        user_id = update.effective_user.id
        
        # Check if this is a reply to /add command (admin function)
        if update.message.reply_to_message and update.message.reply_to_message.text == '/add':
            return  # This will be handled by the add command
        
        # Get the largest photo size
        photo = max(update.message.photo, key=lambda p: p.file_size)
        
        # Check file size
        if photo.file_size > MAX_FILE_SIZE:
            await update.message.reply_text(
                f"❌ File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
            )
            return
        
        # Process the treasure map image
        await self.process_treasure_map_image(update, context)
    
    async def process_treasure_map_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Process a treasure map image and find matches"""
        user_id = update.effective_user.id
        message = update.message
        
        # Send initial status
        status_msg = await message.reply_text("🔍 **Processing your treasure map...**\n\n⏳ Extracting text with OCR...")
        
        try:
            # Download the image
            file = await context.bot.get_file(message.photo[-1].file_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"treasure_{user_id}_{timestamp}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            await file.download_to_drive(file_path)
            
            # Extract text using OCR
            await status_msg.edit_text("🔍 **Processing your treasure map...**\n\n📝 Comparing with reference images...")
            extracted_text = self.ocr.extract_text(file_path)
            
            if not extracted_text.strip():
                await status_msg.edit_text(
                    "⚠️ **No text found in your image.**\n\n"
                    "I couldn't extract any text from your treasure map image. "
                    "This might be because:\n"
                    "• The text is not clearly visible\n"
                    "• The image quality is too low\n"
                    "• The text is in a language I don't support\n\n"
                    "Please try with a clearer image."
                )
                
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return
            
            # Get all reference texts for comparison
            reference_texts = self.db.get_all_reference_texts()
            
            if not reference_texts:
                await status_msg.edit_text(
                    "📭 **No reference images available.**\n\n"
                    "There are no reference images stored in the database yet. "
                    "Please ask an admin to upload some reference images first."
                )
                
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
                return
            
            # Find matches
            threshold = self.db.get_similarity_threshold()
            matches = self.matcher.find_all_matches(extracted_text, reference_texts, threshold)
            
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            if not matches:
                await status_msg.edit_text(
                    "❌ **No similar treasure map found.**\n\n"
                    f"**Extracted text:**\n```\n{extracted_text[:300]}{'...' if len(extracted_text) > 300 else ''}\n```\n\n"
                    f"**Similarity threshold:** {threshold}%\n\n"
                    "The text from your image didn't match any stored reference images. "
                    "This could mean:\n"
                    "• The reference image isn't uploaded yet\n"
                    "• The text is too different\n"
                    "• The threshold is set too high"
                )
                return
            
            # Send the best match
            best_match = matches[0]
            reference_image = self.db.get_reference_image_by_id(best_match['id'])
            
            if reference_image:
                # Send the reference image with match details
                with open(reference_image[2], 'rb') as photo_file:
                    caption = (
                        f"🎯 **Treasure Map Match Found!**\n\n"
                        f"**Similarity:** {best_match['similarity']}%\n"
                        f"**Threshold:** {threshold}%\n\n"
                        f"**Your text:**\n```\n{extracted_text[:200]}{'...' if len(extracted_text) > 200 else ''}\n```\n\n"
                        f"**Reference text:**\n```\n{reference_image[3][:200]}{'...' if len(reference_image[3]) > 200 else ''}\n```"
                    )
                    
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=photo_file,
                        caption=caption,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    await status_msg.delete()
                    
                    # If there are multiple matches, show them
                    if len(matches) > 1:
                        matches_text = f"🔍 **Found {len(matches)} matches:**\n\n"
                        for i, match in enumerate(matches[:5], 1):  # Show top 5
                            ref_img = self.db.get_reference_image_by_id(match['id'])
                            if ref_img:
                                matches_text += f"{i}. **{ref_img[1]}** - {match['similarity']}%\n"
                        
                        if len(matches) > 5:
                            matches_text += f"\n... and {len(matches) - 5} more matches"
                        
                        await message.reply_text(matches_text, parse_mode=ParseMode.MARKDOWN)
            else:
                await status_msg.edit_text(
                    "❌ **Error:** Reference image not found in database."
                )
                
        except Exception as e:
            logger.error(f"Error processing treasure map image: {e}")
            await status_msg.edit_text(
                "❌ **Error:** Failed to process your image. Please try again."
            )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries (delete confirmations)"""
        query = update.callback_query
        user_id = update.effective_user.id
        
        if user_id not in ADMIN_USER_IDS:
            await query.answer("❌ You don't have permission to use this.")
            return
        
        await query.answer()
        
        if query.data.startswith("delete_"):
            image_id = int(query.data.split("_")[1])
            
            # Delete the image
            if self.db.delete_reference_image(image_id, user_id):
                await query.edit_message_text(
                    "✅ **Reference image deleted successfully!**\n\n"
                    "The image and its extracted text have been removed from the database."
                )
            else:
                await query.edit_message_text(
                    "❌ **Error:** Failed to delete the image.\n\n"
                    "The image may not exist or you don't have permission to delete it."
                )
    
    def run(self):
        """Run the bot"""
        logger.info("Starting Treasure Map Bot...")
        self.application.run_polling()

if __name__ == "__main__":
    bot = TreasureMapBot()
    bot.run()