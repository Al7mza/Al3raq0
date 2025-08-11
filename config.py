import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Bot Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN', '8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY')
ADMIN_USER_IDS = [int(x.strip()) for x in os.getenv('ADMIN_USER_IDS', '1694244496').split(',') if x.strip()]

# File Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Database Configuration
DATABASE_PATH = 'treasure_map.db'

# OCR Configuration
DEFAULT_LANGUAGE = 'eng+chi_sim'  # English + Chinese Simplified
TESSERACT_CONFIG = '--psm 6 --oem 3'

# Google API Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k')

# Similarity Configuration
DEFAULT_SIMILARITY_THRESHOLD = 15.0  # 15% minimum similarity

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)