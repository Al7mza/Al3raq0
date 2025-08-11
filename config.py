import os
from dotenv import load_dotenv

load_dotenv()

# Bot Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
ADMIN_USER_IDS = [int(id.strip()) for id in os.getenv('ADMIN_USER_IDS', '').split(',') if id.strip()]

# OCR Configuration
DEFAULT_LANGUAGE = 'chi_sim+eng'  # Chinese simplified + English
TESSERACT_CONFIG = '--psm 6 --oem 3'

# Similarity Configuration
DEFAULT_SIMILARITY_THRESHOLD = 15  # 15%

# Database Configuration
DATABASE_PATH = 'treasure_maps.db'

# File Storage
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)