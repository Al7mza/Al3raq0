import cv2
import numpy as np
import pytesseract
from PIL import Image
from config import DEFAULT_LANGUAGE, TESSERACT_CONFIG

class OCRHandler:
    def __init__(self, language=None):
        """Initialize OCR handler"""
        self.language = language or DEFAULT_LANGUAGE
        self.tesseract_config = TESSERACT_CONFIG
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return morph
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            # Return original image if preprocessing fails
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def extract_text(self, image_path, language=None):
        """Extract text from image using OCR"""
        try:
            # Use specified language or default
            lang = language or self.language
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang,
                config=self.tesseract_config
            )
            
            # Clean extracted text
            cleaned_text = self.clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def extract_text_with_confidence(self, image_path, language=None):
        """Extract text with confidence scores"""
        try:
            lang = language or self.language
            processed_image = self.preprocess_image(image_path)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                processed_image,
                lang=lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            for i, conf in enumerate(data['conf']):
                if conf > 0:  # Filter out low confidence results
                    text_parts.append(data['text'][i])
            
            full_text = ' '.join(text_parts)
            return self.clean_text(full_text)
            
        except Exception as e:
            print(f"Error extracting text with confidence: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common OCR mistake
        text = text.replace('0', 'O')  # Common OCR mistake
        text = text.replace('1', 'I')  # Common OCR mistake
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text.strip()
    
    def set_language(self, language):
        """Set OCR language"""
        self.language = language
    
    def get_language(self):
        """Get current OCR language"""
        return self.language
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            print(f"Error getting supported languages: {e}")
            return ['eng']  # Default to English if error
    
    def test_ocr(self, image_path):
        """Test OCR functionality with a sample image"""
        try:
            text = self.extract_text(image_path)
            return {
                'success': True,
                'text': text,
                'language': self.language,
                'text_length': len(text)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'language': self.language
            }