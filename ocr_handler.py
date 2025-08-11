import cv2
import numpy as np
from PIL import Image
import pytesseract
from config import DEFAULT_LANGUAGE, TESSERACT_CONFIG
import os

class OCRHandler:
    def __init__(self):
        self.language = DEFAULT_LANGUAGE
        self.config = TESSERACT_CONFIG
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply thresholding to get binary image
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up the image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text(self, image_path, language=None):
        """Extract text from image using OCR"""
        try:
            if language is None:
                language = self.language
            
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                # Fallback to original image if preprocessing fails
                processed_image = Image.open(image_path)
            else:
                # Convert OpenCV image to PIL Image
                processed_image = Image.fromarray(processed_image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(
                processed_image,
                lang=language,
                config=self.config
            )
            
            # Clean up the extracted text
            cleaned_text = self.clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_lines = []
        
        for line in lines:
            # Remove common OCR artifacts
            line = line.replace('|', 'I')  # Common OCR mistake
            line = line.replace('0', 'O')  # Common OCR mistake
            line = line.replace('1', 'I')  # Common OCR mistake
            
            # Remove excessive spaces
            line = ' '.join(line.split())
            
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_text_with_confidence(self, image_path, language=None):
        """Extract text with confidence scores"""
        try:
            if language is None:
                language = self.language
            
            # Preprocess the image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                processed_image = Image.open(image_path)
            else:
                processed_image = Image.fromarray(processed_image)
            
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                processed_image,
                lang=language,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            for i, conf in enumerate(data['conf']):
                if conf > 30:  # Filter out low confidence results
                    text_parts.append(data['text'][i])
            
            full_text = ' '.join(text_parts)
            return self.clean_text(full_text)
            
        except Exception as e:
            print(f"Error extracting text with confidence: {e}")
            return ""
    
    def set_language(self, language):
        """Set OCR language"""
        self.language = language
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            print(f"Error getting supported languages: {e}")
            return ['eng', 'chi_sim']