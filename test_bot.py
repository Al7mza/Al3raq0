#!/usr/bin/env python3
"""
Test script for Treasure Map Bot
This script tests all major components to ensure they're working correctly
"""

import os
import sys
import sqlite3
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import telegram
        print("✅ python-telegram-bot imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import python-telegram-bot: {e}")
        return False
    
    try:
        import pytesseract
        print("✅ pytesseract imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pytesseract: {e}")
        return False
    
    try:
        import cv2
        print("✅ opencv-python imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import opencv-python: {e}")
        return False
    
    try:
        from fuzzywuzzy import fuzz
        print("✅ fuzzywuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fuzzywuzzy: {e}")
        return False
    
    try:
        import dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import python-dotenv: {e}")
        return False
    
    return True

def test_tesseract():
    """Test if Tesseract OCR is working"""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        
        # Check if tesseract is available
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is available")
        
        # Check available languages
        languages = pytesseract.get_languages()
        print(f"✅ Available languages: {', '.join(languages)}")
        
        # Check if Chinese is available
        if 'chi_sim' in languages:
            print("✅ Chinese Simplified language pack available")
        else:
            print("⚠️ Chinese Simplified language pack not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Tesseract OCR test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n🔍 Testing database...")
    
    try:
        from database import Database
        
        # Test database initialization
        db = Database()
        print("✅ Database initialized successfully")
        
        # Test settings
        threshold = db.get_similarity_threshold()
        print(f"✅ Default similarity threshold: {threshold}%")
        
        # Test setting threshold
        db.set_similarity_threshold(20)
        new_threshold = db.get_similarity_threshold()
        if new_threshold == 20:
            print("✅ Threshold setting works correctly")
        else:
            print(f"❌ Threshold setting failed: expected 20, got {new_threshold}")
            return False
        
        # Reset to default
        db.set_similarity_threshold(15)
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_ocr():
    """Test OCR functionality"""
    print("\n🔍 Testing OCR handler...")
    
    try:
        from ocr_handler import OCRHandler
        
        ocr = OCRHandler()
        print("✅ OCR handler initialized successfully")
        
        # Test language setting
        ocr.set_language('eng')
        print("✅ Language setting works")
        
        # Test supported languages
        languages = ocr.get_supported_languages()
        print(f"✅ Supported languages: {', '.join(languages[:5])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_text_matcher():
    """Test text matching functionality"""
    print("\n🔍 Testing text matcher...")
    
    try:
        from text_matcher import TextMatcher
        
        matcher = TextMatcher()
        print("✅ Text matcher initialized successfully")
        
        # Test similarity calculation
        text1 = "Hello World"
        text2 = "Hello World!"
        similarity = matcher.calculate_similarity(text1, text2)
        print(f"✅ Similarity calculation works: {similarity}%")
        
        # Test text normalization
        normalized = matcher.normalize_text("Hello, World!")
        if normalized == "hello world":
            print("✅ Text normalization works")
        else:
            print(f"❌ Text normalization failed: expected 'hello world', got '{normalized}'")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Text matcher test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import UPLOAD_FOLDER, DEFAULT_LANGUAGE, DEFAULT_SIMILARITY_THRESHOLD
        
        print(f"✅ Upload folder: {UPLOAD_FOLDER}")
        print(f"✅ Default language: {DEFAULT_LANGUAGE}")
        print(f"✅ Default threshold: {DEFAULT_SIMILARITY_THRESHOLD}%")
        
        # Check if uploads directory exists
        if os.path.exists(UPLOAD_FOLDER):
            print("✅ Uploads directory exists")
        else:
            print("⚠️ Uploads directory doesn't exist (will be created)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n🔍 Testing image processing...")
    
    try:
        import cv2
        from PIL import Image
        
        # Create a test image
        test_image = np.zeros((100, 300, 3), dtype=np.uint8)
        test_image.fill(255)  # White background
        
        # Test OpenCV
        cv2.imwrite("test_image.jpg", test_image)
        if os.path.exists("test_image.jpg"):
            print("✅ OpenCV image creation works")
            os.remove("test_image.jpg")
        else:
            print("❌ OpenCV image creation failed")
            return False
        
        # Test PIL
        pil_image = Image.new('RGB', (100, 100), color='white')
        pil_image.save("test_pil.jpg")
        if os.path.exists("test_pil.jpg"):
            print("✅ PIL image creation works")
            os.remove("test_pil.jpg")
        else:
            print("❌ PIL image creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Treasure Map Bot - Component Test")
    print("====================================")
    print("")
    
    tests = [
        ("Imports", test_imports),
        ("Tesseract OCR", test_tesseract),
        ("Database", test_database),
        ("OCR Handler", test_ocr),
        ("Text Matcher", test_text_matcher),
        ("Configuration", test_config),
        ("Image Processing", test_image_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The bot should work correctly.")
        print("\n📋 Next steps:")
        print("1. Set up your .env file with bot configuration")
        print("2. Run the bot: python3 bot.py")
        print("3. Test with /start command in Telegram")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the bot.")
        print("\n🔧 Common solutions:")
        print("- Install missing dependencies: pip3 install -r requirements.txt")
        print("- Install Tesseract OCR: sudo apt-get install tesseract-ocr")
        print("- Check Python version (3.8+ required)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)