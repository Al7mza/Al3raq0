#!/usr/bin/env python3
"""
Test script for Treasure Map Bot components
"""

def test_imports():
    """Test if all modules can be imported"""
    try:
        from config import BOT_TOKEN, ADMIN_USER_IDS
        print("✅ Config module imported successfully")
        
        from database import Database
        print("✅ Database module imported successfully")
        
        from ocr_handler import OCRHandler
        print("✅ OCR handler module imported successfully")
        
        from text_matcher import TextMatcher
        print("✅ Text matcher module imported successfully")
        
        from bot import TreasureMapBot
        print("✅ Bot module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import BOT_TOKEN, ADMIN_USER_IDS
        print(f"✅ Bot token loaded: {'Yes' if BOT_TOKEN else 'No'}")
        print(f"✅ Admin user IDs: {ADMIN_USER_IDS}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_database():
    """Test database functionality"""
    try:
        from database import Database
        db = Database()
        db.init_database()
        threshold = db.get_similarity_threshold()
        print(f"✅ Database initialized successfully")
        print(f"✅ Default threshold: {threshold}%")
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Treasure Map Bot Components...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Database Test", test_database),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The bot is ready to run.")
        print("\n📋 Next steps:")
        print("1. Make sure your .env file has the correct bot token")
        print("2. Update admin user ID in .env file")
        print("3. Run: python3 bot.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()