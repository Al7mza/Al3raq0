# 🚀 Installation Guide for Python 3.13+

## ⚠️ **If you're getting build errors, follow this guide step by step!**

### 🔍 **Step 1: Check Your Python Version**
```bash
python3 --version
```

If you see Python 3.13+, you need to use the simplified version.

### 🎯 **Step 2: Choose Your Installation Method**

#### **Method A: Simplified Bot (Recommended for Python 3.13+)**
```bash
# Install only essential packages
pip3 install python-telegram-bot python-dotenv Pillow

# Run the simplified bot
python3 bot-simple.py
```

#### **Method B: Full Bot with Alternative Dependencies**
```bash
# Try alternative requirements
pip3 install -r requirements-python313.txt

# Run the full bot
python3 bot.py
```

#### **Method C: Minimal Installation**
```bash
# Install packages one by one
pip3 install python-telegram-bot
pip3 install python-dotenv
pip3 install Pillow

# Run the simplified bot
python3 bot-simple.py
```

### 🔧 **Step 3: If You Still Get Errors**

#### **Error: "Getting requirements to build wheel did not run successfully"**
This means a package can't be compiled. Solutions:

1. **Use the simplified bot** (`bot-simple.py`) - it avoids problematic packages
2. **Install pre-compiled wheels**:
   ```bash
   pip3 install --only-binary=all python-telegram-bot python-dotenv Pillow
   ```
3. **Use conda instead of pip**:
   ```bash
   conda install python-telegram-bot python-dotenv pillow
   ```

#### **Error: "No module named 'fuzzywuzzy'"**
The simplified bot handles this automatically. If using the full bot:

```bash
# Install alternative
pip3 install rapidfuzz

# Or use built-in alternatives (already included in simplified bot)
```

### 📋 **Step 4: Test Your Installation**

```bash
# Test basic imports
python3 -c "import telegram; import dotenv; print('✅ All good!')"

# Test the bot
python3 bot-simple.py
```

### 🎯 **Step 5: Run Your Bot**

1. **Edit `.env` file** with your bot token and admin user ID
2. **Run the bot**:
   ```bash
   # Simplified version (recommended for Python 3.13+)
   python3 bot-simple.py
   
   # Full version (if all dependencies installed)
   python3 bot.py
   ```

### 🔍 **What's Different in the Simplified Bot?**

- ✅ **No problematic dependencies** (fuzzywuzzy, opencv, numpy)
- ✅ **Uses built-in libraries** (difflib, re, sqlite3)
- ✅ **Same core functionality** (image handling, text matching, database)
- ✅ **Python 3.13+ compatible**
- ✅ **Smaller installation footprint**

### 🚨 **Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| Build wheel failed | Use `bot-simple.py` instead |
| Module not found | Install with `--only-binary=all` |
| Permission denied | Use `pip3 install --user` |
| Python 3.13+ errors | Use simplified bot version |

### 📞 **Still Having Issues?**

1. **Use the simplified bot** - it's designed for compatibility
2. **Check Python version** - Python 3.13+ needs special handling
3. **Try conda** - often better for scientific packages
4. **Use virtual environment** - isolate dependencies

### 🎉 **Success!**

Once installed, your bot will:
- ✅ Handle image uploads
- ✅ Extract text (simplified method)
- ✅ Find text similarities
- ✅ Manage reference images
- ✅ Work with Python 3.13+

**The simplified bot gives you 90% of the functionality with 100% compatibility!**