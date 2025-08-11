# 🗺️ Treasure Map Bot

A powerful Telegram bot for storing and searching treasure map images with intelligent text matching.

## 🌟 Features

- 📤 **Photo Upload**: Upload treasure map images
- 🔍 **Smart Search**: Search through maps using text queries
- 📋 **Map Management**: List all maps and your personal maps
- 💾 **Database Storage**: SQLite database for reliable data storage
- 📊 **Search History**: Track search queries and results
- 🌐 **Arabic Support**: Full Arabic language support
- 🔧 **Easy Setup**: Simple installation and configuration

## 🚀 Quick Start

### **Step 1: Download Files**
Download all bot files to your hosting environment.

### **Step 2: Install Dependencies**
```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Or install manually
pip3 install python-telegram-bot
```

### **Step 3: Run Your Bot**
```bash
python3 treasure_map_bot.py
```

## 📋 Commands

- `/start` - Welcome message and bot introduction
- `/help` - Help and usage instructions
- `/search <text>` - Search for maps containing specific text
- `/list` - Show all uploaded maps
- `/mymaps` - Show your personal maps

## 🔧 Configuration

The bot is pre-configured with your settings:

- **Bot Token**: `8217318799:AAF6SEzDub4f3QK7P5p76QL4uBMwalqI7WY`
- **Admin ID**: `1694244496`
- **Google API Key**: `AIzaSyBiW7vLyDMRpcZcWdawzmPXjZqL5rSsl1k`

## 📁 File Structure

```
treasure-map-bot/
├── treasure_map_bot.py    # Main bot file
├── requirements.txt        # Python dependencies
├── setup.sh              # Setup script
├── README.md             # This file
├── uploads/              # Image storage folder
└── treasure_map.db       # Database (created automatically)
```

## 🛠️ Requirements

- **Python**: 3.7 or higher
- **Dependencies**: `python-telegram-bot`
- **Storage**: At least 100MB free space
- **Internet**: Stable connection for Telegram API

## 🔍 How It Works

1. **Upload**: Users send treasure map images
2. **Process**: Bot downloads and stores images locally
3. **Extract**: Bot generates descriptive text for each image
4. **Store**: All data saved to SQLite database
5. **Search**: Users can search through maps using text queries
6. **Match**: Intelligent text similarity matching finds relevant results

## 📊 Database Schema

### Treasure Maps Table
- `id`: Unique identifier
- `user_id`: Telegram user ID
- `username`: Telegram username
- `first_name`: User's first name
- `file_path`: Local file path
- `file_size`: File size in bytes
- `text_content`: Extracted/descriptive text
- `created_at`: Upload timestamp

### Search History Table
- `id`: Unique identifier
- `user_id`: User who performed search
- `query`: Search query text
- `results_count`: Number of results found
- `searched_at`: Search timestamp

## 🚨 Troubleshooting

### **Bot Won't Start**
1. Check Python version: `python3 --version`
2. Verify dependencies: `pip3 list | grep telegram`
3. Check bot token validity
4. Ensure internet connection

### **Photos Not Saving**
1. Check `uploads/` folder permissions
2. Verify disk space availability
3. Check file size limits (10MB max)

### **Search Not Working**
1. Ensure maps have been uploaded
2. Check database file exists
3. Verify text content was generated

### **Permission Errors**
```bash
chmod +x setup.sh
chmod +x treasure_map_bot.py
chmod 755 uploads/
```

## 🔒 Security Features

- File size limits (10MB max)
- Allowed file extensions only
- User authentication via Telegram
- Secure file naming conventions
- Database injection protection

## 📈 Performance Tips

- Keep image sizes reasonable
- Use descriptive text for better search
- Regular database maintenance
- Monitor disk space usage

## 🤝 Support

If you encounter issues:

1. Check the logs in `bot.log`
2. Verify all dependencies are installed
3. Ensure proper file permissions
4. Check internet connectivity

## 📝 License

This project is open source and available under the MIT License.

## 🎯 What's Next?

- Add real OCR capabilities
- Implement image compression
- Add backup functionality
- Create web dashboard
- Add user management features

---

**Made with ❤️ for treasure hunters everywhere!**