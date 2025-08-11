import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH

class Database:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create reference_images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reference_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                language TEXT DEFAULT 'chi_sim+eng',
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                admin_user_id INTEGER NOT NULL
            )
        ''')
        
        # Create settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default similarity threshold
        cursor.execute('''
            INSERT OR IGNORE INTO settings (key, value) VALUES ('similarity_threshold', '15')
        ''')
        
        conn.commit()
        conn.close()
    
    def add_reference_image(self, filename, file_path, extracted_text, admin_user_id, language='chi_sim+eng'):
        """Add a new reference image to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reference_images (filename, file_path, extracted_text, language, admin_user_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (filename, file_path, extracted_text, language, admin_user_id))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
    
    def delete_reference_image(self, image_id, admin_user_id):
        """Delete a reference image from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get file path before deletion
        cursor.execute('SELECT file_path FROM reference_images WHERE id = ? AND admin_user_id = ?', 
                      (image_id, admin_user_id))
        result = cursor.fetchone()
        
        if result:
            file_path = result[0]
            # Delete from database
            cursor.execute('DELETE FROM reference_images WHERE id = ? AND admin_user_id = ?', 
                         (image_id, admin_user_id))
            conn.commit()
            conn.close()
            
            # Delete physical file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return True
        
        conn.close()
        return False
    
    def get_all_reference_images(self, admin_user_id):
        """Get all reference images for an admin"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, extracted_text, upload_date 
            FROM reference_images 
            WHERE admin_user_id = ?
            ORDER BY upload_date DESC
        ''', (admin_user_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_reference_image_by_id(self, image_id):
        """Get a specific reference image by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_path, extracted_text, language
            FROM reference_images 
            WHERE id = ?
        ''', (image_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result
    
    def get_all_reference_texts(self):
        """Get all reference texts for similarity comparison"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, extracted_text FROM reference_images')
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def set_similarity_threshold(self, threshold):
        """Set the similarity threshold"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO settings (key, value, updated_date)
            VALUES ('similarity_threshold', ?)
        ''', (str(threshold),))
        
        conn.commit()
        conn.close()
    
    def get_similarity_threshold(self):
        """Get the current similarity threshold"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT value FROM settings WHERE key = ?', ('similarity_threshold',))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return int(result[0])
        return 15  # Default threshold