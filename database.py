import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH, DEFAULT_SIMILARITY_THRESHOLD

class Database:
    def __init__(self):
        """Initialize database connection"""
        self.db_path = DATABASE_PATH
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create reference_images table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reference_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    extracted_text TEXT NOT NULL,
                    admin_user_id INTEGER NOT NULL,
                    language TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create settings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            
            # Insert default similarity threshold if not exists
            cursor.execute('''
                INSERT OR IGNORE INTO settings (key, value) 
                VALUES ('similarity_threshold', ?)
            ''', (str(DEFAULT_SIMILARITY_THRESHOLD),))
            
            conn.commit()
    
    def add_reference_image(self, filename, file_path, extracted_text, admin_user_id, language):
        """Add a new reference image"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reference_images (filename, file_path, extracted_text, admin_user_id, language)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, file_path, extracted_text, admin_user_id, language))
            conn.commit()
            return cursor.lastrowid
    
    def delete_reference_image(self, image_id, admin_user_id):
        """Delete a reference image"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get file path before deletion
            cursor.execute('''
                SELECT file_path FROM reference_images 
                WHERE id = ? AND admin_user_id = ?
            ''', (image_id, admin_user_id))
            
            result = cursor.fetchone()
            if not result:
                return False
            
            file_path = result[0]
            
            # Delete from database
            cursor.execute('''
                DELETE FROM reference_images 
                WHERE id = ? AND admin_user_id = ?
            ''', (image_id, admin_user_id))
            
            if cursor.rowcount > 0:
                # Delete physical file
                if os.path.exists(file_path):
                    os.remove(file_path)
                conn.commit()
                return True
            
            return False
    
    def get_all_reference_images(self, admin_user_id):
        """Get all reference images for a specific admin"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_path, extracted_text, language, upload_date
                FROM reference_images 
                WHERE admin_user_id = ?
                ORDER BY upload_date DESC
            ''', (admin_user_id,))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_reference_image_by_id(self, image_id):
        """Get a reference image by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_path, extracted_text, language, upload_date
                FROM reference_images 
                WHERE id = ?
            ''', (image_id,))
            
            result = cursor.fetchone()
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None
    
    def get_all_reference_texts(self):
        """Get all reference texts for matching"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, extracted_text
                FROM reference_images
                ORDER BY upload_date DESC
            ''')
            
            return [{'id': row[0], 'text': row[1]} for row in cursor.fetchall()]
    
    def set_similarity_threshold(self, threshold):
        """Set the similarity threshold"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO settings (key, value) 
                VALUES ('similarity_threshold', ?)
            ''', (str(threshold),))
            conn.commit()
    
    def get_similarity_threshold(self):
        """Get the current similarity threshold"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT value FROM settings WHERE key = 'similarity_threshold'
            ''')
            
            result = cursor.fetchone()
            if result:
                try:
                    return int(result[0])
                except ValueError:
                    return DEFAULT_SIMILARITY_THRESHOLD
            
            return DEFAULT_SIMILARITY_THRESHOLD