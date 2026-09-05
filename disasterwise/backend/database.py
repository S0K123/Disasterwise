import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "disasterwise.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pre_image_path TEXT,
            post_image_path TEXT,
            damage_percentage REAL,
            alert_level TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def store_prediction(pre_image_path, post_image_path, damage_percentage, alert_level, message):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (pre_image_path, post_image_path, damage_percentage, alert_level, message)
        VALUES (?, ?, ?, ?, ?)
    ''', (pre_image_path, post_image_path, damage_percentage, alert_level, message))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

if __name__ == "__main__":
    init_db()
