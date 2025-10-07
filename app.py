# app.py
import streamlit as st
import sqlite3
import tempfile
from datetime import datetime
from ocr_module import analyze_image

DB_PATH = "readings.db"  # Relative path for Render

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            filepath TEXT,
            label TEXT,
            value REAL,
            raw_text TEXT,
            annotated_path TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(category, filepath, label, value, raw_text, annotated_path):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO readings (category, filepath, label, value, raw_text, annotated_path, timestamp) VALUES (?,?,?,?,?,?,?)",
        (category, filepath, label, value, raw_text, annotated_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

st.set_page_config(page_title="Control Panel OCR", layout="centered")
st.title("📊 Control Panel OCR Application")

init_db()

category = st.selectbox("Select Input Type", ["genset", "mri", "electrical"])
st.write("Upload an image or capture using your camera (if available).")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Take a picture")
file_obj = uploaded_file or camera_file

if file_obj is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_obj._
