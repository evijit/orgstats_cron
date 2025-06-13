# --- config.py ---
"""Configuration file containing all constants and settings."""

import pandas as pd

# --- File Paths ---
PROCESSED_PARQUET_FILE_PATH = "models_processed.parquet"
HF_PARQUET_URL = 'https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/models.parquet'

# --- Model Size Categories ---
MODEL_SIZE_RANGES = {
    "Small (<1GB)": (0, 1),
    "Medium (1-5GB)": (1, 5),
    "Large (5-20GB)": (5, 20),
    "X-Large (20-50GB)": (20, 50),
    "XX-Large (>50GB)": (50, float('inf'))
}

# --- Tag Mapping for Feature Detection ---
TAG_MAP = {
    'has_audio': ['audio'], 
    'has_speech': ['speech'], 
    'has_music': ['music'],
    'has_robot': ['robot', 'robotics', 'openvla', 'vla'], 
    'has_bio': ['bio'], 
    'has_med': ['medic', 'medical'], 
    'has_series': ['series', 'time-series', 'timeseries'], 
    'has_video': ['video'], 
    'has_image': ['image', 'vision'], 
    'has_text': ['text', 'nlp', 'llm'] 
}

# --- Expected Columns for Final Output ---
FINAL_EXPECTED_COLUMNS = [
    'id', 'downloads', 'downloadsAllTime', 'likes', 'pipeline_tag', 'tags',
    'params', 'size_category', 'organization',
    'has_audio', 'has_speech', 'has_music', 'has_robot', 'has_bio', 'has_med',
    'has_series', 'has_video', 'has_image', 'has_text', 'has_science',
    'is_audio_speech', 'is_biomed',
    'data_download_timestamp'
]

# --- Column Setup Configuration ---
EXPECTED_COLUMNS_SETUP = {
    'id': str, 
    'downloads': float, 
    'downloadsAllTime': float, 
    'likes': float,
    'pipeline_tag': str, 
    'tags': object, 
    'safetensors': object
}

# --- Debugging Configuration ---
MODEL_ID_TO_DEBUG = "openvla/openvla-7b"  # Set to None to disable debugging