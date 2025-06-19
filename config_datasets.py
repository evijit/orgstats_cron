# --- config_datasets.py ---
"""Configuration file for the Datasets pipeline."""

import pandas as pd

# --- File Paths ---
PROCESSED_PARQUET_FILE_PATH = "datasets_processed.parquet"
HF_PARQUET_URL = 'https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/datasets.parquet'

# --- Column Selection for Raw Data Fetching ---
# Optimized list for datasets, excluding model-specific columns.
RAW_DATA_COLUMNS_TO_FETCH = [
    'id',
    'downloads',
    'downloadsAllTime',
    'likes',
    'tags',
    'lastModified'
]

# --- Tag Mapping for Feature Detection (same as models for now) ---
TAG_MAP = {
    'has_audio': ['audio'], 
    'has_speech': ['speech'], 
    'has_music': ['music'],
    'has_robot': ['robot', 'robotics'], 
    'has_bio': ['bio'], 
    'has_med': ['medic', 'medical'], 
    'has_series': ['series', 'time-series', 'timeseries'], 
    'has_video': ['video'], 
    'has_image': ['image', 'vision'], 
    'has_text': ['text', 'nlp', 'llm'] 
}

# --- Expected Columns for Final Output ---
# Note: 'params' and 'size_category' are removed. 'is_audio_speech' is removed as it depends on pipeline_tag.
FINAL_EXPECTED_COLUMNS = [
    'id', 'downloads', 'downloadsAllTime', 'likes', 'tags',
    'organization',
    'has_audio', 'has_speech', 'has_music', 'has_robot', 'has_bio', 'has_med',
    'has_series', 'has_video', 'has_image', 'has_text', 'has_science',
    'is_biomed',
    'data_download_timestamp'
]

# --- Column Setup Configuration ---
# Simplified for datasets.
EXPECTED_COLUMNS_SETUP = {
    'id': str, 
    'downloads': float, 
    'downloadsAllTime': float, 
    'likes': float,
    'tags': object, 
}

# --- Debugging Configuration ---
MODEL_ID_TO_DEBUG = "lmsys/chatbot_arena_conversations" # Example dataset ID for debugging
