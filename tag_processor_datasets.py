# --- tag_processor_datasets.py ---
"""Module for processing tags and creating feature flags for Datasets."""

import pandas as pd
import time
from utils import log_progress
from config_datasets import TAG_MAP
# The core tag parsing function can be reused from the main tag processor
from tag_processor import process_tags_for_series, analyze_tag_distribution

def create_feature_flags(df):
    """Create boolean feature flags for datasets based on tags."""
    log_progress("🚩 Creating feature flags from dataset tags...")
    flag_start = time.time()
    
    df['temp_tags_joined'] = df['tags'].apply(
        lambda tl: '~~~'.join(str(t).lower().strip() for t in tl if pd.notna(t)) if isinstance(tl, list) else ''
    )
    
    for col, keywords in TAG_MAP.items():
        pattern = '|'.join(keywords)
        df[col] = df['temp_tags_joined'].str.contains(pattern, na=False, case=False, regex=True)
    
    df['has_science'] = (
        df['temp_tags_joined'].str.contains('science', na=False, case=False, regex=True) &
        ~df['temp_tags_joined'].str.contains('bigscience', na=False, case=False, regex=True)
    )
    
    df['is_biomed'] = df['has_bio'] | df['has_med']
    del df['temp_tags_joined']
    
    flag_time = time.time() - flag_start
    log_progress(f"✅ Dataset feature flags created in {flag_time:.2f}s")
    return df
