# --- data_processor_datasets.py ---
"""Main data processing module for Datasets."""

import pandas as pd
import time
from utils import log_progress, log_memory_usage, extract_org_from_id, validate_dataframe_structure
from config_datasets import EXPECTED_COLUMNS_SETUP

def setup_initial_dataframe(df_raw, data_download_timestamp):
    """Set up initial DataFrame for datasets."""
    log_progress("🔧 Setting up initial DataFrame structure for Datasets...")
    df = pd.DataFrame()
    
    for col_name, target_dtype in EXPECTED_COLUMNS_SETUP.items():
        if col_name in df_raw.columns:
            df[col_name] = df_raw[col_name]
            if target_dtype == float:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0)
            elif target_dtype == str:
                df[col_name] = df[col_name].astype(str).fillna('')
        else:
            log_progress(f"     Column {col_name} missing, creating default values")
            if col_name in ['downloads', 'downloadsAllTime', 'likes']:
                df[col_name] = 0.0
            elif col_name == 'tags':
                df[col_name] = pd.Series([[] for _ in range(len(df_raw))])
            elif col_name == 'id':
                raise ValueError("'id' column is required but missing from source data")

    df['data_download_timestamp'] = data_download_timestamp
    validate_dataframe_structure(df, list(EXPECTED_COLUMNS_SETUP.keys()))
    log_memory_usage()
    return df

def enrich_data(df):
    """Add organization column to dataset data."""
    log_progress("✨ Enriching dataset data with organization...")
    df['organization'] = df['id'].apply(extract_org_from_id)
    org_count = df['organization'].nunique()
    log_progress(f"   Found {org_count:,} unique organizations.")
    log_memory_usage()
    return df
