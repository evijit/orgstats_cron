# --- data_processor.py ---
"""Main data processing module."""

import pandas as pd
import time
from utils import log_progress, log_memory_usage, extract_model_file_size_gb, extract_org_from_id, get_file_size_category, validate_dataframe_structure
from config import EXPECTED_COLUMNS_SETUP, FINAL_EXPECTED_COLUMNS

def setup_initial_dataframe(df_raw, data_download_timestamp):
    """Set up initial DataFrame with proper column types and defaults."""
    log_progress("🔧 Setting up initial DataFrame structure...")
    setup_start = time.time()
    
    df = pd.DataFrame()
    
    # Process expected columns
    for col_name, target_dtype in EXPECTED_COLUMNS_SETUP.items():
        log_progress(f"   Processing column: {col_name}")
        
        if col_name in df_raw.columns:
            df[col_name] = df_raw[col_name]
            
            # Apply type conversions
            if target_dtype == float:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0)
                log_progress(f"     Converted to float, filled NaN with 0.0")
            elif target_dtype == str:
                df[col_name] = df[col_name].astype(str).fillna('')
                log_progress(f"     Converted to string, filled NaN with empty string")
        else:
            # Create default values for missing columns
            log_progress(f"     Column missing, creating default values")
            if col_name in ['downloads', 'downloadsAllTime', 'likes']:
                df[col_name] = 0.0
            elif col_name == 'pipeline_tag':
                df[col_name] = ''
            elif col_name == 'tags':
                df[col_name] = pd.Series([[] for _ in range(len(df_raw))])
            elif col_name == 'safetensors':
                df[col_name] = None
            elif col_name == 'id':
                log_progress("❌ CRITICAL ERROR: 'id' column missing from source data")
                raise ValueError("'id' column is required but missing from source data")
    
    # Add timestamp
    df['data_download_timestamp'] = data_download_timestamp
    log_progress(f"   Added download timestamp: {data_download_timestamp}")
    
    setup_time = time.time() - setup_start
    log_progress(f"✅ DataFrame setup completed in {setup_time:.2f}s")
    
    # Validate structure
    validate_dataframe_structure(df)
    log_memory_usage()
    
    return df

def calculate_file_sizes(df, df_raw):
    """Calculate file sizes for models."""
    log_progress("📏 Calculating model file sizes...")
    size_start = time.time()
    
    output_filesize_col_name = 'params'
    
    # Check if params column already exists as numeric
    if (output_filesize_col_name in df_raw.columns and 
        pd.api.types.is_numeric_dtype(df_raw[output_filesize_col_name])):
        
        log_progress(f"   Using pre-existing '{output_filesize_col_name}' column")
        df[output_filesize_col_name] = pd.to_numeric(df_raw[output_filesize_col_name], errors='coerce').fillna(0.0)
        
    elif 'safetensors' in df.columns:
        log_progress(f"   Calculating '{output_filesize_col_name}' from 'safetensors' data...")