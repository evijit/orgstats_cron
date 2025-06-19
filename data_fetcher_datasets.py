# --- data_fetcher_datasets.py ---
"""Module for fetching dataset data from Hugging Face."""

import os
import time
import duckdb
import pandas as pd
from utils import log_progress, log_memory_usage
from config_datasets import HF_PARQUET_URL, RAW_DATA_COLUMNS_TO_FETCH

def fetch_raw_data():
    """
    Fetch raw dataset data from Hugging Face, selecting only necessary columns.
    Respects 'TEST_DATA_LIMIT' environment variable for testing.
    """
    log_progress("🚀 Starting DATASET data fetch from Hugging Face")
    log_progress(f"Source URL: {HF_PARQUET_URL}")
    
    fetch_start_time = time.time()
    
    try:
        columns_to_select = ", ".join(f'"{col}"' for col in RAW_DATA_COLUMNS_TO_FETCH)
        query = f"SELECT {columns_to_select} FROM read_parquet('{HF_PARQUET_URL}')"
        log_progress(f"Optimized query will fetch {len(RAW_DATA_COLUMNS_TO_FETCH)} specific columns.")
        
        limit = os.environ.get('TEST_DATA_LIMIT')
        if limit and limit.isdigit():
            query += f" LIMIT {int(limit)}"
            log_progress(f"🧪 Applying test limit: Fetching only {limit} rows.")
            
        log_progress("⏳ Executing DuckDB query to fetch remote dataset data...")
        df_raw = duckdb.sql(query).df()
        data_download_timestamp = pd.Timestamp.now(tz='UTC')
        
        fetch_time = time.time() - fetch_start_time
        log_progress(f"✅ Dataset data fetch completed in {fetch_time:.2f}s")
        
        if df_raw is None or df_raw.empty:
            raise ValueError("Fetched dataset data is empty or None")
        
        log_progress(f"📊 Rows: {len(df_raw):,}, Columns: {len(df_raw.columns)}")
        log_memory_usage()
        
        return df_raw, data_download_timestamp
        
    except Exception as e:
        log_progress(f"❌ ERROR: Could not fetch dataset data: {e}")
        raise

def validate_raw_data(df_raw):
    """Perform validation on raw dataset data."""
    log_progress("🔍 Validating raw dataset data quality...")
    if 'id' not in df_raw.columns:
        raise ValueError("Critical 'id' column is missing from fetched dataset data.")
    log_progress(f"   - Duplicate IDs: {df_raw['id'].duplicated().sum():,}")
    log_progress("✅ Dataset data validation completed.")
    return True
