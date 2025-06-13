# --- data_fetcher.py ---
"""Module for fetching data from Hugging Face."""

import pandas as pd
import duckdb
import time
from utils import log_progress, log_memory_usage
from config import HF_PARQUET_URL

def fetch_raw_data():
    """Fetch raw data from Hugging Face dataset."""
    log_progress("🚀 Starting data fetch from Hugging Face")
    log_progress(f"Source URL: {HF_PARQUET_URL}")
    
    fetch_start_time = time.time()
    
    try:
        log_progress("⏳ Executing DuckDB query to fetch data...")
        query = f"SELECT * FROM read_parquet('{HF_PARQUET_URL}')"
        df_raw = duckdb.sql(query).df()
        data_download_timestamp = pd.Timestamp.now(tz='UTC')
        
        fetch_time = time.time() - fetch_start_time
        log_progress(f"✅ Data fetch completed in {fetch_time:.2f}s")
        
        # Validate fetched data
        if df_raw is None or df_raw.empty:
            raise ValueError("Fetched data is empty or None")
        
        if 'id' not in df_raw.columns:
            raise ValueError("Fetched data must contain 'id' column")
        
        # Log data statistics
        log_progress(f"📊 Data statistics:")
        log_progress(f"   - Rows: {len(df_raw):,}")
        log_progress(f"   - Columns: {len(df_raw.columns)}")
        log_progress(f"   - Column names: {list(df_raw.columns)}")
        log_progress(f"   - Download timestamp: {data_download_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Check for null values in key columns
        key_columns = ['id', 'downloads', 'likes']
        for col in key_columns:
            if col in df_raw.columns:
                null_count = df_raw[col].isnull().sum()
                total_count = len(df_raw)
                null_pct = (null_count / total_count) * 100
                log_progress(f"   - {col}: {null_count:,} nulls ({null_pct:.1f}%)")
        
        log_memory_usage()
        
        return df_raw, data_download_timestamp
        
    except Exception as e:
        log_progress(f"❌ ERROR: Could not fetch data from Hugging Face: {e}")
        raise

def validate_raw_data(df_raw):
    """Perform additional validation on raw data."""
    log_progress("🔍 Validating raw data quality...")
    
    validation_start = time.time()
    
    # Check for completely empty rows
    empty_rows = df_raw.isnull().all(axis=1).sum()
    log_progress(f"   - Empty rows: {empty_rows:,}")
    
    # Check for duplicate IDs
    if 'id' in df_raw.columns:
        duplicate_ids = df_raw['id'].duplicated().sum()
        log_progress(f"   - Duplicate IDs: {duplicate_ids:,}")
        if duplicate_ids > 0:
            log_progress("   ⚠️  WARNING: Found duplicate model IDs")
    
    # Check data types
    log_progress("   - Data types:")
    for col in df_raw.columns:
        dtype = df_raw[col].dtype
        log_progress(f"     {col}: {dtype}")
    
    validation_time = time.time() - validation_start
    log_progress(f"✅ Data validation completed in {validation_time:.2f}s")
    
    return True

if __name__ == "__main__":
    # Test the data fetcher
    try:
        df_raw, timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
        log_progress(f"✅ Data fetcher test successful - {len(df_raw):,} rows fetched")
    except Exception as e:
        log_progress(f"❌ Data fetcher test failed: {e}")
        raise