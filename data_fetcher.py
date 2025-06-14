# --- data_fetcher.py (UPDATED) ---
"""Module for fetching data from Hugging Face."""

import os
import time
import duckdb
import pandas as pd
from utils import log_progress, log_memory_usage
# --- CHANGE: Import the new column list from config ---
from config import HF_PARQUET_URL, RAW_DATA_COLUMNS_TO_FETCH

def fetch_raw_data():
    """
    Fetch raw data from Hugging Face dataset.
    This is optimized to only select the necessary columns.
    If 'TEST_DATA_LIMIT' env var is set, it will fetch only that many rows.
    """
    log_progress("🚀 Starting data fetch from Hugging Face")
    log_progress(f"Source URL: {HF_PARQUET_URL}")
    
    fetch_start_time = time.time()
    
    try:
        # --- CHANGE: Build the query dynamically from the config list ---
        columns_to_select = ", ".join(f'"{col}"' for col in RAW_DATA_COLUMNS_TO_FETCH)
        query = f"SELECT {columns_to_select} FROM read_parquet('{HF_PARQUET_URL}')"
        log_progress(f"Optimized query will fetch {len(RAW_DATA_COLUMNS_TO_FETCH)} specific columns.")
        
        # Check for test mode limit
        limit = os.environ.get('TEST_DATA_LIMIT')
        if limit and limit.isdigit():
            query += f" LIMIT {int(limit)}"
            log_progress(f"🧪 Applying test limit: Fetching only {limit} rows.")
        # --- END OF CHANGE ---
            
        log_progress("⏳ Executing DuckDB query to fetch remote data... (This will be much faster now)")
        
        df_raw = duckdb.sql(query).df()
        data_download_timestamp = pd.Timestamp.now(tz='UTC')
        
        fetch_time = time.time() - fetch_start_time
        log_progress(f"✅ Data fetch completed in {fetch_time:.2f}s")
        
        if df_raw is None or df_raw.empty:
            raise ValueError("Fetched data is empty or None")
        
        log_progress(f"📊 Rows: {len(df_raw):,}, Columns: {len(df_raw.columns)}")
        log_memory_usage()
        
        return df_raw, data_download_timestamp
        
    except Exception as e:
        log_progress(f"❌ ERROR: Could not fetch data from Hugging Face: {e}")
        raise

def validate_raw_data(df_raw):
    """Perform additional validation on raw data."""
    log_progress("🔍 Validating raw data quality...")
    
    validation_start = time.time()
    
    if 'id' not in df_raw.columns:
        raise ValueError("Critical 'id' column is missing from fetched data.")
    
    log_progress(f"   - Duplicate IDs: {df_raw['id'].duplicated().sum():,}")
    
    validation_time = time.time() - validation_start
    log_progress(f"✅ Data validation completed in {validation_time:.2f}s")
    
    return True

if __name__ == "__main__":
    try:
        log_progress("Running data_fetcher.py directly (optimized full dataset)...")
        df_raw, timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
        log_progress(f"✅ Data fetcher direct run successful - {len(df_raw):,} rows fetched")
        print("\nFetched Columns:")
        print(df_raw.columns.tolist())
        print("\nSample Data:")
        print(df_raw.head().to_string())
    except Exception as e:
        log_progress(f"❌ Data fetcher direct run failed: {e}")
        raise