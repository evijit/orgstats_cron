# --- data_fetcher_papers.py ---
"""Module for fetching papers data from Hugging Face."""

import os
import random
import time
import duckdb
import pandas as pd
from utils import log_progress, log_memory_usage
from config_papers import HF_PARQUET_URL, RAW_DATA_COLUMNS_TO_FETCH

_MAX_FETCH_RETRIES = 6
_FETCH_BASE_DELAY = 30  # seconds


def fetch_raw_data():
    """
    Fetch raw papers data from Hugging Face, selecting only necessary columns.
    Respects 'TEST_DATA_LIMIT' environment variable for testing.
    Retries with exponential backoff on HTTP 429 (rate limit) errors.
    """
    log_progress("🚀 Starting PAPERS data fetch from Hugging Face")
    log_progress(f"Source URL: {HF_PARQUET_URL}")

    # Startup jitter: only applied by parallel wave jobs (HF_STARTUP_JITTER=1)
    if os.environ.get('HF_STARTUP_JITTER') == '1':
        jitter = random.uniform(0, 20)
        log_progress(f"⏳ Startup jitter: waiting {jitter:.1f}s to reduce parallel request collisions...")
        time.sleep(jitter)

    fetch_start_time = time.time()

    try:
        columns_to_select = ", ".join(f'"{col}"' for col in RAW_DATA_COLUMNS_TO_FETCH)
        query = f"SELECT {columns_to_select} FROM read_parquet('{HF_PARQUET_URL}')"
        log_progress(f"Optimized query will fetch {len(RAW_DATA_COLUMNS_TO_FETCH)} specific columns.")

        limit = os.environ.get('TEST_DATA_LIMIT')
        if limit and limit.isdigit():
            query += f" LIMIT {int(limit)}"
            log_progress(f"🧪 Applying test limit: Fetching only {limit} rows.")

        with duckdb.connect() as conn:
            hf_token = os.environ.get('HF_TOKEN')
            if hf_token:
                try:
                    conn.execute("INSTALL httpfs; LOAD httpfs;")
                    # Parameterized to avoid token being interpolated into SQL
                    conn.execute("CREATE OR REPLACE SECRET hf_secret (TYPE HTTP, BEARER_TOKEN ?);", [hf_token])
                    log_progress("🔑 Using HF_TOKEN for authenticated HuggingFace access.")
                except Exception:
                    log_progress("⚠️  Could not configure HF_TOKEN for DuckDB connection; proceeding unauthenticated.")

            log_progress("⏳ Executing DuckDB query to fetch remote papers data...")
            df_raw = None
            for attempt in range(_MAX_FETCH_RETRIES):
                try:
                    df_raw = conn.execute(query).df()
                    break
                except Exception as e:
                    if '429' in str(e) and attempt < _MAX_FETCH_RETRIES - 1:
                        delay = _FETCH_BASE_DELAY * (2 ** attempt) + random.uniform(0, 15)
                        log_progress(f"⏳ Rate limited (HTTP 429). Retrying in {delay:.1f}s "
                                     f"(attempt {attempt + 1}/{_MAX_FETCH_RETRIES})...")
                        time.sleep(delay)
                    else:
                        raise

        data_download_timestamp = pd.Timestamp.now(tz='UTC')
        
        fetch_time = time.time() - fetch_start_time
        log_progress(f"✅ Papers data fetch completed in {fetch_time:.2f}s")
        
        if df_raw is None or df_raw.empty:
            raise ValueError("Fetched papers data is empty or None")
        
        log_progress(f"📊 Rows: {len(df_raw):,}, Columns: {len(df_raw.columns)}")
        log_memory_usage()
        
        return df_raw, data_download_timestamp
        
    except Exception as e:
        log_progress(f"❌ ERROR: Could not fetch papers data: {e}")
        raise

def validate_raw_data(df_raw):
    """Perform validation on raw papers data."""
    log_progress("🔍 Validating raw papers data quality...")
    
    if 'paper_id' not in df_raw.columns:
        raise ValueError("Critical 'paper_id' column is missing from fetched papers data.")
    
    log_progress(f"   - Duplicate IDs: {df_raw['paper_id'].duplicated().sum():,}")
    
    # Check paper_ai_keywords column
    if 'paper_ai_keywords' in df_raw.columns:
        non_null_keywords = df_raw['paper_ai_keywords'].notna().sum()
        log_progress(f"   - Papers with keywords: {non_null_keywords:,} ({non_null_keywords/len(df_raw)*100:.1f}%)")
    
    log_progress("✅ Papers data validation completed.")
    return True

if __name__ == "__main__":
    try:
        log_progress("Running data_fetcher_papers.py directly...")
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
