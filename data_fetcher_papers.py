# --- data_fetcher_papers.py ---
"""Module for fetching papers data from Hugging Face."""

import json
import os
import random
import time
import duckdb
import pandas as pd
from utils import log_progress, log_memory_usage
from config_papers import HF_PARQUET_URL, RAW_DATA_COLUMNS_TO_FETCH

_MAX_FETCH_RETRIES = 8
_FETCH_BASE_DELAY = 30   # seconds
_MAX_FETCH_DELAY = 300   # cap on a single backoff sleep

# Transient conditions worth retrying. HTTP 429 from huggingface.co is by far the
# most common one; the others are ordinary network flakiness.
_RETRYABLE_MARKERS = ('429', '500', '502', '503', '504', 'timeout', 'timed out',
                      'connection', 'temporarily', 'reset by peer')

# A snapshot of the papers parquet, fetched once per workflow run and shared with
# every parallel citation job as a build artifact. Fetching it once instead of
# once per job is what keeps HuggingFace from rate-limiting the run.
SNAPSHOT_FILE = os.environ.get('PAPERS_SNAPSHOT_FILE', 'papers_raw.parquet')
SNAPSHOT_META_FILE = os.environ.get('PAPERS_SNAPSHOT_META_FILE', 'papers_raw_meta.json')


def _is_retryable(error):
    """Whether a failed fetch is worth another attempt."""
    message = str(error).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def _apply_test_limit(df_raw):
    """Honour TEST_DATA_LIMIT when reading a snapshot (the SQL path uses LIMIT)."""
    limit = os.environ.get('TEST_DATA_LIMIT')
    if limit and limit.isdigit():
        log_progress(f"🧪 Applying test limit: keeping only {limit} rows.")
        return df_raw.head(int(limit)).copy()
    return df_raw


def _load_snapshot():
    """Load the run-local snapshot instead of re-downloading from HuggingFace."""
    log_progress(f"📂 Reusing local papers snapshot '{SNAPSHOT_FILE}' (skipping HuggingFace fetch).")
    df_raw = _apply_test_limit(pd.read_parquet(SNAPSHOT_FILE))

    if df_raw.empty:
        raise ValueError(f"Local papers snapshot '{SNAPSHOT_FILE}' is empty")

    # Prefer the real download time recorded when the snapshot was written, so the
    # data_download_timestamp column stays truthful for every job in the run.
    download_timestamp = pd.Timestamp.now(tz='UTC')
    try:
        with open(SNAPSHOT_META_FILE) as meta_file:
            download_timestamp = pd.Timestamp(json.load(meta_file)['data_download_timestamp'])
    except Exception:
        log_progress("⚠️  No snapshot metadata found; using current time as download timestamp.")

    log_progress(f"📊 Rows: {len(df_raw):,}, Columns: {len(df_raw.columns)}")
    log_memory_usage()
    return df_raw, download_timestamp


def write_snapshot(df_raw, download_timestamp):
    """Persist a fetched dataframe so other jobs in this run can reuse it."""
    df_raw.to_parquet(SNAPSHOT_FILE, index=False)
    with open(SNAPSHOT_META_FILE, 'w') as meta_file:
        json.dump({'data_download_timestamp': download_timestamp.isoformat(),
                   'total_papers': len(df_raw)}, meta_file)
    log_progress(f"💾 Wrote papers snapshot to {SNAPSHOT_FILE} ({len(df_raw):,} rows)")


def fetch_raw_data(use_snapshot=True):
    """
    Fetch raw papers data from Hugging Face, selecting only necessary columns.

    If a run-local snapshot (see SNAPSHOT_FILE) is present it is reused instead of
    hitting HuggingFace. Respects 'TEST_DATA_LIMIT' for testing, and retries with
    exponential backoff on HTTP 429 (rate limit) and other transient errors.
    """
    if use_snapshot and os.path.exists(SNAPSHOT_FILE):
        return _load_snapshot()

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
                    if _is_retryable(e) and attempt < _MAX_FETCH_RETRIES - 1:
                        delay = min(_FETCH_BASE_DELAY * (2 ** attempt), _MAX_FETCH_DELAY)
                        delay += random.uniform(0, 15)
                        log_progress(f"⏳ Transient fetch error ({type(e).__name__}). Retrying in "
                                     f"{delay:.1f}s (attempt {attempt + 1}/{_MAX_FETCH_RETRIES}): {e}")
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
