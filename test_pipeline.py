# test_pipeline.py
"""
A dedicated script to run a fast, small-scale integration test of the entire
data processing pipeline. This is intended to be run in CI to quickly
validate code changes before the main, full-scale data processing.
"""

import os
import sys
import pandas as pd

from utils import log_progress
from data_fetcher import fetch_raw_data, validate_raw_data
from data_processor import setup_initial_dataframe, calculate_file_sizes, enrich_data
from tag_processor import process_tags_for_series, create_feature_flags

def run_test_pipeline():
    """Executes the full pipeline on a small subset of data."""
    log_progress("🚀 Starting pipeline integration test...")

    try:
        # 1. Fetch a small subset of data
        # The TEST_DATA_LIMIT env var is set in the workflow file.
        log_progress("--- Step 1: Fetching test data ---")
        df_raw, timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
        assert not df_raw.empty, "Raw data fetch returned an empty DataFrame."
        log_progress("✅ Raw data fetched and validated.")

        # 2. Setup and Process Data
        log_progress("\n--- Step 2: Processing data ---")
        df = setup_initial_dataframe(df_raw, timestamp)
        df = calculate_file_sizes(df, df_raw)
        df = enrich_data(df)
        assert 'organization' in df.columns, "Organization column not created."
        assert 'size_category' in df.columns, "Size category column not created."
        log_progress("✅ Data processing and enrichment complete.")

        # 3. Process Tags
        log_progress("\n--- Step 3: Processing tags ---")
        df['tags'] = process_tags_for_series(df['tags'])
        df = create_feature_flags(df)
        assert 'has_robot' in df.columns, "Feature flag 'has_robot' not created."
        log_progress("✅ Tag processing and feature flag creation complete.")

        # Final validation
        log_progress("\n--- Step 4: Final validation ---")
        final_rows, final_cols = df.shape
        log_progress(f"Final test DataFrame shape: ({final_rows}, {final_cols})")
        assert final_rows > 0, "Final DataFrame is empty."
        assert final_cols > 10, "Final DataFrame has too few columns."

    except Exception as e:
        log_progress(f"❌ Pipeline integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_progress("\n🎉✅ Pipeline integration test PASSED! 🎉")


if __name__ == "__main__":
    # Ensure the test limit is set, otherwise fail fast.
    if 'TEST_DATA_LIMIT' not in os.environ:
        print("❌ ERROR: This script should be run with the 'TEST_DATA_LIMIT' environment variable set.")
        print("         This prevents accidentally running it on the full dataset.")
        sys.exit(1)
        
    run_test_pipeline()