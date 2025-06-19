# --- test_pipeline_datasets.py ---
"""Integration test for the dataset processing pipeline."""

import os
import sys
from utils import log_progress
from data_fetcher_datasets import fetch_raw_data, validate_raw_data
from data_processor_datasets import setup_initial_dataframe, enrich_data
from tag_processor_datasets import process_tags_for_series, create_feature_flags

def run_test_pipeline():
    log_progress("🚀 Starting DATASET pipeline integration test...")
    try:
        # 1. Fetch
        df_raw, timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
        assert not df_raw.empty

        # 2. Process
        df = setup_initial_dataframe(df_raw, timestamp)
        df = enrich_data(df)
        assert 'organization' in df.columns
        assert 'size_category' not in df.columns # Verify model-specific column is absent

        # 3. Tags
        df['tags'] = process_tags_for_series(df['tags'])
        df = create_feature_flags(df)
        assert 'has_text' in df.columns

    except Exception as e:
        log_progress(f"❌ DATASET pipeline integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_progress("\n🎉✅ DATASET pipeline integration test PASSED! 🎉")

if __name__ == "__main__":
    if 'TEST_DATA_LIMIT' not in os.environ:
        print("❌ ERROR: Set 'TEST_DATA_LIMIT' for testing.")
        sys.exit(1)
    run_test_pipeline()
