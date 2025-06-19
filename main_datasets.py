# --- main_datasets.py ---
"""Main orchestrator for the Datasets data processing pipeline."""

import os
import sys
import time
import pandas as pd
from utils import log_progress, log_memory_usage
from config_datasets import PROCESSED_PARQUET_FILE_PATH, FINAL_EXPECTED_COLUMNS
from data_fetcher_datasets import fetch_raw_data, validate_raw_data
from data_processor_datasets import setup_initial_dataframe, enrich_data
from tag_processor_datasets import process_tags_for_series, create_feature_flags, analyze_tag_distribution

def save_processed_data(df):
    """Save processed DataFrame to parquet file."""
    log_progress(f"💾 Saving processed dataset data to {PROCESSED_PARQUET_FILE_PATH}...")
    try:
        df.to_parquet(PROCESSED_PARQUET_FILE_PATH, index=False, engine='pyarrow')
        log_progress("✅ Data saved successfully.")
        return True
    except Exception as e:
        log_progress(f"❌ ERROR: Could not save processed data: {e}")
        return False

def main_pipeline():
    """Execute the complete dataset data processing pipeline."""
    log_progress("🚀 Starting HuggingFace DATASETS Data Processing Pipeline")
    log_progress("=" * 70)
    
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        os.remove(PROCESSED_PARQUET_FILE_PATH)

    # Step 1: Data Fetching
    log_progress("\nSTEP 1: Data Fetching")
    df_raw, data_download_timestamp = fetch_raw_data()
    validate_raw_data(df_raw)
    
    # Step 2: Initial Data Processing
    log_progress("\nSTEP 2: Initial Data Processing & Enrichment")
    df = setup_initial_dataframe(df_raw, data_download_timestamp)
    df = enrich_data(df)
    
    # Step 3: Tag Processing
    log_progress("\nSTEP 3: Tag Processing")
    df['tags'] = process_tags_for_series(df['tags'])
    analyze_tag_distribution(df)
    df = create_feature_flags(df)
    
    # Step 4: Finalize DataFrame
    log_progress("\nSTEP 4: Finalizing DataFrame")
    for col in FINAL_EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = False if 'has_' in col or 'is_' in col else None
    df_final = df[[col for col in FINAL_EXPECTED_COLUMNS if col in df.columns]]
    log_progress(f"Final DataFrame shape: {df_final.shape}")
    
    # Step 5: Save Results
    log_progress("\nSTEP 5: Save Results")
    if not save_processed_data(df_final):
        return False

    log_progress("\n🎉 DATASET PIPELINE COMPLETED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    try:
        if main_pipeline():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        log_progress(f"\n💥 UNEXPECTED FATAL ERROR in Datasets pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
