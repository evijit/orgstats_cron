# --- main.py ---
"""
Main orchestrator script for the data processing pipeline.
This version combines a robust, step-by-step execution with the correct
modular function calls.
"""

import os
import sys
import time
import pandas as pd
from utils import log_progress, log_memory_usage
from config import PROCESSED_PARQUET_FILE_PATH, MODEL_ID_TO_DEBUG, FINAL_EXPECTED_COLUMNS
from data_fetcher import fetch_raw_data, validate_raw_data
# <-- CHANGE: Import the correct, existing functions from data_processor
from data_processor import setup_initial_dataframe, calculate_file_sizes, enrich_data
from tag_processor import process_tags_for_series, create_feature_flags, debug_specific_model, analyze_tag_distribution

def save_processed_data(df):
    """Save processed DataFrame to parquet file."""
    log_progress("💾 Saving processed data...")
    save_start = time.time()
    
    try:
        log_progress(f"   Output file: {PROCESSED_PARQUET_FILE_PATH}")
        log_progress(f"   DataFrame shape: {df.shape}")
        log_progress(f"   DataFrame memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        
        df.to_parquet(PROCESSED_PARQUET_FILE_PATH, index=False, engine='pyarrow')
        
        file_size_mb = os.path.getsize(PROCESSED_PARQUET_FILE_PATH) / (1024 * 1024)
        save_time = time.time() - save_start
        log_progress(f"✅ Data saved successfully in {save_time:.2f}s (File size: {file_size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        log_progress(f"❌ ERROR: Could not save processed data: {e}")
        return False

def verify_saved_data():
    """Verify the saved parquet file by reading it back."""
    log_progress("🔍 Verifying saved data...")
    
    try:
        df_verify = pd.read_parquet(PROCESSED_PARQUET_FILE_PATH)
        log_progress("✅ File verification successful:")
        log_progress(f"   Rows: {len(df_verify):,}, Columns: {len(df_verify.columns)}")
        
        # Check key columns
        if 'has_robot' in df_verify.columns:
            log_progress(f"   Robotics models found: {df_verify['has_robot'].sum():,}")
        if 'organization' in df_verify.columns:
            log_progress(f"   Unique organizations: {df_verify['organization'].nunique():,}")
            
        return True
        
    except Exception as e:
        log_progress(f"❌ File verification failed: {e}")
        return False

def cleanup_existing_files():
    """Clean up existing output files before a run."""
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        log_progress(f"🗑️  Removing existing file: {PROCESSED_PARQUET_FILE_PATH}")
        try:
            os.remove(PROCESSED_PARQUET_FILE_PATH)
            log_progress("   File removed successfully.")
        except OSError as e:
            log_progress(f"❌ Error removing file: {e}. Please delete manually and rerun.")
            return False
    return True

def main_pipeline():
    """Execute the complete data processing pipeline."""
    log_progress("🚀 Starting HuggingFace Models Data Processing Pipeline")
    log_progress("=" * 70)
    
    pipeline_start = time.time()
    
    # Step 0: Cleanup
    if not cleanup_existing_files():
        return False
    
    # Step 1: Data Fetching
    log_progress("\nSTEP 1: Data Fetching")
    try:
        df_raw, data_download_timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
    except Exception as e:
        log_progress(f"❌ Data fetching failed: {e}")
        return False
    log_memory_usage()
    
    # <-- CHANGE: Replaced the single call to a non-existent function with the three correct ones. -->
    # Step 2: Initial Data Processing
    log_progress("\nSTEP 2: Initial Data Processing & Enrichment")
    try:
        df = setup_initial_dataframe(df_raw, data_download_timestamp)
        df = calculate_file_sizes(df, df_raw)
        df = enrich_data(df) # Adds 'organization' and 'size_category'
    except Exception as e:
        log_progress(f"❌ Initial data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    log_memory_usage()
    
    # Step 3: Tag Processing
    log_progress("\nSTEP 3: Tag Processing")
    try:
        df['tags'] = process_tags_for_series(df['tags'])
        analyze_tag_distribution(df)
        df = create_feature_flags(df)
        if MODEL_ID_TO_DEBUG:
            debug_specific_model(df, MODEL_ID_TO_DEBUG)
    except Exception as e:
        log_progress(f"❌ Tag processing failed: {e}")
        return False
    log_memory_usage()
    
    # <-- CHANGE: Replaced the call to non-existent `finalize_dataframe` with the correct logic. -->
    # Step 4: Finalize DataFrame
    log_progress("\nSTEP 4: Finalizing DataFrame")
    try:
        # Ensure all expected columns exist, adding any that might be missing
        for col in FINAL_EXPECTED_COLUMNS:
            if col not in df.columns:
                log_progress(f"⚠️  Final column '{col}' not found. Adding with default values.")
                df[col] = False if 'has_' in col or 'is_' in col else None
        
        # Select and reorder columns for the final output
        final_columns_in_df = [col for col in FINAL_EXPECTED_COLUMNS if col in df.columns]
        df_final = df[final_columns_in_df]
        log_progress(f"Final DataFrame shape: {df_final.shape}")
    except Exception as e:
        log_progress(f"❌ Final processing failed: {e}")
        return False
    
    # Step 5: Save Results
    log_progress("\nSTEP 5: Save Results")
    if not save_processed_data(df_final):
        return False
    
    # Step 6: Verification
    log_progress("\nSTEP 6: Verification")
    if not verify_saved_data():
        return False
    
    # Final Summary
    total_time = time.time() - pipeline_start
    log_progress("\n" + "=" * 70)
    log_progress("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    log_progress(f"   Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    log_progress(f"   Processed models: {len(df_final):,}")
    log_progress(f"   Output file: {PROCESSED_PARQUET_FILE_PATH}")
    
    return True

if __name__ == "__main__":
    try:
        if main_pipeline():
            log_progress("✅ Script completed successfully")
            sys.exit(0)
        else:
            log_progress("❌ Script failed")
            sys.exit(1)
    except KeyboardInterrupt:
        log_progress("\n⚠️  Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_progress(f"\n💥 UNEXPECTED FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)