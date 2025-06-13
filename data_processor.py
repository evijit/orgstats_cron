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
            elif col_name == 'tags' or col_name == 'safetensors':
                # Preserve object type for complex data
                df[col_name] = df[col_name]
                log_progress(f"     Preserved as object type")
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
                # Use a Series of Nones to ensure correct object dtype
                df[col_name] = pd.Series([None] * len(df_raw), dtype='object')
            elif col_name == 'id':
                log_progress("❌ CRITICAL ERROR: 'id' column missing from source data")
                raise ValueError("'id' column is required but missing from source data")
    
    # Add timestamp
    df['data_download_timestamp'] = data_download_timestamp
    log_progress(f"   Added download timestamp: {data_download_timestamp}")
    
    setup_time = time.time() - setup_start
    log_progress(f"✅ DataFrame setup completed in {setup_time:.2f}s")
    
    # Validate structure
    validate_dataframe_structure(df, list(EXPECTED_COLUMNS_SETUP.keys()))
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
        # --- THIS BLOCK IS NOW COMPLETE ---
        df[output_filesize_col_name] = df['safetensors'].apply(extract_model_file_size_gb)
        calculated_count = (df[output_filesize_col_name] > 0).sum()
        log_progress(f"   Calculated size for {calculated_count:,} models from 'safetensors'.")

    else:
        log_progress(f"   ⚠️  WARNING: Cannot calculate file size. No '{output_filesize_col_name}' or 'safetensors' data available. Defaulting to 0.")
        df[output_filesize_col_name] = 0.0

    size_time = time.time() - size_start
    log_progress(f"✅ File size calculation completed in {size_time:.2f}s")
    log_memory_usage()

    return df

def enrich_data(df):
    """Add organization and size category columns."""
    log_progress("✨ Enriching data with organization and size categories...")
    enrich_start = time.time()

    # Extract organization
    log_progress("   Extracting organization from model ID...")
    df['organization'] = df['id'].apply(extract_org_from_id)
    org_count = df['organization'].nunique()
    log_progress(f"   Found {org_count:,} unique organizations.")

    # Categorize file size
    log_progress("   Categorizing models by file size...")
    df['size_category'] = df['params'].apply(get_file_size_category)
    log_progress("   Size categories assigned.")
    
    enrich_time = time.time() - enrich_start
    log_progress(f"✅ Data enrichment completed in {enrich_time:.2f}s")
    log_memory_usage()

    return df

if __name__ == "__main__":
    # --- ADDED A MEANINGFUL TEST BLOCK ---
    log_progress("🧪 Testing data_processor module...")
    
    # Create sample raw data
    raw_data = {
        'id': ['org1/modelA', 'org2/modelB', 'unaffiliated_modelC'],
        'downloads': [100, 200, 300],
        'likes': [10, 20, 30],
        'safetensors': [
            '{"total": 500000000}',  # 0.5 GB
            '{"total": 2000000000}', # 2.0 GB
            '{"total": 60000000000}' # 60.0 GB
        ],
        'tags': [['tag1'], ['tag2', 'tag3'], ['tag4']]
    }
    df_raw_test = pd.DataFrame(raw_data)
    timestamp_test = pd.Timestamp.now(tz='UTC')

    try:
        # Step 1: Setup DataFrame
        df_test = setup_initial_dataframe(df_raw_test, timestamp_test)
        
        # Step 2: Calculate file sizes
        df_test = calculate_file_sizes(df_test, df_raw_test)

        # Step 3: Enrich data
        df_test = enrich_data(df_test)

        log_progress("✅ Data processor test successful")
        print("\n--- Final Test DataFrame ---")
        print(df_test[['id', 'organization', 'params', 'size_category']].to_string())
        print("--------------------------\n")

        # Validation checks
        assert 'organization' in df_test.columns
        assert 'size_category' in df_test.columns
        assert df_test.loc[0, 'organization'] == 'org1'
        assert df_test.loc[2, 'organization'] == 'unaffiliated'
        assert df_test.loc[0, 'size_category'] == 'Small (<1GB)'
        assert df_test.loc[1, 'size_category'] == 'Medium (1-5GB)'
        assert df_test.loc[2, 'size_category'] == 'XX-Large (>50GB)'
        log_progress("✅ All assertions passed.")

    except Exception as e:
        log_progress(f"❌ Data processor test failed: {e}")
        raise