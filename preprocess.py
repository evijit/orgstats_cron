#!/usr/bin/env python3
"""
Lightweight version of preprocess.py optimized for GitHub Actions
Focuses on minimal memory usage and fast processing
"""

import pandas as pd
import numpy as np
import duckdb
import time
import gc
import os
import psutil
from datetime import datetime

# Constants
PROCESSED_PARQUET_FILE_PATH = "models_processed.parquet"
HF_PARQUET_URL = 'https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/models.parquet'

def get_memory_mb():
    """Get current memory usage in MB"""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except:
        return 0

def process_lightweight_chunk(chunk_df):
    """
    Lightweight processing focusing only on essential operations
    """
    try:
        # Start with essential columns only
        result = pd.DataFrame()
        
        # Copy essential columns with basic type conversion
        essential_mapping = {
            'id': (str, ''),
            'downloads': (float, 0.0),
            'downloadsAllTime': (float, 0.0), 
            'likes': (float, 0.0),
            'pipeline_tag': (str, ''),
            'tags': (object, [])
        }
        
        for col, (dtype, default) in essential_mapping.items():
            if col in chunk_df.columns:
                if dtype == float:
                    result[col] = pd.to_numeric(chunk_df[col], errors='coerce').fillna(default)
                elif dtype == str:
                    result[col] = chunk_df[col].astype(str).fillna(default)
                else:
                    result[col] = chunk_df[col].fillna(default)
            else:
                if col == 'tags':
                    result[col] = [[] for _ in range(len(chunk_df))]
                else:
                    result[col] = default
        
        # Handle file size - simplified version
        if 'params' in chunk_df.columns and pd.api.types.is_numeric_dtype(chunk_df['params']):
            result['params'] = pd.to_numeric(chunk_df['params'], errors='coerce').fillna(0.0)
        else:
            result['params'] = 0.0  # Default to 0 for simplicity
        
        # Add basic derived columns
        result['organization'] = result['id'].apply(lambda x: str(x).split('/')[0] if '/' in str(x) else 'unaffiliated')
        result['data_download_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Add size category based on params
        def get_size_category(size_gb):
            try:
                size = float(size_gb)
                if size < 1: return "Small (<1GB)"
                elif size < 5: return "Medium (1-5GB)"  
                elif size < 20: return "Large (5-20GB)"
                elif size < 50: return "X-Large (20-50GB)"
                else: return "XX-Large (>50GB)"
            except:
                return "Small (<1GB)"
        
        result['size_category'] = result['params'].apply(get_size_category)
        
        # Simplified boolean flags (no complex tag processing)
        result['has_robot'] = False  # Default to False for speed
        result['has_audio'] = False
        result['has_text'] = False
        result['has_image'] = False
        result['has_video'] = False
        
        return result
        
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

def main():
    """Main lightweight preprocessing function"""
    print(f"🚀 Starting lightweight preprocessing")
    print(f"💾 Initial memory: {get_memory_mb():.1f} MB")
    
    start_time = time.time()
    
    try:
        # Get total row count first
        count_query = f"SELECT COUNT(*) as total FROM read_parquet('{HF_PARQUET_URL}')"
        total_rows = duckdb.sql(count_query).df().iloc[0]['total']
        print(f"📊 Total rows to process: {total_rows:,}")
        
        # Use small chunks for memory efficiency  
        chunk_size = 20000
        processed_chunks = []
        
        print(f"🔄 Processing in chunks of {chunk_size:,}")
        
        for i in range(0, total_rows, chunk_size):
            chunk_start = time.time()
            
            # Load chunk
            limit = min(chunk_size, total_rows - i)
            query = f"SELECT * FROM read_parquet('{HF_PARQUET_URL}') LIMIT {limit} OFFSET {i}"
            chunk_raw = duckdb.sql(query).df()
            
            # Process chunk
            chunk_processed = process_lightweight_chunk(chunk_raw)
            if chunk_processed is None:
                raise Exception(f"Failed to process chunk {i//chunk_size + 1}")
            
            processed_chunks.append(chunk_processed)
            
            # Cleanup
            del chunk_raw, chunk_processed
            gc.collect()
            
            chunk_time = time.time() - chunk_start
            memory_now = get_memory_mb()
            
            print(f"✅ Chunk {i//chunk_size + 1}/{(total_rows-1)//chunk_size + 1} "
                  f"({chunk_time:.1f}s, {memory_now:.1f} MB)")
            
            # Memory safety check
            if memory_now > 1500:  # 1.5GB limit
                print("⚠️ Memory limit approaching, saving current progress")
                break
        
        # Combine and save
        print("🔄 Combining chunks...")
        df_final = pd.concat(processed_chunks, ignore_index=True)
        del processed_chunks
        gc.collect()
        
        print(f"💾 Saving {len(df_final):,} rows to {PROCESSED_PARQUET_FILE_PATH}")
        df_final.to_parquet(PROCESSED_PARQUET_FILE_PATH, index=False, compression='snappy')
        
        file_size = os.path.getsize(PROCESSED_PARQUET_FILE_PATH) / (1024 * 1024)
        total_time = time.time() - start_time
        
        print(f"✅ Success! File: {file_size:.1f} MB, Time: {total_time:.1f}s")
        print(f"💾 Final memory: {get_memory_mb():.1f} MB")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Clean up any existing file
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        os.remove(PROCESSED_PARQUET_FILE_PATH)
    
    success = main()
    exit(0 if success else 1)