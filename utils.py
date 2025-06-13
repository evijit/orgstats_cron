# --- utils.py ---
"""Utility functions for data processing."""

import pandas as pd
import numpy as np
import json
import ast
import time
from tqdm.auto import tqdm
from config import MODEL_SIZE_RANGES

def log_progress(message, start_time=None):
    """Enhanced logging with timestamps and elapsed time."""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    if start_time:
        elapsed = time.time() - start_time
        print(f"[{timestamp}] {message} (Elapsed: {elapsed:.2f}s)")
    else:
        print(f"[{timestamp}] {message}")
    
def extract_model_file_size_gb(safetensors_data):
    """Extract model file size in GB from safetensors data."""
    try:
        if pd.isna(safetensors_data): 
            return 0.0
        
        data_to_parse = safetensors_data
        if isinstance(safetensors_data, str):
            try:
                # Prioritize literal_eval for safety with list/dict-like strings
                if (safetensors_data.startswith('{') and safetensors_data.endswith('}')) or \
                   (safetensors_data.startswith('[') and safetensors_data.endswith(']')):
                    data_to_parse = ast.literal_eval(safetensors_data)
                else: 
                    # Fallback to json.loads for other valid JSON strings
                    data_to_parse = json.loads(safetensors_data)
            except (ValueError, SyntaxError, json.JSONDecodeError): 
                return 0.0
        
        if isinstance(data_to_parse, dict) and 'total' in data_to_parse:
            total_bytes_val = data_to_parse['total']
            try:
                size_bytes = float(total_bytes_val)
                return size_bytes / (1024 * 1024 * 1024) 
            except (ValueError, TypeError): 
                return 0.0
        return 0.0
    except Exception: 
        return 0.0

def extract_org_from_id(model_id):
    """Extract organization name from model ID."""
    if pd.isna(model_id): 
        return "unaffiliated"
    model_id_str = str(model_id)
    return model_id_str.split("/")[0] if "/" in model_id_str else "unaffiliated"

def get_file_size_category(file_size_gb_val):
    """
    Categorize file size into predefined ranges using the config.
    --- THIS FUNCTION IS NOW REFACTORED AND DATA-DRIVEN ---
    """
    try:
        numeric_file_size_gb = float(file_size_gb_val)
        if pd.isna(numeric_file_size_gb) or numeric_file_size_gb < 0:
            numeric_file_size_gb = 0.0
    except (ValueError, TypeError):
        numeric_file_size_gb = 0.0
    
    for category, (min_gb, max_gb) in MODEL_SIZE_RANGES.items():
        if min_gb <= numeric_file_size_gb < max_gb:
            return category
    
    return "Unknown" # Fallback, though (>50GB) should catch large values

def log_memory_usage():
    """Log current memory usage if psutil is available."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        rss_mb = memory_info.rss / (1024 ** 2)
        vms_mb = memory_info.vms / (1024 ** 2)
        
        virtual_memory = psutil.virtual_memory()
        total_gb = virtual_memory.total / (1024 ** 3)
        available_gb = virtual_memory.available / (1024 ** 3)
        
        print(f"🧠 Memory: Process RSS={rss_mb:.1f}MB, System Avail={available_gb:.1f}GB / {total_gb:.1f}GB")
    except ImportError:
        print("🧠 Memory monitoring not available (psutil not installed)")
    except Exception as e:
        print(f"🧠 Memory monitoring error: {e}")

def validate_dataframe_structure(df, expected_columns=None):
    """Validate DataFrame structure and log basic statistics."""
    log_progress(f"Validating DataFrame: Shape = {df.shape}")
    
    missing_cols = []
    if expected_columns:
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            log_progress(f"⚠️  WARNING: Missing expected columns: {missing_cols}")
        else:
            log_progress("✅ All expected columns present.")
    
    # Log memory usage of DataFrame
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    log_progress(f"DataFrame memory usage: {memory_usage_mb:.2f} MB")
    
    return len(missing_cols) == 0 if expected_columns else True