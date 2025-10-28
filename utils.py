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
    log_message = f"[{timestamp}] {message}"

    if start_time:
        elapsed = time.time() - start_time
        log_message += f" (Elapsed: {elapsed:.2f}s)"
        
    # --- THIS IS THE CRITICAL FIX ---
    # Add flush=True to ensure the message is printed immediately in the CI/CD log.
    print(log_message, flush=True)

def extract_model_file_size_gb(safetensors_data):
    """Extract model file size in GB from safetensors data.
    Returns -1.0 if the size is unknown or the data is null/unparseable."""
    try:
        if pd.isna(safetensors_data):
            return -1.0 # Set to -1 for unknown size if the field is null

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
                return -1.0 # Return -1.0 if string parsing fails
        
        if isinstance(data_to_parse, dict) and 'total' in data_to_parse:
            total_bytes_val = data_to_parse['total']
            try:
                size_bytes = float(total_bytes_val)
                return size_bytes / (1024 * 1024 * 1024) 
            except (ValueError, TypeError): 
                return -1.0 # Return -1.0 if 'total' value is not convertible to float
        return -1.0 # Return -1.0 if data is not a dict or 'total' key is missing
    except Exception: 
        return -1.0 # Return -1.0 for any unexpected errors during processing

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
        # Treat -1 (unknown) or any negative value as 0 for categorization purposes,
        # unless specifically handled as a separate category.
        # The prompt implies -1 means "unknown size", which shouldn't fall into a positive range.
        if pd.isna(numeric_file_size_gb) or numeric_file_size_gb < 0:
            return "Unknown" # Or a specific category like "N/A" if defined
    except (ValueError, TypeError):
        return "Unknown" # Cannot categorize if not a number

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
