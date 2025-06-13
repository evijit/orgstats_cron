# --- START OF FILE preprocess.py ---

import pandas as pd
import numpy as np
import json
import ast
from tqdm.auto import tqdm
import time
import os
import duckdb
import re # Import re for the manual regex check in debug

# --- Constants ---
PROCESSED_PARQUET_FILE_PATH = "models_processed.parquet"
HF_PARQUET_URL = 'https://huggingface.co/datasets/cfahlgren1/hub-stats/resolve/main/models.parquet'

MODEL_SIZE_RANGES = {
    "Small (<1GB)": (0, 1),
    "Medium (1-5GB)": (1, 5),
    "Large (5-20GB)": (5, 20),
    "X-Large (20-50GB)": (20, 50),
    "XX-Large (>50GB)": (50, float('inf'))
}

# --- Debugging Constant ---
# <<<<<<< SET THE MODEL ID YOU WANT TO DEBUG HERE >>>>>>>
MODEL_ID_TO_DEBUG = "openvla/openvla-7b" 
# Example: MODEL_ID_TO_DEBUG = "openai-community/gpt2" 
# If you don't have a specific ID, the debug block will just report it's not found.

# --- Utility Functions (extract_model_file_size_gb, extract_org_from_id, process_tags_for_series, get_file_size_category - unchanged from previous correct version) ---
def extract_model_file_size_gb(safetensors_data):
    try:
        if pd.isna(safetensors_data): return 0.0
        data_to_parse = safetensors_data
        if isinstance(safetensors_data, str):
            try:
                if (safetensors_data.startswith('{') and safetensors_data.endswith('}')) or \
                   (safetensors_data.startswith('[') and safetensors_data.endswith(']')):
                    data_to_parse = ast.literal_eval(safetensors_data)
                else: data_to_parse = json.loads(safetensors_data)
            except Exception: return 0.0
        if isinstance(data_to_parse, dict) and 'total' in data_to_parse:
            total_bytes_val = data_to_parse['total']
            try:
                size_bytes = float(total_bytes_val)
                return size_bytes / (1024 * 1024 * 1024) 
            except (ValueError, TypeError): return 0.0
        return 0.0
    except Exception: return 0.0

def extract_org_from_id(model_id):
    if pd.isna(model_id): return "unaffiliated"
    model_id_str = str(model_id)
    return model_id_str.split("/")[0] if "/" in model_id_str else "unaffiliated"

def process_tags_for_series(series_of_tags_values):
    processed_tags_accumulator = []

    for i, tags_value_from_series in enumerate(tqdm(series_of_tags_values, desc="Standardizing Tags", leave=False, unit="row")):
        temp_processed_list_for_row = []
        current_value_for_error_msg = str(tags_value_from_series)[:200] # Truncate for long error messages

        try:
            # Order of checks is important!
            # 1. Handle explicit Python lists first
            if isinstance(tags_value_from_series, list):
                current_tags_in_list = []
                for idx_tag, tag_item in enumerate(tags_value_from_series):
                    try:
                        # Ensure item is not NaN before string conversion if it might be a float NaN in a list
                        if pd.isna(tag_item): continue 
                        str_tag = str(tag_item)
                        stripped_tag = str_tag.strip()
                        if stripped_tag:
                            current_tags_in_list.append(stripped_tag)
                    except Exception as e_inner_list_proc:
                        print(f"ERROR processing item '{tag_item}' (type: {type(tag_item)}) within a list for row {i}. Error: {e_inner_list_proc}. Original list: {current_value_for_error_msg}")
                temp_processed_list_for_row = current_tags_in_list

            # 2. Handle NumPy arrays
            elif isinstance(tags_value_from_series, np.ndarray):
                # Convert to list, then process elements, handling potential NaNs within the array
                current_tags_in_list = []
                for idx_tag, tag_item in enumerate(tags_value_from_series.tolist()): # .tolist() is crucial
                    try:
                        if pd.isna(tag_item): continue # Check for NaN after converting to Python type
                        str_tag = str(tag_item)
                        stripped_tag = str_tag.strip()
                        if stripped_tag:
                            current_tags_in_list.append(stripped_tag)
                    except Exception as e_inner_array_proc:
                        print(f"ERROR processing item '{tag_item}' (type: {type(tag_item)}) within a NumPy array for row {i}. Error: {e_inner_array_proc}. Original array: {current_value_for_error_msg}")
                temp_processed_list_for_row = current_tags_in_list
            
            # 3. Handle simple None or pd.NA after lists and arrays (which might contain pd.NA elements handled above)
            elif tags_value_from_series is None or pd.isna(tags_value_from_series): # Now pd.isna is safe for scalars
                temp_processed_list_for_row = []

            # 4. Handle strings (could be JSON-like, list-like, or comma-separated)
            elif isinstance(tags_value_from_series, str):
                processed_str_tags = []
                # Attempt ast.literal_eval for strings that look like lists/tuples
                if (tags_value_from_series.startswith('[') and tags_value_from_series.endswith(']')) or \
                   (tags_value_from_series.startswith('(') and tags_value_from_series.endswith(')')):
                    try:
                        evaluated_tags = ast.literal_eval(tags_value_from_series)
                        if isinstance(evaluated_tags, (list, tuple)): # Check if eval result is a list/tuple
                            # Recursively process this evaluated list/tuple, as its elements could be complex
                            # For simplicity here, assume elements are simple strings after eval
                            current_eval_list = []
                            for tag_item in evaluated_tags:
                                if pd.isna(tag_item): continue
                                str_tag = str(tag_item).strip()
                                if str_tag: current_eval_list.append(str_tag)
                            processed_str_tags = current_eval_list
                    except (ValueError, SyntaxError):
                        pass # If ast.literal_eval fails, let it fall to JSON or comma split

                # If ast.literal_eval didn't populate, try JSON
                if not processed_str_tags:
                    try:
                        json_tags = json.loads(tags_value_from_series)
                        if isinstance(json_tags, list):
                            # Similar to above, assume elements are simple strings after JSON parsing
                            current_json_list = []
                            for tag_item in json_tags:
                                if pd.isna(tag_item): continue
                                str_tag = str(tag_item).strip()
                                if str_tag: current_json_list.append(str_tag)
                            processed_str_tags = current_json_list
                    except json.JSONDecodeError:
                        # If not a valid JSON list, fall back to comma splitting as the final string strategy
                        processed_str_tags = [tag.strip() for tag in tags_value_from_series.split(',') if tag.strip()]
                    except Exception as e_json_other:
                        print(f"ERROR during JSON processing for string '{current_value_for_error_msg}' for row {i}. Error: {e_json_other}")
                        processed_str_tags = [tag.strip() for tag in tags_value_from_series.split(',') if tag.strip()] # Fallback

                temp_processed_list_for_row = processed_str_tags
            
            # 5. Fallback for other scalar types (e.g., int, float that are not NaN)
            else:
                # This path is for non-list, non-ndarray, non-None/NaN, non-string types.
                # Or for NaNs that slipped through if they are not None or pd.NA (e.g. float('nan'))
                if pd.isna(tags_value_from_series): # Catch any remaining NaNs like float('nan')
                     temp_processed_list_for_row = []
                else:
                    str_val = str(tags_value_from_series).strip()
                    temp_processed_list_for_row = [str_val] if str_val else []
            
            processed_tags_accumulator.append(temp_processed_list_for_row)

        except Exception as e_outer_tag_proc:
            print(f"CRITICAL UNHANDLED ERROR processing row {i}: value '{current_value_for_error_msg}' (type: {type(tags_value_from_series)}). Error: {e_outer_tag_proc}. Appending [].")
            processed_tags_accumulator.append([])
            
    return processed_tags_accumulator

def get_file_size_category(file_size_gb_val):
    try:
        numeric_file_size_gb = float(file_size_gb_val)
        if pd.isna(numeric_file_size_gb): numeric_file_size_gb = 0.0
    except (ValueError, TypeError): numeric_file_size_gb = 0.0
    if 0 <= numeric_file_size_gb < 1: return "Small (<1GB)"
    elif 1 <= numeric_file_size_gb < 5: return "Medium (1-5GB)"
    elif 5 <= numeric_file_size_gb < 20: return "Large (5-20GB)"
    elif 20 <= numeric_file_size_gb < 50: return "X-Large (20-50GB)"
    elif numeric_file_size_gb >= 50: return "XX-Large (>50GB)"
    else: return "Small (<1GB)"


def main_preprocessor():
    print(f"Starting pre-processing script. Output: '{PROCESSED_PARQUET_FILE_PATH}'.")
    overall_start_time = time.time()

    print(f"Fetching fresh data from Hugging Face: {HF_PARQUET_URL}")
    try:
        fetch_start_time = time.time()
        query = f"SELECT * FROM read_parquet('{HF_PARQUET_URL}')"
        df_raw = duckdb.sql(query).df()
        data_download_timestamp = pd.Timestamp.now(tz='UTC')
        
        if df_raw is None or df_raw.empty: raise ValueError("Fetched data is empty or None.")
        if 'id' not in df_raw.columns: raise ValueError("Fetched data must contain 'id' column.")
        
        print(f"Fetched data in {time.time() - fetch_start_time:.2f}s. Rows: {len(df_raw)}. Downloaded at: {data_download_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception as e_fetch:
        print(f"ERROR: Could not fetch data from Hugging Face: {e_fetch}.")
        return

    df = pd.DataFrame()
    print("Processing raw data...")
    proc_start = time.time()

    expected_cols_setup = {
        'id': str, 'downloads': float, 'downloadsAllTime': float, 'likes': float,
        'pipeline_tag': str, 'tags': object, 'safetensors': object
    }
    for col_name, target_dtype in expected_cols_setup.items():
        if col_name in df_raw.columns:
            df[col_name] = df_raw[col_name]
            if target_dtype == float: df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0)
            elif target_dtype == str: df[col_name] = df[col_name].astype(str).fillna('')
        else: 
            if col_name in ['downloads', 'downloadsAllTime', 'likes']: df[col_name] = 0.0
            elif col_name == 'pipeline_tag': df[col_name] = ''
            elif col_name == 'tags': df[col_name] = pd.Series([[] for _ in range(len(df_raw))]) # Initialize with empty lists
            elif col_name == 'safetensors': df[col_name] = None # Initialize with None
            elif col_name == 'id': print("CRITICAL ERROR: 'id' column missing."); return
    
    output_filesize_col_name = 'params' 
    if output_filesize_col_name in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[output_filesize_col_name]):
        print(f"Using pre-existing '{output_filesize_col_name}' column as file size in GB.")
        df[output_filesize_col_name] = pd.to_numeric(df_raw[output_filesize_col_name], errors='coerce').fillna(0.0)
    elif 'safetensors' in df.columns:
        print(f"Calculating '{output_filesize_col_name}' (file size in GB) from 'safetensors' data...")
        df[output_filesize_col_name] = df['safetensors'].apply(extract_model_file_size_gb)
        df[output_filesize_col_name] = pd.to_numeric(df[output_filesize_col_name], errors='coerce').fillna(0.0)
    else:
        print(f"Cannot determine file size. Setting '{output_filesize_col_name}' to 0.0.")
        df[output_filesize_col_name] = 0.0
    
    df['data_download_timestamp'] = data_download_timestamp
    print(f"Added 'data_download_timestamp' column.")

    print("Categorizing models by file size...")
    df['size_category'] = df[output_filesize_col_name].apply(get_file_size_category)

    print("Standardizing 'tags' column...")
    df['tags'] = process_tags_for_series(df['tags']) # This now uses tqdm internally

    # --- START DEBUGGING BLOCK ---
    # This block will execute before the main tag processing loop
    if MODEL_ID_TO_DEBUG and MODEL_ID_TO_DEBUG in df['id'].values: # Check if ID exists
        print(f"\n--- Pre-Loop Debugging for Model ID: {MODEL_ID_TO_DEBUG} ---")
        
        # 1. Check the 'tags' column content after process_tags_for_series
        model_specific_tags_list = df.loc[df['id'] == MODEL_ID_TO_DEBUG, 'tags'].iloc[0]
        print(f"1. Tags from df['tags'] (after process_tags_for_series): {model_specific_tags_list}")
        print(f"   Type of tags: {type(model_specific_tags_list)}")
        if isinstance(model_specific_tags_list, list):
            for i, tag_item in enumerate(model_specific_tags_list):
                print(f"   Tag item {i}: '{tag_item}' (type: {type(tag_item)}, len: {len(str(tag_item))})")
                # Detailed check for 'robotics' specifically
                if 'robotics' in str(tag_item).lower():
                    print(f"     DEBUG: Found 'robotics' substring in '{tag_item}'")
                    print(f"       - str(tag_item).lower().strip(): '{str(tag_item).lower().strip()}'")
                    print(f"       - Is it exactly 'robotics'?: {str(tag_item).lower().strip() == 'robotics'}")
                    print(f"       - Ordinals: {[ord(c) for c in str(tag_item)]}")

        # 2. Simulate temp_tags_joined for this specific model
        if isinstance(model_specific_tags_list, list):
            simulated_temp_tags_joined = '~~~'.join(str(t).lower().strip() for t in model_specific_tags_list if pd.notna(t) and str(t).strip())
        else:
            simulated_temp_tags_joined = ''
        print(f"2. Simulated 'temp_tags_joined' for this model: '{simulated_temp_tags_joined}'")

        # 3. Simulate 'has_robot' check for this model
        robot_keywords = ['robot', 'robotics']
        robot_pattern = '|'.join(robot_keywords)
        manual_robot_check = bool(re.search(robot_pattern, simulated_temp_tags_joined, flags=re.IGNORECASE))
        print(f"3. Manual regex check for 'has_robot' ('{robot_pattern}' in '{simulated_temp_tags_joined}'): {manual_robot_check}")
        print(f"--- End Pre-Loop Debugging for Model ID: {MODEL_ID_TO_DEBUG} ---\n")
    elif MODEL_ID_TO_DEBUG:
        print(f"DEBUG: Model ID '{MODEL_ID_TO_DEBUG}' not found in DataFrame for pre-loop debugging.")
    # --- END DEBUGGING BLOCK ---


    print("Vectorized creation of cached tag columns...")
    tag_time = time.time()
    # This is the original temp_tags_joined creation:
    df['temp_tags_joined'] = df['tags'].apply(
        lambda tl: '~~~'.join(str(t).lower().strip() for t in tl if pd.notna(t) and str(t).strip()) if isinstance(tl, list) else ''
    )
    
    tag_map = {
        'has_audio': ['audio'], 'has_speech': ['speech'], 'has_music': ['music'],
        'has_robot': ['robot', 'robotics','openvla','vla'], 
        'has_bio': ['bio'], 'has_med': ['medic', 'medical'], 
        'has_series': ['series', 'time-series', 'timeseries'], 
        'has_video': ['video'], 'has_image': ['image', 'vision'], 
        'has_text': ['text', 'nlp', 'llm'] 
    }
    for col, kws in tag_map.items():
        pattern = '|'.join(kws) 
        df[col] = df['temp_tags_joined'].str.contains(pattern, na=False, case=False, regex=True)
        
    df['has_science'] = (
        df['temp_tags_joined'].str.contains('science', na=False, case=False, regex=True) &
        ~df['temp_tags_joined'].str.contains('bigscience', na=False, case=False, regex=True) 
    )
    del df['temp_tags_joined'] # Clean up temporary column
    df['is_audio_speech'] = (df['has_audio'] | df['has_speech'] |
                            df['pipeline_tag'].str.contains('audio|speech', case=False, na=False, regex=True))
    df['is_biomed'] = df['has_bio'] | df['has_med']
    print(f"Vectorized tag columns created in {time.time() - tag_time:.2f}s.")

    # --- POST-LOOP DIAGNOSTIC for has_robot & a specific model ---
    if 'has_robot' in df.columns:
        print("\n--- 'has_robot' Diagnostics (Preprocessor - Post-Loop) ---")
        print(df['has_robot'].value_counts(dropna=False))
        
        if MODEL_ID_TO_DEBUG and MODEL_ID_TO_DEBUG in df['id'].values:
            model_has_robot_val = df.loc[df['id'] == MODEL_ID_TO_DEBUG, 'has_robot'].iloc[0]
            print(f"Value of 'has_robot' for model '{MODEL_ID_TO_DEBUG}': {model_has_robot_val}")
            if model_has_robot_val:
                 print(f"  Original tags for '{MODEL_ID_TO_DEBUG}': {df.loc[df['id'] == MODEL_ID_TO_DEBUG, 'tags'].iloc[0]}")

        if df['has_robot'].any():
            print("Sample models flagged as 'has_robot':")
            print(df[df['has_robot']][['id', 'tags', 'has_robot']].head(5))
        else:
            print("No models were flagged as 'has_robot' after processing.")
        print("--------------------------------------------------------\n")
    # --- END POST-LOOP DIAGNOSTIC ---


    print("Adding organization column...")
    df['organization'] = df['id'].apply(extract_org_from_id)

    # Drop safetensors if params was calculated from it, and params didn't pre-exist as numeric
    if 'safetensors' in df.columns and \
       not (output_filesize_col_name in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[output_filesize_col_name])):
        df = df.drop(columns=['safetensors'], errors='ignore')
    
    final_expected_cols = [
        'id', 'downloads', 'downloadsAllTime', 'likes', 'pipeline_tag', 'tags',
        'params', 'size_category', 'organization',
        'has_audio', 'has_speech', 'has_music', 'has_robot', 'has_bio', 'has_med',
        'has_series', 'has_video', 'has_image', 'has_text', 'has_science',
        'is_audio_speech', 'is_biomed',
        'data_download_timestamp'
    ]
    # Ensure all final columns exist, adding defaults if necessary
    for col in final_expected_cols:
        if col not in df.columns:
            print(f"Warning: Final expected column '{col}' is missing! Defaulting appropriately.")
            if col == 'params': df[col] = 0.0
            elif col == 'size_category': df[col] = "Small (<1GB)" # Default size category
            elif 'has_' in col or 'is_' in col : df[col] = False # Default boolean flags to False
            elif col == 'data_download_timestamp': df[col] = pd.NaT # Default timestamp to NaT

    print(f"Data processing completed in {time.time() - proc_start:.2f}s.")
    try:
        print(f"Saving processed data to: {PROCESSED_PARQUET_FILE_PATH}")
        df_to_save = df[final_expected_cols].copy() # Ensure only expected columns are saved
        df_to_save.to_parquet(PROCESSED_PARQUET_FILE_PATH, index=False, engine='pyarrow')
        print(f"Successfully saved processed data.")
    except Exception as e_save:
        print(f"ERROR: Could not save processed data: {e_save}")
        return

    total_elapsed_script = time.time() - overall_start_time
    print(f"Pre-processing finished. Total time: {total_elapsed_script:.2f}s. Final Parquet shape: {df_to_save.shape}")

if __name__ == "__main__":
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        print(f"Deleting existing '{PROCESSED_PARQUET_FILE_PATH}' to ensure fresh processing...")
        try: os.remove(PROCESSED_PARQUET_FILE_PATH)
        except OSError as e: print(f"Error deleting file: {e}. Please delete manually and rerun."); exit()
    
    main_preprocessor()
    
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        print(f"\nTo verify, load parquet and check 'has_robot' and its 'tags':")
        print(f"import pandas as pd; df_chk = pd.read_parquet('{PROCESSED_PARQUET_FILE_PATH}')")
        print(f"print(df_chk['has_robot'].value_counts())")
        if MODEL_ID_TO_DEBUG:
            print(f"print(df_chk[df_chk['id'] == '{MODEL_ID_TO_DEBUG}'][['id', 'tags', 'has_robot']])")
        else:
            print(f"print(df_chk[df_chk['has_robot']][['id', 'tags', 'has_robot']].head())")