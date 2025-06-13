# --- tag_processor.py ---
"""Module for processing tags and creating feature flags."""

import pandas as pd
import numpy as np
import json
import ast
import re
import time
from tqdm.auto import tqdm
from utils import log_progress, log_memory_usage
from config import TAG_MAP, MODEL_ID_TO_DEBUG

def process_tags_for_series(series_of_tags_values):
    """Process and standardize tags column."""
    log_progress("🏷️  Starting tag processing...")
    process_start = time.time()
    
    processed_tags_accumulator = []
    total_rows = len(series_of_tags_values)
    
    # Create progress bar with more frequent updates
    pbar = tqdm(series_of_tags_values, desc="Processing tags", unit="rows", 
                mininterval=1.0, maxinterval=10.0)
    
    error_count = 0
    processed_count = 0
    
    for i, tags_value_from_series in enumerate(pbar):
        temp_processed_list_for_row = []
        current_value_for_error_msg = str(tags_value_from_series)[:200]
        
        try:
            # Update progress every 10,000 rows
            if i % 10000 == 0 and i > 0:
                elapsed = time.time() - process_start
                rate = i / elapsed
                eta = (total_rows - i) / rate if rate > 0 else 0
                pbar.set_postfix({
                    'errors': error_count,
                    'rate': f'{rate:.0f}/s',
                    'eta': f'{eta:.0f}s'
                })
            
            # Order of checks is important!
            # 1. Handle explicit Python lists first
            if isinstance(tags_value_from_series, list):
                current_tags_in_list = []
                for idx_tag, tag_item in enumerate(tags_value_from_series):
                    try:
                        if pd.isna(tag_item): 
                            continue 
                        str_tag = str(tag_item)
                        stripped_tag = str_tag.strip()
                        if stripped_tag:
                            current_tags_in_list.append(stripped_tag)
                    except Exception as e_inner_list_proc:
                        error_count += 1
                        if error_count <= 10:  # Only log first 10 errors
                            log_progress(f"ERROR processing list item '{tag_item}' at row {i}: {e_inner_list_proc}")
                temp_processed_list_for_row = current_tags_in_list

            # 2. Handle NumPy arrays
            elif isinstance(tags_value_from_series, np.ndarray):
                current_tags_in_list = []
                for idx_tag, tag_item in enumerate(tags_value_from_series.tolist()):
                    try:
                        if pd.isna(tag_item): 
                            continue
                        str_tag = str(tag_item)
                        stripped_tag = str_tag.strip()
                        if stripped_tag:
                            current_tags_in_list.append(stripped_tag)
                    except Exception as e_inner_array_proc:
                        error_count += 1
                        if error_count <= 10:
                            log_progress(f"ERROR processing array item '{tag_item}' at row {i}: {e_inner_array_proc}")
                temp_processed_list_for_row = current_tags_in_list
            
            # 3. Handle simple None or pd.NA
            elif tags_value_from_series is None or pd.isna(tags_value_from_series):
                temp_processed_list_for_row = []

            # 4. Handle strings (could be JSON-like, list-like, or comma-separated)
            elif isinstance(tags_value_from_series, str):
                processed_str_tags = []
                
                # Attempt ast.literal_eval for strings that look like lists/tuples
                if (tags_value_from_series.startswith('[') and tags_value_from_series.endswith(']')) or \
                   (tags_value_from_series.startswith('(') and tags_value_from_series.endswith(')')):
                    try:
                        evaluated_tags = ast.literal_eval(tags_value_from_series)
                        if isinstance(evaluated_tags, (list, tuple)):
                            current_eval_list = []
                            for tag_item in evaluated_tags:
                                if pd.isna(tag_item): 
                                    continue
                                str_tag = str(tag_item).strip()
                                if str_tag: 
                                    current_eval_list.append(str_tag)
                            processed_str_tags = current_eval_list
                    except (ValueError, SyntaxError):
                        pass  # Fall through to JSON or comma split

                # If ast.literal_eval didn't populate, try JSON
                if not processed_str_tags:
                    try:
                        json_tags = json.loads(tags_value_from_series)
                        if isinstance(json_tags, list):
                            current_json_list = []
                            for tag_item in json_tags:
                                if pd.isna(tag_item): 
                                    continue
                                str_tag = str(tag_item).strip()
                                if str_tag: 
                                    current_json_list.append(str_tag)
                            processed_str_tags = current_json_list
                    except json.JSONDecodeError:
                        # Fall back to comma splitting
                        processed_str_tags = [tag.strip() for tag in tags_value_from_series.split(',') if tag.strip()]
                    except Exception as e_json_other:
                        error_count += 1
                        if error_count <= 10:
                            log_progress(f"ERROR during JSON processing at row {i}: {e_json_other}")
                        processed_str_tags = [tag.strip() for tag in tags_value_from_series.split(',') if tag.strip()]

                temp_processed_list_for_row = processed_str_tags
            
            # 5. Fallback for other scalar types
            else:
                if pd.isna(tags_value_from_series):
                     temp_processed_list_for_row = []
                else:
                    str_val = str(tags_value_from_series).strip()
                    temp_processed_list_for_row = [str_val] if str_val else []
            
            processed_tags_accumulator.append(temp_processed_list_for_row)
            processed_count += 1

        except Exception as e_outer_tag_proc:
            error_count += 1
            if error_count <= 10:
                log_progress(f"CRITICAL ERROR at row {i}: {e_outer_tag_proc}")
            processed_tags_accumulator.append([])
    
    pbar.close()
    
    process_time = time.time() - process_start
    log_progress(f"✅ Tag processing completed in {process_time:.2f}s")
    log_progress(f"   - Processed: {processed_count:,} rows")
    log_progress(f"   - Errors: {error_count:,}")
    log_progress(f"   - Processing rate: {processed_count/process_time:.0f} rows/second")
    
    return processed_tags_accumulator

def create_feature_flags(df):
    """Create boolean feature flags based on tags."""
    log_progress("🚩 Creating feature flags from tags...")
    flag_start = time.time()
    
    # Create temporary joined tags column for efficient regex matching
    log_progress("   Creating temporary tag search string...")
    df['temp_tags_joined'] = df['tags'].apply(
        lambda tl: '~~~'.join(str(t).lower().strip() for t in tl if pd.notna(t) and str(t).strip()) 
        if isinstance(tl, list) else ''
    )
    
    log_progress(f"   Processing {len(TAG_MAP)} feature flags...")
    
    # Create feature flags using vectorized operations
    for col, keywords in TAG_MAP.items():
        pattern = '|'.join(keywords)
        df[col] = df['temp_tags_joined'].str.contains(pattern, na=False, case=False, regex=True)
        flag_count = df[col].sum()
        log_progress(f"   - {col}: {flag_count:,} models ({(flag_count/len(df)*100):.1f}%)")
    
    # Special case for science (exclude bigscience)
    log_progress("   Processing special 'has_science' flag...")
    df['has_science'] = (
        df['temp_tags_joined'].str.contains('science', na=False, case=False, regex=True) &
        ~df['temp_tags_joined'].str.contains('bigscience', na=False, case=False, regex=True)
    )
    science_count = df['has_science'].sum()
    log_progress(f"   - has_science: {science_count:,} models ({(science_count/len(df)*100):.1f}%)")
    
    # Create composite flags
    log_progress("   Creating composite flags...")
    df['is_audio_speech'] = (
        df['has_audio'] | df['has_speech'] |
        df['pipeline_tag'].str.contains('audio|speech', case=False, na=False, regex=True)
    )
    df['is_biomed'] = df['has_bio'] | df['has_med']
    
    audio_speech_count = df['is_audio_speech'].sum()
    biomed_count = df['is_biomed'].sum()
    log_progress(f"   - is_audio_speech: {audio_speech_count:,} models")
    log_progress(f"   - is_biomed: {biomed_count:,} models")
    
    # Clean up temporary column
    del df['temp_tags_joined']
    
    flag_time = time.time() - flag_start
    log_progress(f"✅ Feature flags created in {flag_time:.2f}s")
    
    return df

def debug_specific_model(df, model_id=None):
    """Debug processing for a specific model ID."""
    if not model_id or model_id not in df['id'].values:
        if model_id:
            log_progress(f"DEBUG: Model ID '{model_id}' not found in DataFrame")
        return
    
    log_progress(f"🔍 Debugging model: {model_id}")
    
    # Get model data
    model_data = df[df['id'] == model_id].iloc[0]
    
    log_progress(f"   Tags: {model_data['tags']}")
    log_progress(f"   Tags type: {type(model_data['tags'])}")
    
    # Check each feature flag
    feature_flags = [col for col in df.columns if col.startswith('has_') or col.startswith('is_')]
    for flag in feature_flags:
        if flag in model_data:
            log_progress(f"   {flag}: {model_data[flag]}")
    
    # Simulate temp_tags_joined
    if isinstance(model_data['tags'], list):
        simulated_temp_tags = '~~~'.join(str(t).lower().strip() for t in model_data['tags'] if pd.notna(t) and str(t).strip())
        log_progress(f"   Simulated tag string: '{simulated_temp_tags}'")
        
        # Manual regex check for robotics
        robot_keywords = ['robot', 'robotics', 'openvla', 'vla']
        robot_pattern = '|'.join(robot_keywords)
        manual_robot_check = bool(re.search(robot_pattern, simulated_temp_tags, flags=re.IGNORECASE))
        log_progress(f"   Manual robot check: {manual_robot_check}")

def analyze_tag_distribution(df):
    """Analyze and log tag distribution statistics."""
    log_progress("📈 Analyzing tag distribution...")
    
    if 'tags' not in df.columns:
        log_progress("   No tags column found")
        return
    
    # Count tag lengths
    tag_lengths = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    log_progress(f"   Tag statistics:")
    log_progress(f"   - Models with no tags: {(tag_lengths == 0).sum():,}")
    log_progress(f"   - Average tags per model: {tag_lengths.mean():.2f}")
    log_progress(f"   - Max tags per model: {tag_lengths.max()}")
    log_progress(f"   - Models with 10+ tags: {(tag_lengths >= 10).sum():,}")
    
    # Get most common tags
    all_tags = []
    for tag_list in df['tags']:
        if isinstance(tag_list, list):
            all_tags.extend([str(tag).lower().strip() for tag in tag_list if pd.notna(tag)])
    
    if all_tags:
        from collections import Counter
        tag_counts = Counter(all_tags)
        log_progress(f"   - Total unique tags: {len(tag_counts)}")
        log_progress(f"   - Top 10 most common tags:")
        for tag, count in tag_counts.most_common(10):
            log_progress(f"     {tag}: {count:,}")

if __name__ == "__main__":
    # Test the tag processor
    log_progress("Testing tag processor...")
    
    # Create sample data
    sample_data = {
        'id': ['test1', 'test2', 'test3'],
        'tags': [['robotics', 'vision'], 'audio,speech', ['text', 'nlp']]
    }
    df_test = pd.DataFrame(sample_data)
    
    df_test['tags'] = process_tags_for_series(df_test['tags'])
    df_test = create_feature_flags(df_test)
    
    log_progress("✅ Tag processor test completed")
    print(df_test[['id', 'tags', 'has_robot', 'has_audio', 'has_text']].to_string())