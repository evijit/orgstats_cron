#!/usr/bin/env python3
"""
Fetch citations for a specific subset of papers (for parallel job processing).
Usage: python fetch_citations_batch.py <start_idx> <end_idx>

Smart caching: Skips papers that were successfully fetched within the last 7 days.
"""

import sys
import pandas as pd
from data_processor_papers import get_paper_citations
from config_papers import CITATION_RATE_LIMIT_DELAY
import time
from datetime import datetime, timedelta
from utils import log_progress

def fetch_citations_batch(df, start_idx, end_idx, previous_data=None):
    """
    Fetch citations for papers in the specified range.
    Skips papers that were successfully fetched within the last 7 days.
    
    Args:
        df: DataFrame with all papers
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
        previous_data: Optional DataFrame with previous citation data (with citation_fetch_date column)
        
    Returns:
        DataFrame with citation data for the specified range
    """
    # Sort by upvotes to prioritize popular papers
    df_sorted = df.sort_values('paper_upvotes', ascending=False, na_position='last').reset_index(drop=True)
    
    # Select the subset
    df_subset = df_sorted.iloc[start_idx:end_idx].copy()
    
    log_progress(f"📚 Fetching citations for papers {start_idx:,} to {end_idx:,}")
    log_progress(f"   Processing {len(df_subset):,} papers")
    log_progress(f"   Estimated time: ~{len(df_subset) * CITATION_RATE_LIMIT_DELAY / 60:.1f} minutes")
    
    citation_counts = []
    semantic_scholar_ids = []
    fetch_dates = []
    successful_fetches = 0
    skipped_fresh = 0
    start_time = time.time()
    
    # Build lookup for previous data
    previous_lookup = {}
    if previous_data is not None:
        # Handle case where citation_fetch_date doesn't exist yet (first run with new system)
        has_fetch_dates = 'citation_fetch_date' in previous_data.columns
        
        for _, row in previous_data.iterrows():
            paper_id = row['paper_id']
            
            # If we have fetch dates, only cache fresh ones
            if has_fetch_dates:
                if pd.notna(row.get('citation_fetch_date')) and pd.notna(row.get('citation_count')):
                    previous_lookup[paper_id] = {
                        'citation_count': row['citation_count'],
                        'semantic_scholar_id': row.get('semantic_scholar_id'),
                        'fetch_date': row['citation_fetch_date']
                    }
            else:
                # First run: treat all existing citations as "old" (need refresh)
                # But keep the semantic_scholar_id for faster lookups
                if pd.notna(row.get('semantic_scholar_id')):
                    previous_lookup[paper_id] = {
                        'semantic_scholar_id': row.get('semantic_scholar_id'),
                        'fetch_date': None  # Mark as needing refresh
                    }
        
        if not has_fetch_dates:
            log_progress(f"   ⚠️  No fetch dates in previous data - will fetch all but use existing IDs for speed")
    
    # Calculate cutoff date (7 days ago)
    cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    for idx, row in enumerate(df_subset.itertuples(), 1):
        paper_id = getattr(row, 'paper_id', '')
        paper_title = getattr(row, 'paper_title', '')
        
        # Log each paper being processed (with count)
        log_progress(f"\n[{idx}/{len(df_subset)}] Paper ID: {paper_id}")
        
        # Check if we have fresh data (< 7 days old)
        if paper_id in previous_lookup:
            prev_fetch_date = previous_lookup[paper_id].get('fetch_date')
            if prev_fetch_date and prev_fetch_date >= cutoff_date:
                # Data is fresh, reuse it
                log_progress(f"   ♻️  Using cached data (fetched {prev_fetch_date})")
                citation_counts.append(previous_lookup[paper_id]['citation_count'])
                semantic_scholar_ids.append(previous_lookup[paper_id]['semantic_scholar_id'])
                fetch_dates.append(prev_fetch_date)
                successful_fetches += 1
                skipped_fresh += 1
                continue
        
        # Skip papers without titles to avoid wasting time
        if not paper_title or not isinstance(paper_title, str) or not paper_title.strip():
            log_progress(f"   ⚠️  Skipping: No title available")
            citation_counts.append(None)
            semantic_scholar_ids.append(None)
            fetch_dates.append(None)
        else:
            # Get existing semantic_scholar_id if available (for faster lookup)
            existing_ss_id = None
            if paper_id in previous_lookup:
                existing_ss_id = previous_lookup[paper_id].get('semantic_scholar_id')
            
            # Fetch with detailed logging enabled
            # Pass existing SS ID for faster lookup (by ID instead of title search)
            citations, ss_id, fetch_date = get_paper_citations(
                paper_id, paper_title, 
                log_details=True, 
                semantic_scholar_id=existing_ss_id
            )
            citation_counts.append(citations)
            semantic_scholar_ids.append(ss_id)
            fetch_dates.append(fetch_date)
            
            if citations is not None:
                successful_fetches += 1
        
        # No explicit delay needed - semanticscholar package handles rate limiting
        
        # Progress updates every 10 papers
        if idx % 10 == 0 or idx == len(df_subset):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(df_subset) - idx) / rate if rate > 0 else 0
            log_progress(f"\n📊 Progress: {idx:,}/{len(df_subset):,} papers processed " +
                        f"({successful_fetches} citations found, {skipped_fresh} cached, {rate:.2f} papers/sec, ETA: {eta/60:.1f}min)")
    
    df_subset['citation_count'] = citation_counts
    df_subset['semantic_scholar_id'] = semantic_scholar_ids
    df_subset['citation_fetch_date'] = fetch_dates
    
    elapsed_time = time.time() - start_time
    log_progress(f"✅ Batch completed in {elapsed_time/60:.1f} minutes")
    log_progress(f"   Found citations for {successful_fetches:,}/{len(df_subset):,} papers")
    log_progress(f"   Reused {skipped_fresh:,} cached citations (< 7 days old)")
    
    return df_subset

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fetch_citations_batch.py <start_idx> <end_idx>")
        sys.exit(1)
    
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    
    # Load the full dataset
    from data_fetcher_papers import fetch_raw_data
    
    log_progress(f"Loading papers data...")
    df, _ = fetch_raw_data()  # Unpack tuple (df, timestamp)
    
    log_progress(f"Total papers in dataset: {len(df):,}")
    
    # Try to load previous data for smart caching
    previous_data = None
    try:
        import os
        from huggingface_hub import hf_hub_download
        
        log_progress("Loading previous citation data from HuggingFace...")
        hf_token = os.environ.get('HF_TOKEN')
        previous_file = hf_hub_download(
            repo_id='evijit/paperverse_daily_data',
            filename='papers_with_semantic_taxonomy.parquet',
            repo_type='dataset',
            token=hf_token
        )
        previous_data = pd.read_parquet(previous_file)
        
        if 'citation_fetch_date' in previous_data.columns:
            fresh_count = (previous_data['citation_fetch_date'] >= 
                          (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')).sum()
            log_progress(f"   Found {fresh_count:,} papers with fresh citations (< 7 days old)")
        else:
            log_progress("   No fetch_date column found, will fetch all papers")
            
    except Exception as e:
        log_progress(f"   Could not load previous data (first run?): {e}")
    
    # Fetch citations for this batch
    df_batch = fetch_citations_batch(df, start_idx, end_idx, previous_data)
    
    # Save batch results
    output_file = f'citations_batch_{start_idx}_{end_idx}.parquet'
    df_batch.to_parquet(output_file, index=False)
    log_progress(f"💾 Saved batch results to {output_file}")
