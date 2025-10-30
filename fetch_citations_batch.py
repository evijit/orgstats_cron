#!/usr/bin/env python3
"""
Fetch citations for a specific subset of papers (for parallel job processing).
Usage: python fetch_citations_batch.py <start_idx> <end_idx>
"""

import sys
import pandas as pd
from data_processor_papers import get_paper_citations
from config_papers import CITATION_RATE_LIMIT_DELAY
import time
from utils import log_progress

def fetch_citations_batch(df, start_idx, end_idx):
    """
    Fetch citations for papers in the specified range.
    
    Args:
        df: DataFrame with all papers
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
        
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
    successful_fetches = 0
    start_time = time.time()
    
    for idx, row in enumerate(df_subset.itertuples(), 1):
        paper_id = getattr(row, 'paper_id', '')
        paper_title = getattr(row, 'paper_title', '')
        
        # Log each paper being processed (with count)
        log_progress(f"\n[{idx}/{len(df_subset)}] Paper ID: {paper_id}")
        
        # Skip papers without titles to avoid wasting time
        if not paper_title or not isinstance(paper_title, str) or not paper_title.strip():
            log_progress(f"   ⚠️  Skipping: No title available")
            citation_counts.append(None)
            semantic_scholar_ids.append(None)
        else:
            # Fetch with detailed logging enabled
            citations, ss_id = get_paper_citations(paper_id, paper_title, log_details=True)
            citation_counts.append(citations)
            semantic_scholar_ids.append(ss_id)
            
            if citations is not None:
                successful_fetches += 1
        
        # No explicit delay needed - semanticscholar package handles rate limiting
        
        # Progress updates every 10 papers
        if idx % 10 == 0 or idx == len(df_subset):
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(df_subset) - idx) / rate if rate > 0 else 0
            log_progress(f"\n📊 Progress: {idx:,}/{len(df_subset):,} papers processed " +
                        f"({successful_fetches} citations found, {rate:.2f} papers/sec, ETA: {eta/60:.1f}min)")
    
    df_subset['citation_count'] = citation_counts
    df_subset['semantic_scholar_id'] = semantic_scholar_ids
    
    elapsed_time = time.time() - start_time
    log_progress(f"✅ Batch completed in {elapsed_time/60:.1f} minutes")
    log_progress(f"   Found citations for {successful_fetches:,}/{len(df_subset):,} papers")
    
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
    
    # Fetch citations for this batch
    df_batch = fetch_citations_batch(df, start_idx, end_idx)
    
    # Save batch results
    output_file = f'citations_batch_{start_idx}_{end_idx}.parquet'
    df_batch.to_parquet(output_file, index=False)
    log_progress(f"💾 Saved batch results to {output_file}")
