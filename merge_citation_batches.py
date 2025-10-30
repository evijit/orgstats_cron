#!/usr/bin/env python3
"""
Merge citation batch results from parallel jobs.
"""

import pandas as pd
import glob
import os
from utils import log_progress

def merge_citation_batches(output_file='citations_merged.parquet'):
    """
    Merge all citation batch files into a single file.
    
    Args:
        output_file: Output filename for merged results
        
    Returns:
        DataFrame with merged citation data
    """
    # Find all batch files
    batch_files = sorted(glob.glob('citations_batch_*.parquet'))
    
    if not batch_files:
        log_progress("❌ No citation batch files found!")
        return None
    
    log_progress(f"📦 Merging {len(batch_files)} citation batch files...")
    
    dfs = []
    total_papers = 0
    total_with_citations = 0
    
    for batch_file in batch_files:
        log_progress(f"   Loading {batch_file}...")
        df_batch = pd.read_parquet(batch_file)
        dfs.append(df_batch)
        
        total_papers += len(df_batch)
        total_with_citations += df_batch['citation_count'].notna().sum()
    
    # Combine all batches
    df_merged = pd.concat(dfs, ignore_index=True)
    
    log_progress(f"✅ Merged {len(batch_files)} batches:")
    log_progress(f"   Total papers: {total_papers:,}")
    log_progress(f"   Papers with citations: {total_with_citations:,} ({total_with_citations/total_papers*100:.1f}%)")
    
    # Save merged results
    df_merged.to_parquet(output_file, index=False)
    log_progress(f"💾 Saved merged results to {output_file}")
    
    # Show citation statistics
    valid_citations = df_merged['citation_count'].dropna()
    if len(valid_citations) > 0:
        log_progress(f"\n📊 Citation Statistics:")
        log_progress(f"   Mean: {valid_citations.mean():.1f}")
        log_progress(f"   Median: {valid_citations.median():.1f}")
        log_progress(f"   Max: {valid_citations.max():.0f}")
        log_progress(f"   Total citations: {valid_citations.sum():.0f}")
    
    return df_merged

if __name__ == '__main__':
    df = merge_citation_batches()
    
    if df is not None:
        log_progress("\n✅ Citation merge complete!")
        
        # Clean up batch files
        batch_files = glob.glob('citations_batch_*.parquet')
        log_progress(f"\n🧹 Cleaning up {len(batch_files)} batch files...")
        for f in batch_files:
            os.remove(f)
        log_progress("✅ Cleanup complete!")
