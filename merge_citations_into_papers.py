#!/usr/bin/env python3
"""
Merge citation data into the final papers dataset.
Preserves previous citation data when new fetch fails (incremental updates).
"""

import pandas as pd
import os
from huggingface_hub import hf_hub_download
from utils import log_progress

def merge_citations_into_papers():
    """
    Merge citation data from parallel jobs into final dataset.
    Preserves previous day's citations when new data is missing.
    """
    
    log_progress('📦 Merging citations into final dataset...')
    
    # Load the papers with taxonomy (current run)
    df_papers = pd.read_parquet('papers_with_semantic_taxonomy.parquet')
    log_progress(f'   Loaded papers dataset: {len(df_papers):,} papers')
    
    # Load the merged citations (from today's batch jobs)
    df_citations = pd.read_parquet('citations_merged.parquet')
    log_progress(f'   Loaded citations dataset: {len(df_citations):,} papers with citations')
    
    # Try to download previous day's data to preserve existing citations
    previous_citations = {}
    try:
        log_progress('   Downloading previous day\'s data from HuggingFace...')
        hf_token = os.environ.get('HF_TOKEN')
        previous_file = hf_hub_download(
            repo_id='evijit/paperverse_daily_data',
            filename='papers_with_semantic_taxonomy.parquet',
            repo_type='dataset',
            token=hf_token
        )
        df_previous = pd.read_parquet(previous_file)
        
        # Extract previous citations (only keep valid ones)
        prev_with_citations = df_previous[df_previous['citation_count'].notna()]
        for _, row in prev_with_citations.iterrows():
            previous_citations[row['paper_id']] = {
                'citation_count': row['citation_count'],
                'semantic_scholar_id': row.get('semantic_scholar_id')
            }
        
        log_progress(f'   Found {len(previous_citations):,} papers with citations from previous day')
    except Exception as e:
        log_progress(f'   ⚠️  Could not load previous data (first run?): {e}')
    
    # Merge new citations on paper_id
    df_final = df_papers.merge(
        df_citations[['paper_id', 'citation_count', 'semantic_scholar_id']], 
        on='paper_id', 
        how='left',
        suffixes=('', '_new')
    )
    
    # Apply incremental update logic
    if 'citation_count_new' in df_final.columns:
        # For each paper, use new data if available, otherwise use previous data
        for idx, row in df_final.iterrows():
            paper_id = row['paper_id']
            new_citation = row['citation_count_new']
            new_ss_id = row.get('semantic_scholar_id_new')
            
            # If new data is missing but we have previous data, use previous
            if pd.isna(new_citation) and paper_id in previous_citations:
                df_final.at[idx, 'citation_count_new'] = previous_citations[paper_id]['citation_count']
                df_final.at[idx, 'semantic_scholar_id_new'] = previous_citations[paper_id]['semantic_scholar_id']
        
        # Replace with merged data
        df_final['citation_count'] = df_final['citation_count_new']
        df_final['semantic_scholar_id'] = df_final['semantic_scholar_id_new']
        df_final = df_final.drop(columns=['citation_count_new', 'semantic_scholar_id_new'])
    else:
        # Fallback: use previous data for papers with no citation data
        for idx, row in df_final.iterrows():
            paper_id = row['paper_id']
            if pd.isna(row.get('citation_count')) and paper_id in previous_citations:
                df_final.at[idx, 'citation_count'] = previous_citations[paper_id]['citation_count']
                df_final.at[idx, 'semantic_scholar_id'] = previous_citations[paper_id]['semantic_scholar_id']
    
    # Save final dataset
    df_final.to_parquet('papers_with_semantic_taxonomy.parquet', index=False)
    df_final.to_csv('papers_with_semantic_taxonomy.csv', index=False)
    
    # Show statistics
    with_citations = df_final['citation_count'].notna().sum()
    new_citations = df_citations['citation_count'].notna().sum()
    preserved_citations = with_citations - new_citations
    
    log_progress(f'✅ Merged! Final dataset has {len(df_final):,} papers')
    log_progress(f'   Papers with citations: {with_citations:,} ({with_citations/len(df_final)*100:.1f}%)')
    log_progress(f'   New citations fetched today: {new_citations:,}')
    log_progress(f'   Citations preserved from previous day: {preserved_citations:,}')
    
    return df_final

if __name__ == '__main__':
    merge_citations_into_papers()
