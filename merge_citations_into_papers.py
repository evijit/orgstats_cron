#!/usr/bin/env python3
"""
Merge citation data into the final papers dataset.
"""

import pandas as pd
from utils import log_progress

def merge_citations_into_papers():
    """Merge citation data from parallel jobs into final dataset."""
    
    log_progress('📦 Merging citations into final dataset...')
    
    # Load the papers with taxonomy
    df_papers = pd.read_parquet('papers_with_semantic_taxonomy.parquet')
    log_progress(f'   Loaded papers dataset: {len(df_papers):,} papers')
    
    # Load the merged citations
    df_citations = pd.read_parquet('citations_merged.parquet')
    log_progress(f'   Loaded citations dataset: {len(df_citations):,} papers with citations')
    
    # Merge on paper_id
    df_final = df_papers.merge(
        df_citations[['paper_id', 'citation_count', 'semantic_scholar_id']], 
        on='paper_id', 
        how='left',
        suffixes=('', '_from_batch')
    )
    
    # Use citation data from batch if available
    if 'citation_count_from_batch' in df_final.columns:
        df_final['citation_count'] = df_final['citation_count_from_batch'].fillna(df_final.get('citation_count'))
        df_final['semantic_scholar_id'] = df_final['semantic_scholar_id_from_batch'].fillna(df_final.get('semantic_scholar_id'))
        df_final = df_final.drop(columns=['citation_count_from_batch', 'semantic_scholar_id_from_batch'])
    
    # Save final dataset
    df_final.to_parquet('papers_with_semantic_taxonomy.parquet', index=False)
    df_final.to_csv('papers_with_semantic_taxonomy.csv', index=False)
    
    # Show statistics
    with_citations = df_final['citation_count'].notna().sum()
    log_progress(f'✅ Merged! Final dataset has {len(df_final):,} papers')
    log_progress(f'   Papers with citations: {with_citations:,} ({with_citations/len(df_final)*100:.1f}%)')
    
    return df_final

if __name__ == '__main__':
    merge_citations_into_papers()
