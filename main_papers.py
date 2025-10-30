# --- main_papers.py ---
"""Main orchestrator for the Papers data processing pipeline with semantic taxonomy."""

import os
import sys
import time
import json
import pandas as pd
from utils import log_progress, log_memory_usage
from config_papers import (
    PROCESSED_PARQUET_FILE_PATH,
    FINAL_EXPECTED_COLUMNS,
    HF_REPO_ID,
    HF_REPO_TYPE
)
from data_fetcher_papers import fetch_raw_data, validate_raw_data
from data_processor_papers import (
    setup_initial_dataframe,
    enrich_data,
    apply_semantic_taxonomy
)

def save_processed_data(df):
    """Save processed DataFrame to parquet and CSV files."""
    log_progress(f"💾 Saving processed papers data...")
    
    try:
        # Save as Parquet
        log_progress(f"   Saving to Parquet: {PROCESSED_PARQUET_FILE_PATH}")
        df.to_parquet(PROCESSED_PARQUET_FILE_PATH, index=False, engine='pyarrow')
        
        # Also save as CSV for compatibility
        csv_path = PROCESSED_PARQUET_FILE_PATH.replace('.parquet', '.csv')
        log_progress(f"   Saving to CSV: {csv_path}")
        df.to_csv(csv_path, index=False)
        
        log_progress("✅ Data saved successfully.")
        return True
    except Exception as e:
        log_progress(f"❌ ERROR: Could not save processed data: {e}")
        return False

def save_summary_reports(df):
    """Save summary reports and statistics."""
    log_progress("📊 Generating summary reports...")
    
    try:
        total_papers = len(df)
        classified_papers = df['primary_category'].notna().sum()
        
        # Citation statistics
        papers_with_citations = df['citation_count'].notna().sum()
        total_citations = df['citation_count'].sum() if papers_with_citations > 0 else 0
        
        # Category distribution
        category_counts = df['primary_category'].value_counts()
        subcategory_counts = df['primary_subcategory'].value_counts()
        topic_counts = df['primary_topic'].value_counts()
        
        # Calculate matching statistics
        all_scores = []
        match_level_counts = {}
        
        for details in df['matched_keywords_details']:
            if details and isinstance(details, list):
                for match in details:
                    if isinstance(match, dict) and 'score' in match:
                        all_scores.append(match['score'])
                        level = match.get('match_level', 'unknown')
                        match_level_counts[level] = match_level_counts.get(level, 0) + 1
        
        # Save text report
        with open('taxonomy_report.txt', 'w') as f:
            f.write("SEMANTIC TAXONOMY CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total papers: {total_papers:,}\n")
            f.write(f"Classified papers: {classified_papers:,} ({classified_papers/total_papers*100:.1f}%)\n")
            f.write(f"Unclassified papers: {total_papers - classified_papers:,}\n")
            
            # Multi-classification statistics
            multi_cat_papers = df['taxonomy_categories'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False).sum()
            f.write(f"Multi-category papers: {multi_cat_papers:,} ({multi_cat_papers/total_papers*100:.1f}%)\n")
            avg_cats_per_paper = df['taxonomy_categories'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
            f.write(f"Average categories per paper: {avg_cats_per_paper:.2f}\n\n")
            
            f.write("CITATION STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Papers with citations: {papers_with_citations:,} ({papers_with_citations/total_papers*100:.1f}%)\n")
            if papers_with_citations > 0:
                f.write(f"Total citations: {int(total_citations):,}\n")
                f.write(f"Average citations per paper: {total_citations/papers_with_citations:.1f}\n\n")
            else:
                f.write("No citation data available\n\n")
            
            if all_scores:
                f.write(f"Average matching score: {sum(all_scores)/len(all_scores):.3f}\n")
                f.write(f"Total keyword matches: {len(all_scores):,}\n\n")
            
            f.write("TOP CATEGORIES\n")
            f.write("-" * 80 + "\n")
            for cat, count in category_counts.head(20).items():
                f.write(f"{str(cat):50s} {count:6d} ({count/total_papers*100:5.1f}%)\n")
            
            f.write("\n\nTOP SUBCATEGORIES\n")
            f.write("-" * 80 + "\n")
            for subcat, count in subcategory_counts.head(30).items():
                f.write(f"{str(subcat):50s} {count:6d} ({count/total_papers*100:5.1f}%)\n")
            
            f.write("\n\nTOP TOPICS\n")
            f.write("-" * 80 + "\n")
            for topic, count in topic_counts.head(50).items():
                f.write(f"{str(topic):50s} {count:6d} ({count/total_papers*100:5.1f}%)\n")
        
        log_progress("✅ Saved summary to taxonomy_report.txt")
        
        # Save JSON distribution
        # Multi-classification stats
        multi_cat_papers = df['taxonomy_categories'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False).sum()
        avg_cats_per_paper = df['taxonomy_categories'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        
        distribution_summary = {
            'total_papers': int(total_papers),
            'classified_papers': int(classified_papers),
            'coverage_percentage': float(classified_papers/total_papers*100) if total_papers > 0 else 0,
            'multi_classification': {
                'multi_category_papers': int(multi_cat_papers),
                'multi_category_percentage': float(multi_cat_papers/total_papers*100) if total_papers > 0 else 0,
                'average_categories_per_paper': float(avg_cats_per_paper)
            },
            'citation_statistics': {
                'papers_with_citations': int(papers_with_citations),
                'total_citations': int(total_citations),
                'average_citations': float(total_citations/papers_with_citations) if papers_with_citations > 0 else 0
            },
            'category_distribution': {k: int(v) for k, v in category_counts.head(20).items()},
            'subcategory_distribution': {k: int(v) for k, v in subcategory_counts.head(30).items()},
            'topic_distribution': {k: int(v) for k, v in topic_counts.head(50).items()},
            'match_statistics': {
                'total_matches': len(all_scores),
                'average_score': float(sum(all_scores)/len(all_scores)) if all_scores else 0,
                'matches_by_level': {k: int(v) for k, v in match_level_counts.items()}
            }
        }
        
        with open('taxonomy_distribution.json', 'w') as f:
            json.dump(distribution_summary, f, indent=2)
        
        log_progress("✅ Saved distribution to taxonomy_distribution.json")
        
        return True
    except Exception as e:
        log_progress(f"❌ ERROR generating summary reports: {e}")
        import traceback
        traceback.print_exc()
        return False

def upload_to_huggingface(df):
    """Upload processed data to HuggingFace repository."""
    log_progress(f"☁️  Uploading to HuggingFace: {HF_REPO_ID}")
    
    # Check if HF token is available
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        log_progress("⚠️  WARNING: HF_TOKEN not found in environment variables.")
        log_progress("   Skipping upload. Set HF_TOKEN to enable automatic upload.")
        return False
    
    try:
        from huggingface_hub import HfApi, create_repo
        
        api = HfApi(token=hf_token)
        
        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                exist_ok=True,
                token=hf_token
            )
            log_progress(f"   Repository ensured: {HF_REPO_ID}")
        except Exception as e:
            log_progress(f"   Repository exists or error creating: {e}")
        
        # Upload files
        files_to_upload = [
            PROCESSED_PARQUET_FILE_PATH,
            PROCESSED_PARQUET_FILE_PATH.replace('.parquet', '.csv'),
            'taxonomy_report.txt',
            'taxonomy_distribution.json'
        ]
        
        for file_path in files_to_upload:
            if os.path.exists(file_path):
                log_progress(f"   Uploading {file_path}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path,
                    repo_id=HF_REPO_ID,
                    repo_type=HF_REPO_TYPE,
                    token=hf_token
                )
                log_progress(f"   ✅ Uploaded {file_path}")
        
        log_progress(f"✅ All files uploaded to {HF_REPO_ID}")
        return True
        
    except ImportError:
        log_progress("❌ ERROR: huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        log_progress(f"❌ ERROR uploading to HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_pipeline():
    """Execute the complete papers data processing pipeline."""
    log_progress("🚀 Starting HuggingFace PAPERS Data Processing Pipeline")
    log_progress("=" * 70)
    
    pipeline_start = time.time()
    
    # Cleanup existing files
    if os.path.exists(PROCESSED_PARQUET_FILE_PATH):
        os.remove(PROCESSED_PARQUET_FILE_PATH)
    
    # Step 1: Data Fetching
    log_progress("\nSTEP 1: Data Fetching")
    try:
        df_raw, data_download_timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
    except Exception as e:
        log_progress(f"❌ Data fetching failed: {e}")
        return False
    
    # Step 2: Initial Data Processing
    log_progress("\nSTEP 2: Initial Data Processing & Enrichment")
    try:
        df = setup_initial_dataframe(df_raw, data_download_timestamp)
        df = enrich_data(df)
    except Exception as e:
        log_progress(f"❌ Initial data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Fetch Citations
    log_progress("\nSTEP 3: Fetching Citations")
    try:
        from data_processor_papers import fetch_citations
        df = fetch_citations(df)
    except Exception as e:
        log_progress(f"⚠️  Citation fetching failed (non-critical): {e}")
        log_progress("   Continuing without citations...")
        df['citation_count'] = None
    
    # Step 4: Semantic Taxonomy Mapping
    log_progress("\nSTEP 4: Semantic Taxonomy Mapping")
    try:
        df = apply_semantic_taxonomy(df)
    except Exception as e:
        log_progress(f"❌ Taxonomy mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Finalize DataFrame
    log_progress("\nSTEP 5: Finalizing DataFrame")
    try:
        for col in FINAL_EXPECTED_COLUMNS:
            if col not in df.columns:
                log_progress(f"⚠️  Final column '{col}' not found. Adding with default values.")
                df[col] = None
        
        final_columns_in_df = [col for col in FINAL_EXPECTED_COLUMNS if col in df.columns]
        df_final = df[final_columns_in_df]
        log_progress(f"Final DataFrame shape: {df_final.shape}")
    except Exception as e:
        log_progress(f"❌ Final processing failed: {e}")
        return False
    
    # Step 6: Save Results
    log_progress("\nSTEP 6: Save Results")
    if not save_processed_data(df_final):
        return False
    
    if not save_summary_reports(df_final):
        log_progress("⚠️  Summary reports generation failed, but continuing...")
    
    # Step 7: Upload to HuggingFace
    log_progress("\nSTEP 7: Upload to HuggingFace")
    upload_to_huggingface(df_final)
    
    # Final Summary
    total_time = time.time() - pipeline_start
    log_progress("\n" + "=" * 70)
    log_progress("🎉 PAPERS PIPELINE COMPLETED SUCCESSFULLY!")
    log_progress(f"   Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    log_progress(f"   Processed papers: {len(df_final):,}")
    log_progress(f"   Output files:")
    log_progress(f"      - {PROCESSED_PARQUET_FILE_PATH}")
    log_progress(f"      - {PROCESSED_PARQUET_FILE_PATH.replace('.parquet', '.csv')}")
    log_progress(f"      - taxonomy_report.txt")
    log_progress(f"      - taxonomy_distribution.json")
    
    return True

if __name__ == "__main__":
    try:
        if main_pipeline():
            log_progress("✅ Script completed successfully")
            sys.exit(0)
        else:
            log_progress("❌ Script failed")
            sys.exit(1)
    except KeyboardInterrupt:
        log_progress("\n⚠️  Script interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_progress(f"\n💥 UNEXPECTED FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
