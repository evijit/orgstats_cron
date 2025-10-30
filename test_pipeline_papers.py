# --- test_pipeline_papers.py ---
"""Integration test for the papers processing pipeline."""

import os
import sys
from utils import log_progress
from data_fetcher_papers import fetch_raw_data, validate_raw_data
from data_processor_papers import setup_initial_dataframe, enrich_data, apply_semantic_taxonomy

def run_test_pipeline():
    log_progress("🚀 Starting PAPERS pipeline integration test...")
    try:
        # 1. Fetch
        log_progress("--- Step 1: Fetching test data ---")
        df_raw, timestamp = fetch_raw_data()
        validate_raw_data(df_raw)
        assert not df_raw.empty, "Raw data fetch returned an empty DataFrame."
        log_progress("✅ Raw data fetched and validated.")

        # 2. Process
        log_progress("\n--- Step 2: Processing data ---")
        df = setup_initial_dataframe(df_raw, timestamp)
        df = enrich_data(df)
        assert 'organization_name' in df.columns, "Organization_name column not created."
        log_progress("✅ Data processing and enrichment complete.")

        # 3. Fetch Citations (optional, may skip in test)
        log_progress("\n--- Step 3: Fetching citations (test mode) ---")
        try:
            from data_processor_papers import fetch_citations
            df = fetch_citations(df)
            log_progress("✅ Citation fetching complete.")
        except Exception as e:
            log_progress(f"⚠️  Citation fetching skipped in test: {e}")
            df['citation_count'] = None
        
        # 4. Taxonomy Mapping
        log_progress("\n--- Step 4: Applying semantic taxonomy ---")
        df = apply_semantic_taxonomy(df)
        assert 'primary_category' in df.columns, "Primary category column not created."
        assert 'taxonomy_categories' in df.columns, "Taxonomy categories column not created."
        log_progress("✅ Semantic taxonomy mapping complete.")

        # Final validation
        log_progress("\n--- Step 5: Final validation ---")
        final_rows, final_cols = df.shape
        log_progress(f"Final test DataFrame shape: ({final_rows}, {final_cols})")
        assert final_rows > 0, "Final DataFrame is empty."
        assert final_cols > 10, "Final DataFrame has too few columns."
        
        # Show sample results
        classified_count = df['primary_category'].notna().sum()
        log_progress(f"Classified papers: {classified_count}/{final_rows} ({classified_count/final_rows*100:.1f}%)")
        
        if classified_count > 0:
            log_progress("\nSample classified papers:")
            sample = df[df['primary_category'].notna()].head(3)
            for idx, row in sample.iterrows():
                log_progress(f"  Paper: {row['paper_id']}")
                log_progress(f"    Title: {row.get('paper_title', 'N/A')}")
                log_progress(f"    Category: {row['primary_category']}")
                log_progress(f"    Subcategory: {row['primary_subcategory']}")
                log_progress(f"    Topic: {row['primary_topic']}")

    except Exception as e:
        log_progress(f"❌ PAPERS pipeline integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    log_progress("\n🎉✅ PAPERS pipeline integration test PASSED! 🎉")

if __name__ == "__main__":
    if 'TEST_DATA_LIMIT' not in os.environ:
        print("❌ ERROR: Set 'TEST_DATA_LIMIT' for testing.")
        print("   Example: export TEST_DATA_LIMIT=50")
        sys.exit(1)
    run_test_pipeline()
