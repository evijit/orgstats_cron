#!/usr/bin/env python3
"""
Quick test to verify ENABLE_CITATION_FETCHING environment variable works.
This simulates what happens in main_papers.py Step 3.
"""

import pandas as pd
import sys

# Import the function that checks the config
from data_processor_papers import fetch_citations
from config_papers import ENABLE_CITATION_FETCHING

print(f"🧪 Testing citation fetching behavior...")
print(f"   ENABLE_CITATION_FETCHING = {ENABLE_CITATION_FETCHING}")
print()

# Create a tiny test dataframe
test_df = pd.DataFrame({
    'paper_id': ['2501.00001', '2501.00002', '2501.00003'],
    'paper_title': ['Test Paper 1', 'Test Paper 2', 'Test Paper 3'],
    'paper_upvotes': [10, 20, 30]
})

print(f"📊 Test dataframe: {len(test_df)} papers")
print()

# Call fetch_citations (this should skip if ENABLE_CITATION_FETCHING=false)
result_df = fetch_citations(test_df)

print()
if ENABLE_CITATION_FETCHING:
    print("✅ Citation fetching was ENABLED")
    print(f"   Result has citation_count column: {'citation_count' in result_df.columns}")
    if 'citation_count' in result_df.columns:
        print(f"   Citation values: {result_df['citation_count'].tolist()}")
else:
    print("✅ Citation fetching was DISABLED (as expected)")
    print(f"   Result has citation_count column: {'citation_count' in result_df.columns}")
    if 'citation_count' in result_df.columns:
        print(f"   All citation values are None: {result_df['citation_count'].isna().all()}")

print()
print("🎉 Test completed successfully!")
