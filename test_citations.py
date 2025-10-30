#!/usr/bin/env python3
"""Test citation fetching with title only, returning both citations and S2 ID."""

import sys
sys.path.insert(0, '/Users/avijit/Documents/orgstats_cron')

from data_processor_papers import get_paper_citations

print("🧪 Testing Citation Fetching (Title Only + S2 ID)...")
print("=" * 70)

# Test papers
test_cases = [
    ("Attention Is All You Need", "1706.03762"),
    ("LoRA: Low-Rank Adaptation of Large Language Models", "2106.09685"),
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
]

print("\nSearching by title (first result):")
print("-" * 70)

for title, arxiv_id in test_cases:
    print(f"\n📄 {title[:50]}...")
    
    citations, ss_id = get_paper_citations(arxiv_id, title)
    
    if citations is not None:
        print(f"   ✅ Citations: {citations:,}")
    else:
        print(f"   ⚠️  No citations found")
    
    if ss_id:
        print(f"   🆔 Semantic Scholar ID: {ss_id}")
    else:
        print(f"   ⚠️  No S2 ID found")

print("\n" + "=" * 70)
print("✅ Test complete!")
