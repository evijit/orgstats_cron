#!/usr/bin/env python3
"""Test citation fetching with title and authors."""

import sys
sys.path.insert(0, '/Users/avijit/Documents/orgstats_cron')

from data_processor_papers import get_paper_citations

print("🧪 Testing Citation Fetching with Title + Authors...")
print("=" * 70)

# Test papers with different author formats
test_cases = [
    {
        "name": "Attention Is All You Need",
        "paper_id": "1706.03762",
        "paper_title": "Attention Is All You Need",
        "paper_authors": [
            {"name": "Ashish Vaswani"},
            {"name": "Noam Shazeer"},
        ]
    },
    {
        "name": "LoRA",
        "paper_id": "2106.09685",
        "paper_title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "paper_authors": [{"name": "Edward Hu"}]
    },
    {
        "name": "BERT",
        "paper_id": "1810.04805",
        "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "paper_authors": "Jacob Devlin, Ming-Wei Chang"  # Test string format
    },
]

print("\nTesting with title + first author:")
print("-" * 70)

for test in test_cases:
    print(f"\n📄 {test['name']}")
    print(f"   Title: {test['paper_title'][:50]}...")
    
    authors = test['paper_authors']
    if isinstance(authors, list) and len(authors) > 0:
        first_author = authors[0].get('name', 'N/A') if isinstance(authors[0], dict) else str(authors[0])
    elif isinstance(authors, str):
        first_author = authors.split(',')[0].strip()
    else:
        first_author = "N/A"
    
    print(f"   First Author: {first_author}")
    
    citations = get_paper_citations(
        test['paper_id'],
        test['paper_title'],
        test['paper_authors']
    )
    
    if citations is not None:
        print(f"   ✅ Citations: {citations:,}")
    else:
        print(f"   ⚠️  No citations found")

print("\n" + "=" * 70)
print("✅ Test complete!")
