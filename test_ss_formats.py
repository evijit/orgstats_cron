#!/usr/bin/env python3
"""Test Semantic Scholar API with correct paper ID formats."""

print("🧪 Testing Semantic Scholar API formats...")
print("=" * 70)

from semanticscholar import SemanticScholar

sch = SemanticScholar()

# From the URL: https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776
# The paper has a Semantic Scholar ID

test_queries = [
    # Try the S2 paper ID from URL
    ("S2 Paper ID", "204e3073870fae3d05bcbc2f6a8e263d9b72e776"),
    
    # Try arXiv format
    ("ARXIV prefix", "ARXIV:1706.03762"),
    ("arXiv prefix", "arXiv:1706.03762"),
    
    # Try CorpusId (sometimes works)
    ("CorpusID", "CorpusID:13756489"),
    
    # Try just the arxiv number
    ("Plain number", "1706.03762"),
]

for label, query in test_queries:
    print(f"\n{label}: {query}")
    try:
        paper = sch.get_paper(query)
        if paper:
            print(f"   ✅ Found: {paper.title[:60]}...")
            print(f"   📊 Citations: {paper.citationCount}")
            print(f"   🆔 Paper ID: {paper.paperId}")
            print(f"   📝 ArXiv ID: {paper.externalIds.get('ArXiv', 'N/A')}")
        else:
            print("   ❌ Not found")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")

print("\n" + "=" * 70)
print("Testing search by arXiv ID...")

# Also try searching
try:
    results = sch.search_paper("arXiv:1706.03762", limit=1)
    for paper in results:
        print(f"✅ Search found: {paper.title}")
        print(f"   Citations: {paper.citationCount}")
        print(f"   ArXiv: {paper.externalIds.get('ArXiv', 'N/A')}")
except Exception as e:
    print(f"❌ Search error: {e}")
