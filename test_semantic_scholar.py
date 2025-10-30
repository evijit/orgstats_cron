#!/usr/bin/env python3
"""Test different ways to query Semantic Scholar API."""

print("🧪 Testing Semantic Scholar API with different formats...")
print("=" * 70)

from semanticscholar import SemanticScholar

sch = SemanticScholar()
paper_id = "1706.03762"

# Test different formats
formats = [
    ("arXiv ID", f"arXiv:{paper_id}"),
    ("ArXiv ID", f"ArXiv:{paper_id}"),
    ("ARXIV ID", f"ARXIV:{paper_id}"),
    ("DOI format", f"DOI:10.48550/arXiv.{paper_id}"),
    ("Plain arXiv", paper_id),
    ("URL format", f"https://arxiv.org/abs/{paper_id}"),
]

for label, query in formats:
    print(f"\n{label}: {query}")
    try:
        paper = sch.get_paper(query)
        if paper:
            print(f"   ✅ Found: {paper.title}")
            print(f"   📊 Citations: {paper.citationCount}")
        else:
            print("   ❌ Not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
