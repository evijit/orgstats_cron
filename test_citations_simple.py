#!/usr/bin/env python3
"""Simple test for citation fetching via Semantic Scholar API."""

print("🧪 Testing Citation Fetching via Semantic Scholar API...")
print("=" * 70)

# Test 1: Direct Semantic Scholar API
print("\n1. Testing Semantic Scholar API directly...")
try:
    from semanticscholar import SemanticScholar
    
    sch = SemanticScholar()
    
    # Test with the Transformer paper
    paper_id = "1706.03762"
    doi = f"10.48550/arXiv.{paper_id}"
    
    print(f"   Looking up DOI: {doi}")
    paper = sch.get_paper(f"DOI:{doi}")
    
    if paper:
        print(f"   ✅ Found paper: {paper.title}")
        print(f"   📊 Citation count: {paper.citationCount}")
    else:
        print("   ❌ Paper not found")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Using paperscraper's get_citations_by_doi
print("\n2. Testing paperscraper get_citations_by_doi...")
try:
    from paperscraper.citations import get_citations_by_doi
    
    paper_id = "1706.03762"
    doi = f"10.48550/arXiv.{paper_id}"
    
    print(f"   Looking up DOI: {doi}")
    citations = get_citations_by_doi(doi)
    
    if citations is not None:
        print(f"   ✅ Citations found: {citations}")
    else:
        print("   ❌ Citations not found (returned None)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Test our custom function
print("\n3. Testing our get_paper_citations function...")
try:
    import sys
    sys.path.insert(0, '/Users/avijit/Documents/orgstats_cron')
    from data_processor_papers import get_paper_citations
    
    paper_id = "1706.03762"
    paper_title = "Attention Is All You Need"
    
    print(f"   Paper ID: {paper_id}")
    print(f"   Paper Title: {paper_title}")
    
    citations = get_paper_citations(paper_id, paper_title)
    
    if citations is not None:
        print(f"   ✅ Citations found: {citations}")
    else:
        print("   ⚠️  Citations not found (returned None)")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with multiple papers
print("\n4. Testing with multiple arXiv papers...")
test_papers = [
    ("1706.03762", "Attention Is All You Need"),
    ("2106.09685", "LoRA: Low-Rank Adaptation"),
    ("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers"),
]

try:
    from data_processor_papers import get_paper_citations
    
    for paper_id, title in test_papers:
        citations = get_paper_citations(paper_id, title)
        status = f"{citations:,}" if citations is not None else "N/A"
        print(f"   {paper_id}: {status} citations")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ Citation test complete!")
