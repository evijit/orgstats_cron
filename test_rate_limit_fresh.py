#!/usr/bin/env python3
"""
Test Semantic Scholar rate limiting with fresh start.
Wait 5+ minutes, then test with conservative delays.
"""

import time
from datetime import datetime
from data_processor_papers import get_paper_citations
from utils import log_progress

def test_rate_limiting():
    """Test citation fetching with conservative rate limiting."""
    
    # Test papers (using known titles)
    test_papers = [
        ("test1", "Attention Is All You Need"),
        ("test2", "BERT: Pre-training of Deep Bidirectional Transformers"),
        ("test3", "ImageNet Classification with Deep Convolutional Neural Networks"),
        ("test4", "Generative Adversarial Networks"),
        ("test5", "Deep Residual Learning for Image Recognition"),
    ]
    
    log_progress("=" * 80)
    log_progress(f"🧪 Testing Semantic Scholar Rate Limiting")
    log_progress(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    log_progress(f"   Strategy: 5 seconds between requests")
    log_progress(f"   Expected: 100 requests per 5 minutes = 1 every 3 seconds")
    log_progress(f"   Our pace: 1 every 5 seconds (conservative)")
    log_progress("=" * 80)
    
    results = []
    successful = 0
    rate_limited = 0
    
    for i, (paper_id, title) in enumerate(test_papers, 1):
        log_progress(f"\n[{i}/{len(test_papers)}] Testing: {title[:60]}...")
        
        start = time.time()
        citations, ss_id = get_paper_citations(paper_id, title, log_details=True)
        elapsed = time.time() - start
        
        result = {
            'paper_id': paper_id,
            'title': title,
            'citations': citations,
            'ss_id': ss_id,
            'elapsed': elapsed,
            'success': citations is not None
        }
        results.append(result)
        
        if citations is not None:
            successful += 1
        else:
            rate_limited += 1
        
        # Wait 5 seconds before next request (except after last one)
        if i < len(test_papers):
            log_progress(f"   ⏱️  Waiting 5 seconds before next request...")
            time.sleep(5)
    
    # Summary
    log_progress("\n" + "=" * 80)
    log_progress("📊 TEST RESULTS")
    log_progress("=" * 80)
    log_progress(f"Total papers tested: {len(test_papers)}")
    log_progress(f"Successful: {successful} ({successful/len(test_papers)*100:.1f}%)")
    log_progress(f"Failed/Rate-limited: {rate_limited} ({rate_limited/len(test_papers)*100:.1f}%)")
    log_progress("")
    log_progress("Details:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        log_progress(f"  {status} {r['paper_id']}: {r['citations'] if r['success'] else 'FAILED'} citations ({r['elapsed']:.2f}s)")
    log_progress("=" * 80)
    
    if successful == len(test_papers):
        log_progress("✅ All requests successful! Rate limiting strategy is working.")
        log_progress("   Ready to deploy to production workflow.")
    elif successful >= len(test_papers) * 0.8:  # 80%+ success
        log_progress("⚠️  Mostly successful but some failures.")
        log_progress("   May need longer delays or API key for production.")
    else:
        log_progress("❌ High failure rate. Options:")
        log_progress("   1. Wait longer for rate limit to fully reset")
        log_progress("   2. Increase delay to 8-10 seconds")
        log_progress("   3. Get Semantic Scholar API key for higher limits")

if __name__ == '__main__':
    log_progress("⏰ This test assumes rate limit has been reset (5+ minutes since last API call)")
    log_progress("   If you've made recent requests, wait 5-10 minutes before running this test.")
    log_progress("")
    input("Press ENTER when ready to start test...")
    
    test_rate_limiting()
