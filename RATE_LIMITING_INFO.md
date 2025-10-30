# Semantic Scholar API Rate Limiting

## Current Status

The citation fetching feature uses the Semantic Scholar API to retrieve citation counts for papers. Understanding and respecting rate limits is critical for the workflow to function properly.

## Official Rate Limits

**Without API Key:** 100 requests per 5 minutes
- Average: 1 request every 3 seconds
- Sliding window (previous requests count toward the limit)

**With API Key:** 5,000 requests per 5 minutes
- Average: ~16 requests per second
- Much more suitable for large-scale batch processing

## Current Implementation

### Strategy
- **5 seconds between ALL requests** (conservative approach)
- **30 second wait on 429 errors** with single retry
- **10 second timeout** per request
- **Skip papers without titles** to avoid wasted API calls

### Code Locations
- `data_processor_papers.py`: Core `get_paper_citations()` function with retry logic
- `fetch_citations_batch.py`: Batch processing with 5s delays between requests
- `config_papers.py`: `CITATION_RATE_LIMIT_DELAY = 5` setting

## Performance Estimates

### Without API Key (Current)
For **200 papers per job**:
- Time: ~1,800 seconds = **30 minutes per job**
- Success rate: ~70-80% (may vary based on rate limit accumulation)
- GitHub Actions limit: 6 hours per job ✅ (plenty of headroom)

For **10,556 total papers** across 53 jobs:
- Total time: ~26.5 hours (distributed across parallel jobs)
- With 15 parallel jobs per wave: ~2 hours per wave
- 4 waves sequentially: ~**8 hours total**

### With API Key (Recommended)
For **200 papers per job**:
- Time: ~400 seconds = **6-7 minutes per job**
- Success rate: ~100%
- Much faster and more reliable

For **10,556 total papers**:
- Total time: ~5-6 hours (distributed)
- With 15 parallel jobs per wave: ~25-30 minutes per wave
- 4 waves sequentially: ~**2 hours total**

## Testing Procedure

### Before Production Deployment

1. **Wait for rate limit reset**: 5+ minutes since last API call
2. **Run fresh test**: `python test_rate_limit_fresh.py`
3. **Verify success rate**: Should be 100% for 5 test papers
4. **If successful**: Deploy to production workflow
5. **If rate limited**: Wait longer or get API key

### Local Testing
```bash
# Wait 5+ minutes for fresh start
python test_rate_limit_fresh.py

# Test small batch (will take ~90 seconds for 10 papers)
python fetch_citations_batch.py 0 10

# Check results
python -c "import pandas as pd; df = pd.read_parquet('citations_batch_0_10.parquet'); print(df[['paper_id', 'citation_count']].to_string())"
```

## Options for Production

### Option 1: Current Strategy (No API Key)
✅ **Pros:**
- No API key needed
- Free
- Works immediately

❌ **Cons:**
- Slower (~8 hours total)
- 70-80% success rate (some papers may be skipped)
- Vulnerable to rate limit accumulation

**Recommended for:** Small datasets, testing, or if API key is not available

### Option 2: Get Semantic Scholar API Key (Recommended)
✅ **Pros:**
- 50x higher rate limit (5,000 req/5min)
- ~100% success rate
- Faster (~2 hours total)
- More reliable

❌ **Cons:**
- Requires API key registration
- Need to add key to GitHub Secrets

**How to get API key:**
1. Visit: https://www.semanticscholar.org/product/api
2. Register for API key (free, academic use)
3. Add to GitHub Secrets as `SEMANTIC_SCHOLAR_API_KEY`
4. Update `data_processor_papers.py` to include header:
   ```python
   headers = {"x-api-key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY")}
   response = requests.get(url, params=params, headers=headers, timeout=10)
   ```

**Recommended for:** Production deployments, large datasets

### Option 3: Alternative Citation Sources
Consider other APIs with different rate limits:
- **OpenAlex**: More generous rate limits
- **Crossref**: Academic citation database
- **Google Scholar** (via unofficial APIs): Higher limits but less reliable

## Production Readiness Checklist

- [x] REST API implementation (faster than Python package)
- [x] Conservative 5s delays between requests
- [x] Retry logic on 429 errors (30s wait)
- [x] Skip papers without titles
- [x] Detailed logging for monitoring
- [x] 4-wave parallel processing (15+15+15+8 jobs)
- [x] All waves in single daily_update.yml workflow
- [ ] **Fresh rate limit test** (run `test_rate_limit_fresh.py`)
- [ ] **Small batch validation** (10-20 papers)
- [ ] **Decision**: With or without API key?
- [ ] **If using API key**: Add to GitHub Secrets and update code

## Current Blocking Issue

**Rate limit accumulation from testing:** Previous local test runs have used up rate limit quota. Need to wait 5+ minutes for full reset before validating the production strategy.

**Next Step:** Run `test_rate_limit_fresh.py` after waiting for rate limit reset to confirm 5-second delays work consistently.
