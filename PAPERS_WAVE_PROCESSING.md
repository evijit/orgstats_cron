# Papers Citation Processing - Wave System

## Overview

Due to GitHub Actions concurrent job limits (20 jobs for public repos) and the large number of papers (~10,519), we've implemented a wave-based processing system that allows you to process papers in batches.

**Important:** The wave system is **separate from** the daily update workflow. It's designed for manual execution because:
- Each wave takes ~5 hours
- Waves must run sequentially (wave 0, then 1, then 2, etc.)
- GitHub Actions workflow_dispatch doesn't support automatic sequential triggers
- You manually trigger each wave after the previous one completes

The original `daily_update.yml` still runs on schedule and handles models and datasets, but caps papers at 20 jobs (4,000 papers max).

## The Math

- **Total papers**: ~10,519
- **Papers per job**: 200 (~5 hours per job based on observed API performance)
- **Total jobs needed**: ~53 jobs
- **Jobs per wave**: 15 (configurable, default 15 to leave room for other workflows)
- **Number of waves**: 4 waves (15 + 15 + 15 + 8)

## Workflows

### 1. `papers_citations_waves.yml` - Process Citations in Waves

This is the main workflow you'll run multiple times, once per wave.

**How to run:**
1. Go to Actions → "Papers Citations Processing (Waves)"
2. Click "Run workflow"
3. Enter the wave number (starting from 0)
4. Optionally adjust jobs_per_wave (default: 15)
5. Click "Run workflow"

**Wave sequence for 10,519 papers:**
- Wave 0: Jobs 0-14 (papers 0-2,999) - 15 jobs
- Wave 1: Jobs 15-29 (papers 3,000-5,999) - 15 jobs  
- Wave 2: Jobs 30-44 (papers 6,000-8,999) - 15 jobs
- Wave 3: Jobs 45-52 (papers 9,000-10,519) - 8 jobs

Each wave takes ~5 hours (all jobs run in parallel within the wave).

**Important:** Each wave uploads artifacts with retention of 7 days, so you need to complete all waves and the final merge within that time.

### 2. `papers_final_merge.yml` - Merge All Waves

Run this AFTER all waves are complete.

**What it does:**
1. Downloads all citation artifacts from all waves
2. Merges them into a single file
3. Processes papers with semantic taxonomy
4. Merges citations into the final output
5. Uploads to HuggingFace dataset

**How to run:**
1. Wait for all waves (0-3) to complete successfully
2. Go to Actions → "Papers Final Merge (After All Waves)"
3. Click "Run workflow"
4. Click "Run workflow"

## Manual Processing Commands

If you want to test locally or run specific batches:

```bash
# Test wave matrix generation
python generate_matrix_wave.py 10519 0 15

# Process a specific batch locally
python fetch_citations_batch.py 0 200

# Merge all citation batches
python merge_citation_batches.py

# Merge citations into final papers dataset
python merge_citations_into_papers.py
```

## Monitoring Progress

Each wave workflow will report:
- Total papers to process
- Whether more waves are needed
- What wave number to run next

You can also check the artifacts section of each workflow run to see the citation batch files.

## Time Estimates

- **Per wave**: ~5-6 hours (15 parallel jobs, each processing 200 papers)
- **Total processing time**: ~20-24 hours (4 waves sequentially)
- **Final merge**: ~30-60 minutes

## Troubleshooting

**If a job fails in a wave:**
- The wave continues with other jobs (fail-fast: false)
- Re-run the same wave to retry failed jobs
- Artifacts are kept for 7 days

**If you need to start over:**
- Delete all artifacts with pattern `citations-wave-*`
- Start from wave 0

**If artifacts expire (after 7 days):**
- You'll need to re-run all waves that expired
- Consider running waves closer together to avoid expiration

## Alternative: Original Single-Wave Workflow

The original `daily_update.yml` workflow still exists but caps at 20 jobs, so it will only process 4,000 papers (jobs 0-19). It's kept for backward compatibility and smaller processing needs.

## Integration with Daily Updates

**Current setup:**
- `daily_update.yml` runs automatically on schedule (2 AM UTC daily)
- It processes models and datasets successfully
- Papers section caps at 20 parallel jobs (4,000 papers max)

**Recommended approach for full paper processing:**

1. **Weekly wave processing** (manual):
   - Run wave system (waves 0-3) once per week
   - Takes ~20-24 hours total (can be spread over a few days)
   - Processes all ~10,519 papers with citations

2. **Daily updates** (automatic):
   - Let `daily_update.yml` continue running for models and datasets
   - Consider disabling papers section in daily workflow to avoid conflicts

**To disable papers in daily workflow:**
You can comment out the papers jobs in `.github/workflows/daily_update.yml` if you prefer to only use the wave system for papers.

## Logging and Monitoring

Each paper fetch now logs detailed information:
- Paper ID and title being searched
- Citation count found
- Semantic Scholar ID
- Search errors (if any)
- Progress updates every 10 papers

Example log output:
```
[1/200] Paper ID: 2510.22236
🔍 Searching: Attention Is All You Need
   ✅ Found: 146,554 citations (ID: 204e3073870fae3d05bcbc2f6a8e263d9b72e776)

📊 Progress: 10/200 papers processed (9 citations found, 0.18 papers/sec, ETA: 17.5min)
```
