# Papers Citation Fan-Out

## Overview

Citation fetching is spread across parallel GitHub Actions jobs generated from the
**live paper count on every run**. Nothing about the fan-out is hand-declared, so it
keeps covering the whole dataset as the Hub grows.

This replaced a system of six hand-written "wave" job blocks in the workflow. That
design had a ceiling nobody was watching: when the dataset grew past what the
declared waves could hold, the extra papers were silently dropped. At 16,985 papers
against four declared waves, ~5,000 of the least-upvoted papers were going
unprocessed every night with no error anywhere.

## How it works

1. **`papers-setup`**
   - Runs `fetch_papers_snapshot.py`, which downloads the papers parquet from
     HuggingFace **exactly once per run** and uploads it as the
     `papers-raw-snapshot` artifact, along with the previous run's citations for the
     batch cache.
   - Runs `generate_matrix.py`, which emits one matrix entry per batch covering
     every paper, and exports it as a job output.

2. **`papers-fetch-citations`** — one job per matrix entry.
   - `max-parallel` caps how many run at once. **This is what creates the waves**:
     the scheduler starts the next batch as soon as a slot frees up, so there is no
     barrier waiting for a whole wave's slowest job (the old design lost ~30 minutes
     per wave to exactly that).
   - Each job downloads the shared snapshot rather than hitting HuggingFace itself.
   - `fail-fast: false`, so one bad batch doesn't cancel the rest.

3. **`papers-merge-and-process`**
   - Downloads every `citations-batch-*` artifact plus the snapshot, merges, applies
     the semantic taxonomy, uploads to HuggingFace.
   - Papers missing from this run keep their previous citation values, so partial
     failures degrade gracefully instead of losing data.

### Why the snapshot matters

Every job used to fetch the papers parquet from HuggingFace itself. With 60+ jobs
requesting the same file, HuggingFace returned HTTP 429, and whichever job lost the
race burned through its entire retry ladder and failed — the cause of most nightly
failures. One fetch, shared as an artifact, removes that class of failure.

## Configuration

There is **one** knob, `CITATION_CONCURRENCY` (workflow-level `env`):

```yaml
env:
  CITATION_CONCURRENCY: '15'
```

It feeds both `max-parallel` and each job's `CITATION_JOB_CONCURRENCY`, routed
through a `papers-setup` output because the `strategy` block can read the `needs`
context but not `env`. Both values coming from one place is deliberate — if they
disagreed, the run would quietly exceed the Semantic Scholar rate limit.

Raising it does **not** make the run faster. The Semantic Scholar ceiling is 1
request/second *cumulative across all jobs*, so more parallelism just means each job
paces itself more slowly. What it does change is per-job duration, which matters for
the 6-hour job timeout. Keep it under the 20 concurrent jobs public repos get,
leaving room for the models and datasets jobs.

Batch size is chosen by `generate_matrix.py` and needs no tuning: it stays at 200
papers until the paper count would need more than 200 batches, then grows to keep
the matrix under GitHub's 256-entry limit.

## Scaling

Verified end to end at each of these sizes — coverage is complete at every one:

| Papers | Batches | Per batch | Per job | Full cold pass |
|--------:|--------:|----------:|--------:|---------------:|
| 16,985 | 85 | 200 | ~1.0 h | ~5.4 h |
| 25,000 | 125 | 200 | ~1.0 h | ~8.0 h |
| 40,000 | 200 | 200 | ~1.0 h | ~12.8 h |
| 100,000 | 200 | 500 | ~2.4 h | ~31.9 h |
| 200,000 | 200 | 1,000 | ~4.8 h | ~63.9 h |

Past ~200,000 papers a batch no longer fits in a 6-hour job, and
`generate_matrix.py` emits a `::warning::` saying so. It still covers every paper
rather than dropping the tail — a slow job that might time out is recoverable, a
silently truncated dataset is not.

**The real ceiling is the API, not the workflow.** At 1 req/s, a cold pass over N
papers cannot take less than N seconds no matter how it's arranged. The 7-day
citation cache is what makes this sustainable: in steady state only ~1/7 of the
dataset needs refetching, so a daily run is roughly a seventh of the "full cold
pass" column. If that ever stops being enough, a source with a more generous limit
(OpenAlex, Crossref) is the fix — not more parallelism.

## Performance characteristics

Each live citation request costs ~17.25s of deliberate spacing at 15-way
concurrency. Papers fetched within the last 7 days come from cache at no cost.

| | Live fetches | Wall clock |
|---|---|---|
| Cold cache (first run) | ~16,985 | ~5.4 h |
| Steady state (7-day cache) | ~2,400 | ~45 min |

For reference, before the shared snapshot and the API key, runs took ~8 hours and
covered only 12,000 of 16,985 papers.

## Testing

```bash
# Paper count for the current dataset (reuses a local snapshot if present)
python -c "
from data_fetcher_papers import fetch_raw_data
df, _ = fetch_raw_data()
print(f'Papers: {len(df):,}')"

# The matrix the workflow would use (JSON on stdout, diagnostics on stderr)
python generate_matrix.py 16985

# Coverage across growth scenarios - what protects against the old silent-truncation bug
python3 - <<'PYTEST'
from generate_matrix import generate_matrix
for total in [16985, 25000, 40000, 100000, 200000]:
    m = generate_matrix(total)
    covered = sum(j['end_idx'] - j['start_idx'] for j in m)
    ppj = m[0]['end_idx'] - m[0]['start_idx']
    status = '✅' if covered == total and len(m) <= 256 else '❌'
    print(f"{status} {total:>7,} -> {len(m):>3} batches x {ppj:>4} = {covered:,} covered")
PYTEST
```

See [RATE_LIMITING_INFO.md](RATE_LIMITING_INFO.md) for the rate-limit arithmetic.
