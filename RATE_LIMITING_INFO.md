# Semantic Scholar API Rate Limiting

Citation counts come from the Semantic Scholar API. That API is the slowest part of
the papers pipeline by a wide margin, so the pacing below is what determines how
long a daily run takes.

## The limit

**With our API key: 1 request per second, cumulative across all endpoints.**

"Cumulative" is the important word. The ceiling is *not* per process, per job, or
per endpoint — every request the pipeline makes anywhere counts against the same
1 req/s budget. Fifteen parallel jobs each sending 1 req/s is 15 req/s, and the
overwhelming majority of those get rejected.

(Without a key, requests fall back to a shared unauthenticated pool — roughly
100 requests per 5 minutes, shared with everyone else on the internet.)

The key is stored as the `SEMANTIC_SCHOLAR_KEY` repository secret and is sent as
the `x-api-key` header. That is the only name the code reads, in both the workflow
and locally.

## How the budget is divided

Because the ceiling is global and parallel jobs cannot coordinate, each job paces
itself to its *share* of the budget:

```
interval_per_job = (jobs_running_in_parallel / 1.0 req_per_sec) * 1.15 safety_factor
```

With the workflow's 15-way fan-out, that is **17.25s between requests in each
job**, for an expected aggregate of **0.87 req/s** — just under the ceiling.

- `config_papers.py` computes `SS_MIN_REQUEST_INTERVAL` from
  `CITATION_JOB_CONCURRENCY`, which the workflow sets to the wave's `max-parallel`.
  **If you change `max-parallel`, that env var must change with it** or the run
  will exceed the limit.
- `data_processor_papers.py` calls `throttle_semantic_scholar_request()` before
  every request, and reuses one authenticated client per process.

Throttling to the *expected* rate still allows occasional collisions, since two
jobs can fire in the same second. The `semanticscholar` package turns a 429 into a
30-second wait and retries up to 10 times, which absorbs those.

### A note on fan-out geometry

Under a global rate ceiling, parallelism no longer buys throughput. A wave's
duration depends only on how many papers are in it:

```
wave_duration ≈ papers_per_wave * 1.15 seconds
```

...whichever way those papers are split across jobs. 15 jobs × 200 papers and
5 jobs × 600 papers take the same wall-clock time; the second just uses a third of
the runner minutes. `PAPERS_PER_JOB` and `JOBS_PER_WAVE` in the workflow env are
the two knobs, and both `calculate_waves.py` and `generate_matrix_wave.py` take
them as parameters so they cannot drift apart.

## Timing

Per-request cost is ~17.25s of deliberate spacing. Papers whose citations were
fetched within the last 7 days are served from cache and cost nothing (see the
smart-caching logic in `fetch_citations_batch.py`).

| Scenario | Live fetches | Per wave | 6 waves total |
|---|---|---|---|
| Cold cache (worst case) | 200 per job | ~58 min | **~5.8 h** |
| Steady state (7-day cache) | ~29 per job | ~10 min | **~1 h** |

For reference, before the key and the snapshot fix, a run took ~8 hours and
covered only 12,000 of 16,985 papers, because live fetches were costing ~65s each
to rate-limit collisions.

## Local testing

```bash
export SEMANTIC_SCHOLAR_KEY=...          # otherwise the unauthenticated pool is used
export CITATION_JOB_CONCURRENCY=1        # running alone: use the whole budget

# Small batch. At concurrency=1 the spacing is 1.15s per request.
python fetch_citations_batch.py 0 10

python -c "import pandas as pd; df = pd.read_parquet('citations_batch_0_10.parquet'); print(df[['paper_id','citation_count','citation_fetch_date']].to_string())"
```

If a run-local `papers_raw.parquet` snapshot is present it is reused instead of
re-downloading from HuggingFace; see `fetch_papers_snapshot.py`.

## Alternative sources

If the 1 req/s ceiling ever becomes the binding constraint on coverage, these have
more generous limits:

- **OpenAlex** — most generous, has citation counts
- **Crossref** — academic citation database
