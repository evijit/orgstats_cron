#!/usr/bin/env python3
"""
Fetch the papers dataset from HuggingFace exactly once per workflow run.

The wave jobs used to each pull the full parquet themselves, which meant 60+
concurrent requests for the same file and HTTP 429 responses that no amount of
retrying could get past. This script fetches it once; every other job in the run
consumes the resulting snapshot as a build artifact.

Writes:
  papers_raw.parquet       - the raw papers data
  papers_raw_meta.json     - real download timestamp + row count
  papers_count.txt         - row count only, for the workflow to read
  previous_citations.parquet - yesterday's citations (best effort, for caching)
"""

import os
import sys

import pandas as pd

from data_fetcher_papers import (SNAPSHOT_FILE, fetch_raw_data, validate_raw_data,
                                 write_snapshot)
from utils import log_progress

COUNT_FILE = os.environ.get('PAPERS_COUNT_FILE', 'papers_count.txt')
PREVIOUS_FILE = os.environ.get('PAPERS_PREVIOUS_FILE', 'previous_citations.parquet')
PREVIOUS_REPO = 'evijit/paperverse_daily_data'
PREVIOUS_FILENAME = 'papers_with_semantic_taxonomy.parquet'


def download_previous_citations():
    """
    Fetch the previous run's output once, so the wave jobs can reuse citations
    without each downloading it themselves. Best effort: the pipeline works fine
    without it, it just re-fetches citations that were already fresh.
    """
    try:
        from huggingface_hub import hf_hub_download

        log_progress("📥 Downloading previous citation data for the batch cache...")
        previous_path = hf_hub_download(
            repo_id=PREVIOUS_REPO,
            filename=PREVIOUS_FILENAME,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
        )
        df_previous = pd.read_parquet(previous_path)

        # Only the columns the citation cache needs — keeps the artifact small.
        cache_columns = [col for col in ('paper_id', 'citation_count', 'semantic_scholar_id',
                                         'citation_fetch_date') if col in df_previous.columns]
        df_previous[cache_columns].to_parquet(PREVIOUS_FILE, index=False)
        log_progress(f"   Saved {len(df_previous):,} previous records to {PREVIOUS_FILE}")
    except Exception as e:
        log_progress(f"   ⚠️  Could not load previous citation data (first run?): {e}")
        # Write an empty placeholder so the artifact contents are the same shape on
        # every run. An empty file means "no shared cache"; the batch jobs then fall
        # back to downloading the previous data themselves.
        pd.DataFrame(columns=['paper_id', 'citation_count', 'semantic_scholar_id',
                              'citation_fetch_date']).to_parquet(PREVIOUS_FILE, index=False)


def main():
    # use_snapshot=False: this script is the thing that creates the snapshot.
    df_raw, download_timestamp = fetch_raw_data(use_snapshot=False)
    validate_raw_data(df_raw)
    write_snapshot(df_raw, download_timestamp)

    with open(COUNT_FILE, 'w') as count_file:
        count_file.write(f"{len(df_raw)}\n")
    log_progress(f"🔢 Wrote paper count ({len(df_raw):,}) to {COUNT_FILE}")

    download_previous_citations()

    if not os.path.exists(SNAPSHOT_FILE):
        log_progress(f"❌ Snapshot file {SNAPSHOT_FILE} was not created")
        return 1

    log_progress("✅ Papers snapshot ready for all downstream jobs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
