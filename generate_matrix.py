#!/usr/bin/env python3
"""
Generate the GitHub Actions matrix for parallel citation fetching.

Emits one entry per batch, covering EVERY paper. There are no hand-declared waves:
the workflow's `max-parallel` decides how many batches run at once, so the sequence
of "waves" falls out of the scheduler and scales with the dataset automatically.

Batch size adapts to the paper count so the matrix stays within GitHub's limit as
the Hub grows. JSON goes to stdout; diagnostics go to stderr so the workflow can
capture the matrix cleanly.

Usage: python generate_matrix.py <total_papers> [--max-jobs N] [--papers-per-job N]
"""

import argparse
import json
import math
import sys

# GitHub allows at most 256 entries in a matrix. Stay clear of the ceiling.
MAX_MATRIX_JOBS = 200

# Smallest useful batch. Below this, per-job setup overhead (checkout, pip install,
# artifact download) starts to dominate the actual work.
MIN_PAPERS_PER_JOB = 200

# Largest batch that comfortably fits a job. At the 17.25s request spacing a 15-way
# fan-out implies, 1,000 papers is ~4.8h against the 6h ceiling.
MAX_PAPERS_PER_JOB = 1000


def choose_papers_per_job(total_papers, max_jobs=MAX_MATRIX_JOBS):
    """
    Pick a batch size that covers every paper without overflowing the matrix.

    Grows the batch only once the paper count would otherwise need more than
    max_jobs batches, so day-to-day runs keep a stable, small batch size.
    """
    papers_per_job = max(MIN_PAPERS_PER_JOB, math.ceil(total_papers / max_jobs))

    if papers_per_job > MAX_PAPERS_PER_JOB:
        # Coverage wins over speed: a slow job that might time out is recoverable
        # (the merge step keeps the previous day's citations), silently dropping the
        # tail of the dataset is not. Make the tradeoff loud.
        print(f"::warning::{total_papers:,} papers needs {papers_per_job:,} per job to fit "
              f"{max_jobs} jobs, above the {MAX_PAPERS_PER_JOB:,} that fits a 6h job. "
              f"Jobs may hit the timeout - raise max-parallel or split the run.",
              file=sys.stderr)

    return papers_per_job


def generate_matrix(total_papers, papers_per_job=None, max_jobs=MAX_MATRIX_JOBS):
    """Build the full batch list covering papers [0, total_papers)."""
    if total_papers < 1:
        raise ValueError(f"total_papers must be positive, got {total_papers}")

    if papers_per_job is None:
        papers_per_job = choose_papers_per_job(total_papers, max_jobs)

    jobs = []
    for job_id, start in enumerate(range(0, total_papers, papers_per_job)):
        jobs.append({
            'job_id': job_id,
            'start_idx': start,
            'end_idx': min(start + papers_per_job, total_papers),
        })

    return jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('total_papers', type=int)
    parser.add_argument('--max-jobs', type=int, default=MAX_MATRIX_JOBS,
                        help=f'matrix entry cap (default {MAX_MATRIX_JOBS})')
    parser.add_argument('--papers-per-job', type=int, default=None,
                        help='override the adaptive batch size')
    args = parser.parse_args()

    matrix = generate_matrix(args.total_papers, args.papers_per_job, args.max_jobs)

    covered = sum(job['end_idx'] - job['start_idx'] for job in matrix)
    if covered != args.total_papers:
        print(f"::error::Matrix covers {covered:,} of {args.total_papers:,} papers",
              file=sys.stderr)
        sys.exit(1)

    print(f"{len(matrix)} batches of {matrix[0]['end_idx'] - matrix[0]['start_idx']} "
          f"papers, covering all {args.total_papers:,}", file=sys.stderr)
    print(json.dumps(matrix))
