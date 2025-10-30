#!/usr/bin/env python3
"""
Split papers processing into multiple parallel jobs for citation fetching.
This script determines how many jobs are needed and generates job parameters.
"""

import os
import sys
import math

# Configuration
CITATION_RATE_LIMIT_DELAY = 3  # seconds per request
MAX_JOB_TIME_MINUTES = 300  # 5 hours (leaving 1 hour buffer in 6hr limit)
PAPERS_PER_JOB = int(MAX_JOB_TIME_MINUTES * 60 / CITATION_RATE_LIMIT_DELAY)

def calculate_job_split(total_papers):
    """
    Calculate how many parallel jobs are needed.
    
    Args:
        total_papers: Total number of papers to process
        
    Returns:
        dict with job configuration
    """
    num_jobs = math.ceil(total_papers / PAPERS_PER_JOB)
    
    return {
        'total_papers': total_papers,
        'papers_per_job': PAPERS_PER_JOB,
        'num_jobs': num_jobs,
        'estimated_time_per_job_minutes': (PAPERS_PER_JOB * CITATION_RATE_LIMIT_DELAY) / 60
    }

def generate_job_matrix(total_papers):
    """
    Generate GitHub Actions matrix configuration.
    
    Returns:
        List of job configurations
    """
    config = calculate_job_split(total_papers)
    num_jobs = config['num_jobs']
    papers_per_job = config['papers_per_job']
    
    jobs = []
    for job_id in range(num_jobs):
        start_idx = job_id * papers_per_job
        end_idx = min(start_idx + papers_per_job, total_papers)
        
        jobs.append({
            'job_id': job_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'count': end_idx - start_idx
        })
    
    return jobs

if __name__ == '__main__':
    # Get total papers from command line or environment
    total_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    
    config = calculate_job_split(total_papers)
    
    print(f"📊 Citation Fetching Job Split Configuration")
    print(f"=" * 70)
    print(f"Total papers: {config['total_papers']:,}")
    print(f"Papers per job: {config['papers_per_job']:,}")
    print(f"Number of parallel jobs: {config['num_jobs']}")
    print(f"Estimated time per job: {config['estimated_time_per_job_minutes']:.1f} minutes")
    print(f"=" * 70)
    
    jobs = generate_job_matrix(total_papers)
    
    print(f"\nJob Matrix:")
    for job in jobs:
        print(f"  Job {job['job_id']}: Papers {job['start_idx']:,} - {job['end_idx']:,} ({job['count']:,} papers)")
    
    # Output for GitHub Actions
    if os.getenv('GITHUB_ACTIONS'):
        import json
        matrix_json = json.dumps({'include': jobs})
        print(f"\n::set-output name=matrix::{matrix_json}")
        print(f"::set-output name=num_jobs::{config['num_jobs']}")
