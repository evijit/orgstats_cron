#!/usr/bin/env python3
"""
Generate GitHub Actions matrix for parallel citation fetching.
Outputs JSON that can be consumed by matrix strategy.
"""

import sys
import json
import math

def generate_matrix(total_papers):
    """Generate job matrix for GitHub Actions."""
    # Conservative: 200 papers per job, optimized client should finish in 10-20 min
    papers_per_job = 200  # Safe batch size
    # Cap at 20 jobs (GitHub Actions concurrent limit for public repos)
    num_jobs = min(20, math.ceil(total_papers / papers_per_job))
    
    jobs = []
    for i in range(num_jobs):
        start = i * papers_per_job
        end = min(start + papers_per_job, total_papers)
        jobs.append({
            'job_id': i,
            'start_idx': start,
            'end_idx': end
        })
    
    return jobs

if __name__ == '__main__':
    total_papers = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    matrix = generate_matrix(total_papers)
    print(json.dumps(matrix))
