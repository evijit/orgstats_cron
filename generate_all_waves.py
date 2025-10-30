#!/usr/bin/env python3
"""
Generate complete matrix for all papers across all waves.
Returns a list of all jobs with wave information.
"""

import sys
import json
import math

def generate_all_waves_matrix(total_papers, papers_per_job=200, jobs_per_wave=15):
    """
    Generate job matrix for ALL waves.
    
    Args:
        total_papers: Total number of papers to process
        papers_per_job: Papers per job (default: 200)
        jobs_per_wave: Jobs per wave (default: 15)
        
    Returns:
        List of job configs with wave information
    """
    total_jobs = math.ceil(total_papers / papers_per_job)
    num_waves = math.ceil(total_jobs / jobs_per_wave)
    
    all_jobs = []
    
    for wave_num in range(num_waves):
        wave_start_job = wave_num * jobs_per_wave
        wave_end_job = min(wave_start_job + jobs_per_wave, total_jobs)
        
        for job_id in range(wave_start_job, wave_end_job):
            start_idx = job_id * papers_per_job
            end_idx = min(start_idx + papers_per_job, total_papers)
            
            all_jobs.append({
                'wave': wave_num,
                'job_id': job_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'wave_job_id': job_id - wave_start_job  # Position within wave
            })
    
    return all_jobs

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_all_waves.py <total_papers> [papers_per_job] [jobs_per_wave]")
        sys.exit(1)
    
    total_papers = int(sys.argv[1])
    papers_per_job = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    jobs_per_wave = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    matrix = generate_all_waves_matrix(total_papers, papers_per_job, jobs_per_wave)
    
    # Print summary to stderr for logging
    num_waves = max(job['wave'] for job in matrix) + 1 if matrix else 0
    print(f"Total papers: {total_papers}", file=sys.stderr)
    print(f"Total jobs: {len(matrix)}", file=sys.stderr)
    print(f"Number of waves: {num_waves}", file=sys.stderr)
    print(f"Jobs per wave: {jobs_per_wave}", file=sys.stderr)
    
    # Print JSON matrix to stdout
    print(json.dumps(matrix))
