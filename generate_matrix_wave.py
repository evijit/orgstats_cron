#!/usr/bin/env python3
"""
Generate GitHub Actions matrix for a specific wave of parallel citation fetching.
"""

import sys
import json
import math

def generate_matrix_wave(total_papers, wave_number, jobs_per_wave):
    """
    Generate job matrix for a specific wave.
    
    Args:
        total_papers: Total number of papers to process
        wave_number: Which wave (0-based)
        jobs_per_wave: Number of jobs per wave
        
    Returns:
        JSON array of job configs for this wave
    """
    papers_per_job = 1000  # ~30-45 min per job (reused client, fast responses)
    total_jobs = math.ceil(total_papers / papers_per_job)
    
    # Calculate which jobs belong to this wave
    wave_start_job = wave_number * jobs_per_wave
    wave_end_job = min(wave_start_job + jobs_per_wave, total_jobs)
    
    jobs = []
    for job_id in range(wave_start_job, wave_end_job):
        start_idx = job_id * papers_per_job
        end_idx = min(start_idx + papers_per_job, total_papers)
        jobs.append({
            'job_id': job_id,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    return jobs

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python generate_matrix_wave.py <total_papers> <wave_number> <jobs_per_wave>")
        sys.exit(1)
    
    total_papers = int(sys.argv[1])
    wave_number = int(sys.argv[2])
    jobs_per_wave = int(sys.argv[3])
    
    matrix = generate_matrix_wave(total_papers, wave_number, jobs_per_wave)
    print(json.dumps(matrix))
