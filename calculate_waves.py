#!/usr/bin/env python3
"""
Calculate the number of waves needed for papers processing.
"""

import sys
import math

def calculate_waves(total_papers, papers_per_job=200, jobs_per_wave=15):
    """
    Calculate how many waves are needed to process all papers.
    
    Args:
        total_papers: Total number of papers to process
        papers_per_job: Papers per job (default: 200)
        jobs_per_wave: Jobs per wave (default: 15)
        
    Returns:
        Number of waves needed
    """
    total_jobs = math.ceil(total_papers / papers_per_job)
    num_waves = math.ceil(total_jobs / jobs_per_wave)
    return num_waves

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python calculate_waves.py <total_papers> [papers_per_job] [jobs_per_wave]")
        sys.exit(1)
    
    total_papers = int(sys.argv[1])
    papers_per_job = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    jobs_per_wave = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    
    num_waves = calculate_waves(total_papers, papers_per_job, jobs_per_wave)
    print(num_waves)
