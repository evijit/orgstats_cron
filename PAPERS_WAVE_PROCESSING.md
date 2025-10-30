# Papers Wave Processing System

## Overview

The papers pipeline uses a **dynamic wave-based parallel processing system** to handle citation fetching for all papers efficiently within GitHub Actions constraints. The system automatically scales with paper count growth.

## Why Waves?

- **Rate Limiting**: Semantic Scholar API has rate limits (~30s per paper with package)
- **Time Constraints**: GitHub Actions has 6-hour timeout per job
- **Parallelism Limits**: Public repos have 20 concurrent jobs max
- **Data Size**: Currently ~10,556 papers (and growing)
- **Scalability**: System adapts to future paper count increases

## Dynamic Wave System Design

### How It Works

1. **Setup Phase** (`papers-setup` job):
   - Fetches paper data to count total papers
   - Calculates required waves: `num_waves = ceil(total_papers / 200 / 15)`
   - Stores `total_papers` and `num_waves` as outputs

2. **Wave Processing** (Waves 0-3, conditional):
   - Each wave has a setup job that generates job matrix
   - Fetch jobs run in parallel (up to 15 at once)
   - Wave N only runs if `num_waves >= N+1`
   - Each wave waits for previous wave to complete

3. **Merge Phase** (`papers-merge-and-process` job):
   - Downloads all citation artifacts from all waves
   - Merges citations with main paper data
   - Applies semantic taxonomy
   - Uploads to HuggingFace

### Configuration

- **Papers per job**: 200 (conservative for ~1.8-hour processing time)
- **Jobs per wave**: 15 (matches GitHub Actions public repo limit)
- **Number of waves**: **Calculated dynamically** based on paper count

### Current Scale (10,556 Papers)

```
Wave 0: Jobs  0-14 (Papers     0-3,000) - 15 jobs
Wave 1: Jobs 15-29 (Papers 3,000-6,000) - 15 jobs  
Wave 2: Jobs 30-44 (Papers 6,000-9,000) - 15 jobs
Wave 3: Jobs 45-52 (Papers 9,000-10,556) - 8 jobs
Total: 53 jobs across 4 waves (~1.8 hours per wave = 7 hours total)
```

### Future Scaling

| Papers | Jobs | Waves | Total Time* |
|--------|------|-------|-------------|
| 10,556 | 53   | 4     | ~7 hours    |
| 12,000 | 60   | 4     | ~7 hours    |
| 15,000 | 75   | 5     | **Needs Wave 4** |
| 20,000 | 100  | 7     | **Needs Waves 4-6** |

*Assumes ~1.8 hours per wave (sequential)

## Wave Limits & Adding New Waves

The workflow currently supports up to **4 waves** (Wave 0-3). Each wave handles up to 3,000 papers.

**Maximum with 4 waves**: ~12,000 papers

### When to Add More Waves

Monitor paper count growth. When approaching 12,000 papers (requiring 5+ waves), add new wave jobs to `.github/workflows/daily_update.yml`.

### How to Add Wave 4 (Template)

1. Copy this to the workflow file after Wave 3:

```yaml
  # Papers Wave 4
  papers-wave-4-setup:
    name: "Papers Wave 4 - Setup"
    needs: [papers-setup, papers-wave-3-fetch]
    if: |
      always() && 
      needs.papers-setup.outputs.num_waves >= 5 &&
      (needs.papers-wave-3-fetch.result == 'success' || needs.papers-wave-3-fetch.result == 'skipped')
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.split.outputs.matrix }}
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Calculate Wave 4 jobs
      id: split
      run: |
        PAPER_COUNT=${{ needs.papers-setup.outputs.total_papers }}
        MATRIX=$(python generate_matrix_wave.py $PAPER_COUNT 4 15)
        echo "Wave 4 Matrix: $MATRIX"
        echo "matrix=$MATRIX" >> $GITHUB_OUTPUT

  papers-wave-4-fetch:
    name: "Papers Wave 4 - Job ${{ matrix.job_id }}"
    needs: papers-wave-4-setup
    if: needs.papers-wave-4-setup.result == 'success'
    runs-on: ubuntu-latest
    timeout-minutes: 360
    strategy:
      matrix:
        include: ${{ fromJson(needs.papers-wave-4-setup.outputs.matrix) }}
      fail-fast: false
      max-parallel: 15
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Fetch citations for batch
      run: |
        python fetch_citations_batch.py ${{ matrix.start_idx }} ${{ matrix.end_idx }}
    - name: Upload batch artifact
      uses: actions/upload-artifact@v4
      with:
        name: citations-wave-4-batch-${{ matrix.job_id }}
        path: citations_batch_*.parquet
        retention-days: 7
```

2. Update `papers-merge-and-process` needs array:
```yaml
needs: [papers-setup, papers-wave-0-fetch, papers-wave-1-fetch, papers-wave-2-fetch, papers-wave-3-fetch, papers-wave-4-fetch]
```

3. Repeat for Wave 5, 6, etc. as needed

## Testing the Dynamic System

```bash
# Check current paper count and waves needed
python -c "from data_fetcher_papers import fetch_raw_data; df, _ = fetch_raw_data(); import subprocess; waves = subprocess.run(['python', 'calculate_waves.py', str(len(df)), '200', '15'], capture_output=True, text=True); print(f'Papers: {len(df):,}, Waves needed: {waves.stdout.strip()}')"

# Test wave generation for specific wave
python generate_matrix_wave.py 10556 0 15  # Wave 0

# Test all waves
for i in {0..3}; do echo "Wave $i:"; python generate_matrix_wave.py 10556 $i 15 | python -c "import sys, json; m=json.load(sys.stdin); print(f'  {len(m)} jobs: papers {m[0][\"start_idx\"]}-{m[-1][\"end_idx\"]}')"; done

# Run comprehensive validation
python3 << 'PYTEST'
import subprocess, json
papers = 10556
waves = int(subprocess.run(['python', 'calculate_waves.py', str(papers), '200', '15'], capture_output=True, text=True).stdout.strip())
all_covered = set()
for w in range(waves):
    matrix = json.loads(subprocess.run(['python', 'generate_matrix_wave.py', str(papers), str(w), '15'], capture_output=True, text=True).stdout)
    for job in matrix:
        all_covered.update(range(job['start_idx'], job['end_idx']))
print(f"✅ Coverage: {len(all_covered)}/{papers} papers" if len(all_covered) == papers else f"❌ Coverage issue")
PYTEST
```

## Performance Characteristics

- **Per paper**: ~30-35 seconds (semanticscholar package with timeout handling)
- **Per job** (200 papers): ~1.5-2 hours
- **Per wave** (15 jobs parallel): ~1.5-2 hours  
- **Total** (4 waves sequential): ~6-8 hours

