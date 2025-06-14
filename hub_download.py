import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
# CORRECTED: Import Union for backward-compatible type hints
from typing import Union
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- CONFIGURATION ---
SOURCE_REPO_ID = "cfahlgren1/hub-stats"
DEST_REPO_ID = "hfmlsoc/hub_weekly_snapshots"
ITEMS = ['spaces', 'models', 'datasets', 'daily_papers']

# CORRECTED: Changed 'datetime.date | None' to 'Union[datetime.date, None]'
def get_latest_snapshot_date(api: HfApi, item: str) -> Union[datetime.date, None]:
    """
    Inspects the destination repo to find the latest snapshot date for a given item
    by parsing the folder names (e.g., 'spaces/2023-10-16').
    """
    print(f"Checking for latest existing snapshot in '{DEST_REPO_ID}' for '{item}'...")
    try:
        files = api.list_repo_files(repo_id=DEST_REPO_ID, repo_type="dataset")
        dates = []
        for f_path in files:
            if f_path.startswith(f"{item}/"):
                parts = f_path.split('/')
                if len(parts) > 1:
                    try:
                        # parts[1] should be the date string 'YYYY-MM-DD'
                        snapshot_date = datetime.strptime(parts[1], '%Y-%m-%d').date()
                        dates.append(snapshot_date)
                    except ValueError:
                        continue # Ignore parts that aren't valid dates
        if dates:
            latest_date = max(dates)
            print(f"  -> Found latest snapshot date: {latest_date}")
            return latest_date
    except Exception as e:
        print(f"  -> Could not inspect destination repo, assuming it's empty. Error: {e}")
    
    print("  -> No existing snapshots found.")
    return None

def process_item(api: HfApi, item: str, output_base_dir: Path):
    """
    Fetches weekly snapshots for an item, starting after the last one that was saved.
    """
    print(f"\n--- Processing: {item} ---")
    dest_dir = output_base_dir / item
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = f"{item}.parquet"

    # 1. Get all commits for the source file
    try:
        source_commits = api.list_repo_commits(repo_id=SOURCE_REPO_ID, repo_type="dataset", path=file_path)
        if not source_commits:
            print(f"No commits found for file '{file_path}' in source repo. Skipping.")
            return
    except Exception as e:
        print(f"Could not fetch source commit history: {e}")
        return

    # 2. Determine the date to start processing from
    latest_saved_date = get_latest_snapshot_date(api, item)
    if latest_saved_date:
        start_date = latest_saved_date + timedelta(weeks=1)
    else:
        # If no snapshots exist, start from the date of the very first commit
        start_date = source_commits[-1].created_at.date()

    end_date = datetime.now(timezone.utc).date()
    print(f"Checking for new snapshots from {start_date} to {end_date}")

    # 3. Loop through weeks and download missing snapshots
    current_date = start_date
    while current_date <= end_date:
        next_week_date = current_date + timedelta(weeks=1)
        
        # Find the latest commit that occurred before the end of the current week
        weekly_commit = None
        for commit in source_commits: # Commits are already sorted newest to oldest
            if commit.created_at.date() < next_week_date:
                weekly_commit = commit
                break
        
        if weekly_commit:
            print(f"New snapshot needed for week of {current_date}. Using commit {weekly_commit.oid[:8]}")
            weekly_folder = dest_dir / str(current_date)
            
            try:
                hf_hub_download(
                    repo_id=SOURCE_REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    revision=weekly_commit.oid,
                    local_dir=weekly_folder,
                    local_dir_use_symlinks=False
                )
                print(f"  -> Successfully downloaded snapshot to {weekly_folder}")
            except HfHubHTTPError as e:
                print(f"  -> ERROR downloading file from commit {weekly_commit.oid[:8]}: {e}")

        current_date = next_week_date

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <output_directory>")
        sys.exit(1)

    output_base_dir = Path(sys.argv[1])
    api = HfApi()
    
    for item in ITEMS:
        process_item(api, item, output_base_dir)

    print("\n--- All items processed successfully. ---")

if __name__ == "__main__":
    main()