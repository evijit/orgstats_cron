import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Union
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- CONFIGURATION ---
SOURCE_REPO_ID = "cfahlgren1/hub-stats"
DEST_REPO_ID = "hfmlsoc/hub_weekly_snapshots"
ITEMS = ['spaces', 'models', 'datasets', 'daily_papers']
# A temporary directory for the metadata-only clone of the source repo
SOURCE_CLONE_DIR = "temp_source_repo"

def setup_source_repo():
    """
    Clones the source repo without file content (blobless clone) if it doesn't exist,
    or pulls the latest changes if it does. This is very fast and uses minimal disk space.
    """
    print(f"Setting up metadata-only clone of {SOURCE_REPO_ID}...")
    if not os.path.exists(SOURCE_CLONE_DIR):
        print("  -> Cloning source repo (metadata only)...")
        subprocess.run([
            "git", "clone", "--filter=blob:none", "--no-checkout",
            f"https://huggingface.co/datasets/{SOURCE_REPO_ID}", SOURCE_CLONE_DIR
        ], check=True)
    else:
        print("  -> Pulling latest changes from source repo...")
        subprocess.run(["git", "-C", SOURCE_CLONE_DIR, "fetch", "--all"], check=True)
    print("  -> Source repo setup complete.")

def get_latest_snapshot_date(api: HfApi, item: str) -> Union[datetime.date, None]:
    """Inspects the destination repo to find the latest snapshot date for a given item."""
    print(f"Checking for latest existing snapshot in '{DEST_REPO_ID}' for '{item}'...")
    try:
        files = api.list_repo_files(repo_id=DEST_REPO_ID, repo_type="dataset")
        dates = [datetime.strptime(p.split('/')[1], '%Y-%m-%d').date() for p in files if p.startswith(f"{item}/") and len(p.split('/')) > 1]
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
    Uses git log on the metadata-only clone to find commits and then downloads snapshots.
    """
    print(f"\n--- Processing: {item} ---")
    dest_dir = output_base_dir / item
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = f"{item}.parquet"

    # 1. Get commit history using git log on our local, lightweight clone
    try:
        git_log_cmd = ["git", "-C", SOURCE_CLONE_DIR, "log", "main", "--pretty=format:%H|%ct", "--follow", "--", file_path]
        result = subprocess.run(git_log_cmd, capture_output=True, text=True, check=True)
        commits_info = result.stdout.strip().split('\n')
        if not commits_info or not commits_info[0]:
            print(f"No commit history found for {file_path}. Skipping.")
            return
        # Parse commits: (hexsha, timestamp)
        source_commits = [(line.split('|')[0], int(line.split('|')[1])) for line in commits_info]
        source_commits.sort(key=lambda x: x[1]) # Sort oldest to newest
    except subprocess.CalledProcessError as e:
        print(f"Could not get git log for {file_path}. Error: {e.stderr}")
        return

    # 2. Determine the date to start processing from
    latest_saved_date = get_latest_snapshot_date(api, item)
    start_date = (latest_saved_date + timedelta(weeks=1)) if latest_saved_date else source_commits[0][1] and datetime.fromtimestamp(source_commits[0][1]).date()
    end_date = datetime.now(timezone.utc).date()
    print(f"Checking for new snapshots from {start_date} to {end_date}")

    # 3. Loop through weeks and download missing snapshots
    current_date = start_date
    while current_date <= end_date:
        next_week_date = current_date + timedelta(weeks=1)
        next_week_ts = int(datetime.combine(next_week_date, datetime.min.time()).timestamp())
        
        # Find the latest commit for the current week
        weekly_commit_hex = None
        for hexsha, timestamp in reversed(source_commits):
            if timestamp < next_week_ts:
                weekly_commit_hex = hexsha
                break
        
        if weekly_commit_hex:
            print(f"New snapshot needed for week of {current_date}. Using commit {weekly_commit_hex[:8]}")
            weekly_folder = dest_dir / str(current_date)
            try:
                # CLEANED: Removed the deprecated 'local_dir_use_symlinks' parameter
                hf_hub_download(repo_id=SOURCE_REPO_ID, filename=file_path, repo_type="dataset",
                                revision=weekly_commit_hex, local_dir=weekly_folder)
                print(f"  -> Successfully downloaded snapshot to {weekly_folder}")
            except HfHubHTTPError as e:
                print(f"  -> ERROR downloading file from commit {weekly_commit_hex[:8]}: {e}")

        current_date = next_week_date

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <output_directory>")
        sys.exit(1)
    
    try:
        setup_source_repo()
        output_base_dir = Path(sys.argv[1])
        api = HfApi()
        for item in ITEMS:
            process_item(api, item, output_base_dir)
        print("\n--- All items processed successfully. ---")
    finally:
        # Clean up the temporary source repo clone
        if os.path.exists(SOURCE_CLONE_DIR):
            print(f"Cleaning up temporary directory: {SOURCE_CLONE_DIR}")
            shutil.rmtree(SOURCE_CLONE_DIR)

if __name__ == "__main__":
    main()