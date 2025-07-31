import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Union, List, Tuple
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
        print(f"  -> Cloning source repo (metadata only) from {SOURCE_REPO_ID}...")
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

def get_source_commits(item: str) -> List[Tuple[str, int]]:
    """
    Uses git log on the metadata-only clone to get commit history for a specific file.
    """
    file_path = f"{item}.parquet"
    try:
        git_log_cmd = ["git", "-C", SOURCE_CLONE_DIR, "log", "main", "--pretty=format:%H|%ct", "--follow", "--", file_path]
        result = subprocess.run(git_log_cmd, capture_output=True, text=True, check=True)
        commits_info = result.stdout.strip().split('\n')
        if not commits_info or not commits_info[0]:
            print(f"No commit history found for {file_path}. Skipping.")
            return []
        # Parse commits: (hexsha, timestamp)
        source_commits = [(line.split('|')[0], int(line.split('|')[1])) for line in commits_info]
        source_commits.sort(key=lambda x: x[1])  # Sort oldest to newest
        return source_commits
    except subprocess.CalledProcessError as e:
        print(f"Could not get git log for {file_path}. Error: {e.stderr}")
        return []

def process_item(api: HfApi, item: str, output_base_dir: Path, backfill_start_date: str = None, backfill_end_date: str = None):
    """
    Finds and downloads snapshots for a given item, either from the last saved point
    or within a specified backfill date range.
    """
    print(f"\n--- Processing: {item} ---")
    dest_dir = output_base_dir / item
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = f"{item}.parquet"

    source_commits = get_source_commits(item)
    if not source_commits:
        return

    if backfill_start_date and backfill_end_date:
        start_date = datetime.strptime(backfill_start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(backfill_end_date, '%Y-%m-%d').date()
        print(f"Backfilling from {start_date} to {end_date}")
    else:
        latest_saved_date = get_latest_snapshot_date(api, item)
        start_date = (latest_saved_date + timedelta(weeks=1)) if latest_saved_date else datetime.fromtimestamp(source_commits[0][1], timezone.utc).date()
        end_date = datetime.now(timezone.utc).date()
        print(f"Checking for new snapshots from {start_date} to {end_date}")

    current_date = start_date
    while current_date <= end_date:
        next_week_date = current_date + timedelta(weeks=1)
        next_week_ts = int(datetime.combine(next_week_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())

        # Find the latest commit for the current week
        weekly_commit_hex = None
        for hexsha, timestamp in reversed(source_commits):
            if timestamp < next_week_ts:
                weekly_commit_hex = hexsha
                break

        if weekly_commit_hex:
            print(f"Snapshot identified for week of {current_date}. Using commit {weekly_commit_hex[:8]}")
            weekly_folder = dest_dir / str(current_date)
            # Clean up existing directory if backfilling
            if backfill_start_date and os.path.exists(weekly_folder):
                print(f"  -> Removing existing directory for backfill: {weekly_folder}")
                shutil.rmtree(weekly_folder)

            try:
                hf_hub_download(
                    repo_id=SOURCE_REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    revision=weekly_commit_hex,
                    local_dir=weekly_folder
                )
                print(f"  -> Successfully downloaded snapshot to {weekly_folder}")
            except HfHubHTTPError as e:
                print(f"  -> ERROR downloading file from commit {weekly_commit_hex[:8]}: {e}")

        current_date = next_week_date

def main():
    parser = argparse.ArgumentParser(description="Download weekly snapshots from Hugging Face Hub.")
    parser.add_argument("output_directory", help="The directory to save the snapshots.")
    parser.add_argument("--backfill-start-date", help="The start date for backfilling (YYYY-MM-DD).", default=None)
    parser.add_argument("--backfill-end-date", help="The end date for backfilling (YYYY-MM-DD).", default=None)
    args = parser.parse_args()

    if (args.backfill_start_date and not args.backfill_end_date) or (not args.backfill_start_date and args.backfill_end_date):
        print("Error: Both --backfill-start-date and --backfill-end-date must be provided for backfilling.")
        sys.exit(1)

    try:
        setup_source_repo()
        output_base_dir = Path(args.output_directory)
        api = HfApi()
        for item in ITEMS:
            process_item(api, item, output_base_dir, args.backfill_start_date, args.backfill_end_date)
        print("\n--- All items processed successfully. ---")
    finally:
        # Clean up the temporary source repo clone
        if os.path.exists(SOURCE_CLONE_DIR):
            print(f"Cleaning up temporary directory: {SOURCE_CLONE_DIR}")
            shutil.rmtree(SOURCE_CLONE_DIR)

if __name__ == "__main__":
    main()