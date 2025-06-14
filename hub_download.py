import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# --- CONFIGURATION ---
# The source repository we are getting data FROM
SOURCE_REPO_ID = "cfahlgren1/hub-stats"
# List of repository types to process
ITEMS = ['spaces', 'models', 'datasets', 'daily_papers']

def process_item(item, output_base_dir):
    """
    Fetches the commit history for an item and downloads only the necessary
    weekly snapshots that don't already exist.
    """
    print(f"\n--- Processing: {item} ---")
    
    api = HfApi()
    file_path = f"{item}.parquet"
    dest_dir = output_base_dir / item
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch commit history (metadata only, very fast)
    try:
        print(f"Fetching commit history for {SOURCE_REPO_ID}...")
        all_commits = api.list_commits(repo_id=SOURCE_REPO_ID, repo_type="dataset")
        # Filter for commits that actually touched our file of interest
        commits = [c for c in all_commits if file_path in c.scope]
        if not commits:
            print(f"No commits found for file '{file_path}'. Skipping item.")
            return
    except Exception as e:
        print(f"Could not fetch commit history: {e}")
        return

    # Sort commits by date, oldest first
    commits.sort(key=lambda c: c.created_at)
    
    # 2. Determine date range for snapshots
    start_date = commits[0].created_at.date()
    end_date = datetime.now(timezone.utc).date()
    print(f"Checking for snapshots from {start_date} to {end_date}")

    current_date = start_date
    while current_date <= end_date:
        next_week_date = current_date + timedelta(weeks=1)
        
        # 3. Check if this week's snapshot already exists
        weekly_folder = dest_dir / str(current_date)
        if weekly_folder.exists():
            # print(f"Snapshot for {current_date} already exists. Skipping.")
            current_date = next_week_date
            continue

        # 4. Find the latest commit for the current week
        weekly_commit = None
        for commit in reversed(commits):
            if commit.created_at.date() < next_week_date:
                weekly_commit = commit
                break
        
        if weekly_commit:
            print(f"New snapshot needed for week of {current_date}. Using commit {weekly_commit.oid[:8]}")
            
            # 5. Download ONLY the required file from that specific commit
            try:
                hf_hub_download(
                    repo_id=SOURCE_REPO_ID,
                    filename=file_path,
                    repo_type="dataset",
                    revision=weekly_commit.oid,
                    local_dir=weekly_folder,
                    local_dir_use_symlinks=False # Ensures file is copied, not symlinked
                )
                # hf_hub_download places the file in local_dir, but we might want to rename it
                # In this case, it saves it as "models.parquet" inside the date folder, which is perfect.
                print(f"  -> Successfully downloaded snapshot to {weekly_folder}")
            except HfHubHTTPError as e:
                # This can happen if the file didn't exist in that specific commit
                print(f"  -> ERROR: Could not download file from commit {weekly_commit.oid[:8]}. It may not have existed yet. {e}")
                # No need to cleanup, as hf_hub_download won't create the folder on failure
        
        current_date = next_week_date

def main():
    if len(sys.argv) < 2:
        print("Error: Please provide the target output directory as an argument.")
        print(f"Usage: python {sys.argv[0]} <output_directory>")
        sys.exit(1)

    output_base_dir = Path(sys.argv[1])
    print(f"Output will be saved in: {output_base_dir}")

    for item in ITEMS:
        process_item(item, output_base_dir)

    print("\n--- All items processed successfully. ---")

if __name__ == "__main__":
    main()