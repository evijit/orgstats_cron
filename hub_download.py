import os
import subprocess
import datetime
from pathlib import Path
import shutil
import sys

# --- CONFIGURATION ---
# List of repository types to process
ITEMS = ['spaces', 'models', 'datasets', 'daily_papers']
# The source repository containing the parquet files
SOURCE_REPO_URL = "https://huggingface.co/datasets/cfahlgren1/hub-stats"
# A temporary directory to clone the source repository into
CLONE_DIR = "temp_source_repo"

def run_command(command, cwd=None):
    """Runs a command and raises an exception if it fails."""
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd)
    return result

def setup_repo():
    """Clones or updates the source repository."""
    if not os.path.exists(CLONE_DIR):
        print(f"Cloning repository with Git LFS: {SOURCE_REPO_URL}")
        run_command(["git", "clone", "--depth", "1", SOURCE_REPO_URL, CLONE_DIR])
        # We need the full history for 'git log', so unshallow the clone
        run_command(["git", "fetch", "--unshallow"], cwd=CLONE_DIR)
    else:
        print(f"Repository already exists at {CLONE_DIR}, pulling latest changes.")
        run_command(["git", "pull"], cwd=CLONE_DIR)

def process_item_history(item, output_base_dir):
    """
    Processes the commit history for a specific item (e.g., 'models') and
    creates weekly snapshots.
    """
    file_path = f"{item}.parquet"
    dest_dir = output_base_dir / item
    print(f"\n--- Processing: {item} ---")
    print(f"Destination directory: {dest_dir}")

    # Get all commits that affected the file
    print(f"Finding commits for file: {file_path}")
    try:
        git_log_cmd = ["git", "log", "--pretty=format:%H|%ct", "--follow", file_path]
        result = run_command(git_log_cmd, cwd=CLONE_DIR)
        commits_info = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Could not get git log for {file_path}. It might not exist in the repo. Skipping.")
        print(f"Error: {e.stderr}")
        return

    if not commits_info or commits_info[0] == '':
        print(f"No commits found for {file_path}. Skipping.")
        return

    commits = [(hexsha, int(ts)) for hexsha, ts in (line.split('|') for line in commits_info if '|' in line)]
    commits.sort(key=lambda x: x[1]) # Sort by timestamp, oldest first

    start_date = datetime.datetime.fromtimestamp(commits[0][1]).date()
    end_date = datetime.date.today()
    
    dest_dir.mkdir(parents=True, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        next_week_date = current_date + datetime.timedelta(weeks=1)
        next_week_timestamp = int(datetime.datetime.combine(next_week_date, datetime.time.min).timestamp())

        # Find the latest commit before the end of the current week
        weekly_commit_hex = None
        for hexsha, timestamp in reversed(commits):
            if timestamp < next_week_timestamp:
                weekly_commit_hex = hexsha
                break

        if weekly_commit_hex:
            weekly_folder = dest_dir / str(current_date)
            # Only process if the folder doesn't already exist in our destination
            if weekly_folder.exists():
                # print(f"Snapshot for {current_date} already exists. Skipping.")
                current_date = next_week_date
                continue

            print(f"Processing week of {current_date} using commit {weekly_commit_hex[:8]}")
            weekly_folder.mkdir(parents=True, exist_ok=True)
            output_path = weekly_folder / f"{item}.parquet"
            
            try:
                # Use git show to extract the file from a specific commit without checking out
                # This is much faster than doing a full checkout for each week
                git_show_cmd = ["git", "show", f"{weekly_commit_hex}:{file_path}"]
                with open(output_path, 'wb') as f:
                    subprocess.run(git_show_cmd, stdout=f, check=True, cwd=CLONE_DIR)

                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  -> Saved: {output_path} ({file_size_mb:.2f} MB)")

            except subprocess.CalledProcessError as e:
                print(f"  -> ERROR: Could not extract file from commit {weekly_commit_hex}. It may not have existed yet.")
                shutil.rmtree(weekly_folder) # Clean up empty folder on error

        current_date = next_week_date


def main():
    """Main function to run the snapshot generation."""
    if len(sys.argv) < 2:
        print("Error: Please provide the target output directory as an argument.")
        print(f"Usage: python {sys.argv[0]} <output_directory>")
        sys.exit(1)

    output_base_dir = Path(sys.argv[1])
    print(f"Output will be saved in: {output_base_dir}")

    try:
        run_command(["git", "lfs", "install"])
        setup_repo()
        for item in ITEMS:
            process_item_history(item, output_base_dir)
        print("\n--- All items processed successfully. ---")
    except subprocess.CalledProcessError as e:
        print("\n--- A critical command failed. ---")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    finally:
        if os.path.exists(CLONE_DIR):
            print(f"Cleaning up temporary directory: {CLONE_DIR}")
            shutil.rmtree(CLONE_DIR)

if __name__ == "__main__":
    main()