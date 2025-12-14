"""
Filter trajectory files to keep only human data (exclude Collector/Killer/Runner files).
Creates a new folder with only human trajectory files.
"""

import shutil
from pathlib import Path


def filter_human_trajectories(source_dir: Path, target_dir: Path):
    """
    Copy trajectory files that don't contain Collector/Killer/Runner in filename.
    
    Args:
        source_dir: Source directory containing trajectory files
        target_dir: Target directory to copy filtered files to
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        return
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Patterns to exclude
    exclude_patterns = ["Collector", "Killer", "Runner"]
    
    # Get all JSON files from source directory
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"[WARN] No JSON files found in {source_dir}")
        return
    
    print(f"[INFO] Found {len(json_files)} JSON files in {source_dir}")
    
    # Filter files
    human_files = []
    excluded_files = []
    
    for json_file in json_files:
        filename = json_file.name
        # Check if filename contains any exclude pattern (case-insensitive)
        should_exclude = any(
            pattern.lower() in filename.lower() 
            for pattern in exclude_patterns
        )
        
        if should_exclude:
            excluded_files.append(json_file)
        else:
            human_files.append(json_file)
    
    print(f"[INFO] Human files: {len(human_files)}")
    print(f"[INFO] Excluded files (Collector/Killer/Runner): {len(excluded_files)}")
    
    # Copy human files to target directory
    copied_count = 0
    for source_file in human_files:
        target_file = target_dir / source_file.name
        try:
            shutil.copy2(source_file, target_file)
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to copy {source_file.name}: {e}")
    
    print(f"\n[INFO] Successfully copied {copied_count} human trajectory files to {target_dir}")
    print(f"[INFO] Done!")
    
    # Print some examples
    if human_files:
        print(f"\n[INFO] Example human files:")
        for f in human_files[:5]:
            print(f"  - {f.name}")
        if len(human_files) > 5:
            print(f"  ... and {len(human_files) - 5} more")
    
    if excluded_files:
        print(f"\n[INFO] Example excluded files:")
        for f in excluded_files[:5]:
            print(f"  - {f.name}")
        if len(excluded_files) > 5:
            print(f"  ... and {len(excluded_files) - 5} more")


def main():
    """Main function."""
    source_dir = Path("data/player_trajectory_no_timeout")
    target_dir = Path("data/player_trajectory_no_timeout_human")
    
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"{'='*60}")
    
    filter_human_trajectories(source_dir, target_dir)


if __name__ == "__main__":
    main()

