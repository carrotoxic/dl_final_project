"""
Visualize .rep replay files with full level rendering.
Creates images with traces drawn on complete level visuals (tiles, enemies, etc.).
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add funstudy to path to import original modules
FUNSTUDY_ROOT = Path(__file__).parent / 'funstudy' / 'EDRL-TAC-codes-v3'
sys.path.insert(0, str(FUNSTUDY_ROOT))

try:
    import jpype
    from jpype import JString, JInt, JBoolean
except ImportError:
    print("Error: JPype1 is required. Install with: pip install JPype1")
    sys.exit(1)

# Import original modules for level rendering
from src.smb.proxy import MarioSurveyProxy
from src.smb.level import MarioLevel
from src.utils.filesys import getpath


class ReplayVisualizer:
    """Replay visualizer using original MarioProxy and level rendering."""
    
    def __init__(self):
        """Initialize using original MarioSurveyProxy."""
        self.proxy = MarioSurveyProxy()


def find_level_file(rep_file: str, level_dir: Path) -> Optional[Path]:
    """Find corresponding level file from replay filename."""
    # Extract level identifier: *_lvl{num}.rep or *_{Agent}-{num}.rep
    patterns = ['lvl', 'Collector', 'Killer', 'Runner']
    p_ext = rep_file.find('.rep')
    
    if p_ext < 0:
        return None
    
    for pattern in patterns:
        p = rep_file.find(pattern)
        if p >= 0:
            level_name = rep_file[p:p_ext] + '.lvl'
            level_path = level_dir / level_name
            if level_path.exists():
                return level_path
    
    return None


def visualize_rep_file(rep_path: Path, level_dir: Path, output_dir: Path, 
                       visualizer: ReplayVisualizer) -> bool:
    """Visualize a single replay file with full level rendering."""
    rep_file = rep_path.name
    print(f"\nProcessing: {rep_file}")
    
    # Find level file
    level_path = find_level_file(rep_file, level_dir)
    if not level_path:
        print(f"  [WARN] Could not find level file for {rep_file}")
        return False
    
    print(f"  Level file: {level_path.name}")
    
    try:
        # Load level using original MarioLevel
        lvl = MarioLevel.from_file(str(level_path))
        
        # Replay and get results using original proxy
        gameres = visualizer.proxy.reproduce(
            str(level_path.absolute()),
            str(rep_path.absolute())
        )
        
        # Extract trace from results
        if 'trace' not in gameres or not gameres['trace']:
            print(f"  [WARN] No trace data found")
            return False
        
        trace = gameres['trace']
        
        # Create output image with full level rendering + trace
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{rep_path.stem}_trace.png"
        
        # Use original to_img_with_trace method for full level rendering
        lvl.to_img_with_trace(
            trace, 
            save_path=str(output_path.absolute()),
            color='black',
            lw=3
        )
        
        print(f"  [OK] Saved: {output_path.name}")
        print(f"       Trace length: {len(trace)}, Completion: {gameres.get('completing-ratio', 0):.2%}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Process all .rep files."""
    # Paths
    reps_dir = Path('data/reps')
    output_dir = Path('data/reps_visualized')
    
    # Find level directory
    possible_level_dirs = [
        FUNSTUDY_ROOT / 'exp_data' / 'survey data' / 'formal' / 'levels',
        Path('data/player_trajectory'),
    ]
    
    level_dir = None
    for dir_path in possible_level_dirs:
        if dir_path.exists():
            level_dir = dir_path
            break
    
    if level_dir is None:
        print("Error: Could not find level directory!")
        return
    
    if not reps_dir.exists():
        print(f"Error: {reps_dir} does not exist!")
        return
    
    # Get all .rep files
    rep_files = sorted(reps_dir.glob('*.rep'))
    if not rep_files:
        print(f"No .rep files found in {reps_dir}")
        return
    
    print(f"Found {len(rep_files)} replay files")
    print(f"Level directory: {level_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize visualizer
    try:
        visualizer = ReplayVisualizer()
    except Exception as e:
        print(f"Error initializing visualizer: {e}")
        print("\nTo fix this:")
        print("  1. Install Java JDK 8 or later from: https://adoptium.net/")
        print("  2. Or set JAVA_HOME environment variable:")
        print("     - Windows: $env:JAVA_HOME = 'C:\\Program Files\\Java\\jdk-XX'")
        print("     - Or set it permanently in System Environment Variables")
        print("  3. Restart your terminal after installing Java")
        return
    
    # Process files (with progress tracking)
    success_count = 0
    failed_count = 0
    
    for idx, rep_path in enumerate(rep_files, 1):
        if visualize_rep_file(rep_path, level_dir, output_dir, visualizer):
            success_count += 1
        else:
            failed_count += 1
        
        # Print progress every 50 files
        if idx % 50 == 0:
            print(f"\n[Progress] {idx}/{len(rep_files)} files processed ({success_count} succeeded, {failed_count} failed)")
    
    print(f"\n{'='*50}")
    print(f"Completed: {success_count}/{len(rep_files)} files succeeded")
    if failed_count > 0:
        print(f"Failed: {failed_count} files (may be due to corrupted replays or mismatched level files)")
    print(f"Images saved to: {output_dir}")


if __name__ == '__main__':
    main()
