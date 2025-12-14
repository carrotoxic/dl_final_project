"""
Generate state trajectories with (j, k, r, c, d, e) tuples for each time step.

BEST APPROACH: Use original JSON files for accurate action encoding, combined with
replay files for game statistics (kills, coins, deaths).

State tuple definition:
- j: number of jumps (cumulative)
- k: number of enemies killed (cumulative)
- r: number of times player started running (cumulative)
- c: number of coins collected (cumulative)
- d: number of times player died (cumulative)
- e: unique encoding for each action

This version uses original JSON survey files which contain the 'actions0' field
with numeric action encoding, providing accurate action data.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add funstudy to path
FUNSTUDY_ROOT = Path(__file__).parent / 'funstudy' / 'EDRL-TAC-codes-v3'
sys.path.insert(0, str(FUNSTUDY_ROOT))

from src.smb.proxy import MarioSurveyProxy
from src.utils.filesys import getpath
import jpype
from jpype import JString


def decode_action_code(action_value: int) -> Dict[str, bool]:
    """
    Decode numeric action encoding to button states.
    
    Based on Mario AI Framework action encoding:
    - Bit 0 (1): LEFT
    - Bit 1 (2): RIGHT
    - Bit 2 (4): DOWN
    - Bit 3 (8): JUMP
    - Bit 4 (16): SPEED/RUN
    
    The numeric value is the sum of pressed button flags.
    
    Returns:
        Dictionary with button states
    """
    return {
        'LEFT': bool(action_value & 0x01),
        'RIGHT': bool(action_value & 0x02),
        'DOWN': bool(action_value & 0x04),
        'JUMP': bool(action_value & 0x08),
        'SPEED': bool(action_value & 0x10),
        'raw': action_value
    }


def extract_from_json(json_path: Path, replay_stats: Optional[Dict] = None) -> Tuple[List[Tuple[int, int, int, int, int, int]], Dict]:
    """
    Extract state trajectory from original JSON survey file.
    
    Args:
        json_path: Path to original survey JSON file
        replay_stats: Optional dict with final stats from replay (kills, coins, lives, etc.)
                     If None, will use interpolation for kills/coins
    
    Returns:
        Tuple of (state_trajectory, metadata)
    """
    with json_path.open('r') as f:
        data = json.load(f)
    
    elements = data['elementData1']
    
    # Track cumulative statistics
    cumulative_jumps = 0
    cumulative_runs = 0
    cumulative_deaths = 0
    prev_jump_state = False
    prev_run_state = False
    prev_x = None
    
    state_trajectory = []
    
    # Extract replay stats if available
    total_kills = replay_stats.get('#kills', 0) if replay_stats else 0
    total_coins = replay_stats.get('#coins', 0) if replay_stats else 0
    final_lives = replay_stats.get('lives', 3) if replay_stats else 3
    initial_lives = 3  # Default starting lives
    
    # Track positions for interpolating kills/coins based on progress
    positions = []
    
    # First pass: collect all positions
    valid_indices = []
    for i in range(1, len(elements)):
        if elements[i] is None:
            break
        if 'marioX1' not in elements[i] or 'actions0' not in elements[i]:
            continue
        positions.append(elements[i]['marioX1'])
        valid_indices.append(i)
    
    if not positions:
        return [], {'error': 'No valid position data found'}
    
    max_x = max(positions)
    min_x = min(positions)
    x_range = max_x - min_x if max_x > min_x else 1
    
    # Second pass: extract states
    for idx, i in enumerate(valid_indices):
        elem = elements[i]
        x = float(elem['marioX1'])
        y = float(elem['marioY2'])
        time_step = elem.get('time5', idx + 1)
        
        # Decode action
        action_value = elem['actions0'].get('0', 0)  # Get main action encoding
        action_decoded = decode_action_code(action_value)
        
        # Detect jumps (transition from not-jumping to jumping)
        current_jump = action_decoded['JUMP']
        if current_jump and not prev_jump_state:
            cumulative_jumps += 1
        prev_jump_state = current_jump
        
        # Detect run starts (transition from not-running to running)
        current_run = action_decoded['SPEED']
        if current_run and not prev_run_state:
            cumulative_runs += 1
        prev_run_state = current_run
        
        # Detect deaths (position resets)
        if prev_x is not None and x < prev_x - 100:
            cumulative_deaths += 1
        prev_x = x
        
        # Interpolate kills and coins based on position progress
        # More accurate than time-based interpolation since position reflects actual progress
        if x_range > 0 and idx < len(positions):
            # Use position-based progress
            progress = (x - min_x) / x_range
            progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
        else:
            # Fallback to time-based
            progress = (idx + 1) / len(valid_indices) if len(valid_indices) > 1 else 1.0
        
        current_kills = int(total_kills * progress)
        current_coins = int(total_coins * progress)
        
        # Create state tuple: (j, k, r, c, d, e)
        state = (
            cumulative_jumps,      # j - cumulative jumps
            current_kills,         # k - cumulative kills at this step
            cumulative_runs,       # r - cumulative run starts
            current_coins,         # c - cumulative coins at this step
            cumulative_deaths,     # d - cumulative deaths
            action_value           # e - action encoding (raw numeric value)
        )
        state_trajectory.append(state)
    
    # Ensure final values match totals
    if state_trajectory and replay_stats:
        final_state = list(state_trajectory[-1])
        final_state[1] = total_kills  # k
        final_state[3] = total_coins  # c
        state_trajectory[-1] = tuple(final_state)
    
    metadata = {
        'source': 'json_survey_file',
        'json_path': str(json_path),
        'n_steps': len(state_trajectory),
        'total_kills': total_kills,
        'total_coins': total_coins,
        'initial_lives': initial_lives,
        'final_lives': final_lives,
        'estimated_deaths': cumulative_deaths,
    }
    
    return state_trajectory, metadata


def find_replay_file(json_name: str, reps_dir: Path) -> Optional[Path]:
    """Find corresponding .rep file for a JSON file."""
    # JSON name format: <uuid>_<level>.json
    # REP name format: <uuid>_<level>.rep
    rep_name = json_name.replace('.json', '.rep')
    rep_path = reps_dir / rep_name
    
    if rep_path.exists():
        return rep_path
    return None


def find_level_file(json_name: str, levels_dir: Path) -> Optional[Path]:
    """Find corresponding .lvl file for a JSON file."""
    # Extract level name from JSON filename
    # Format: <uuid>_lvl<num>.json or <uuid>_Collector-<num>.json etc.
    parts = json_name.replace('.json', '').split('_', 1)
    if len(parts) < 2:
        return None
    
    level_name = parts[1]  # e.g., "lvl194" or "Collector-48"
    
    # Try direct match
    lvl_path = levels_dir / f"{level_name}.lvl"
    if lvl_path.exists():
        return lvl_path
    
    return None


def get_replay_stats(proxy: MarioSurveyProxy, lvl_path: Path, rep_path: Path) -> Optional[Dict]:
    """Get game statistics from replay file."""
    try:
        result = proxy.reproduce(
            str(lvl_path.absolute()),
            str(rep_path.absolute()),
            get_timestep_stats=False
        )
        return result
    except Exception as e:
        print(f"[WARN] Could not get replay stats: {e}")
        return None


def save_state_trajectory(state_traj: List[Tuple[int, int, int, int, int, int]], 
                         output_path: Path, 
                         metadata: dict = None):
    """Save state trajectory to JSON file."""
    result = {
        'state_trajectory': [[j, k, r, c, d, e] for j, k, r, c, d, e in state_traj],
        'metadata': metadata or {}
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[INFO] Saved state trajectory to {output_path}")


def main():
    """Main function to extract state trajectories."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract accurate state trajectories from JSON survey files")
    parser.add_argument("--json_path", type=str, required=True, help="Path to original survey JSON file")
    parser.add_argument("--reps_dir", type=str, default="data/reps", help="Directory containing .rep files (optional, for stats)")
    parser.add_argument("--levels_dir", type=str, default="funstudy/EDRL-TAC-codes-v3/exp_data/survey data/formal/levels", help="Directory containing .lvl files")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[ERROR] JSON file not found: {json_path}")
        return
    
    print(f"Processing JSON file: {json_path.name}")
    
    # Try to get replay stats for accurate kills/coins
    replay_stats = None
    proxy = None
    
    reps_dir = Path(args.reps_dir)
    levels_dir = Path(args.levels_dir)
    
    rep_path = find_replay_file(json_path.name, reps_dir)
    lvl_path = find_level_file(json_path.name, levels_dir)
    
    if rep_path and lvl_path:
        print(f"  Found replay file: {rep_path.name}")
        print(f"  Found level file: {lvl_path.name}")
        
        try:
            proxy = MarioSurveyProxy()
            replay_stats = get_replay_stats(proxy, lvl_path, rep_path)
            if replay_stats:
                print(f"  Got replay stats: {replay_stats.get('#kills', 0)} kills, {replay_stats.get('#coins', 0)} coins")
        except Exception as e:
            print(f"  [WARN] Could not load replay stats: {e}")
    else:
        print(f"  [INFO] Replay file or level file not found, will use interpolation")
    
    # Extract state trajectory
    state_traj, metadata = extract_from_json(json_path, replay_stats)
    
    if not state_traj:
        print("[ERROR] Failed to extract state trajectory")
        return
    
    print(f"\nExtracted {len(state_traj)} state tuples")
    print(f"\nFirst 10 states:")
    for i, state in enumerate(state_traj[:10]):
        j, k, r, c, d, e = state
        print(f"  Step {i}: j={j}, k={k}, r={r}, c={c}, d={d}, e={e}")
    
    print(f"\nLast 10 states:")
    for i, state in enumerate(state_traj[-10:], len(state_traj)-10):
        j, k, r, c, d, e = state
        print(f"  Step {i}: j={j}, k={k}, r={r}, c={c}, d={d}, e={e}")
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
    else:
        json_name = json_path.stem
        output_path = Path(f"data/state_trajectories/{json_name}_state.json")
    
    save_state_trajectory(state_traj, output_path, metadata)


if __name__ == '__main__':
    main()

