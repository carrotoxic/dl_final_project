"""
Generate state trajectories with (j, k, r, c, d, e) tuples for each time step.

BEST APPROACH: Use original JSON survey files for accurate action encoding.

State tuple definition:
- j: number of jumps (cumulative) - counted when jump button transitions from False to True
- k: number of enemies killed (cumulative) - interpolated based on position progress
- r: number of times player started running (cumulative) - counted when speed button transitions
- c: number of coins collected (cumulative) - interpolated based on position progress
- d: number of times player died (cumulative) - detected from position resets
- e: unique encoding for each action - numeric value from JSON actions0 field

This script supports both .rep files (limited) and JSON survey files (accurate).
Use --json_path for accurate action data.
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add funstudy to path
FUNSTUDY_ROOT = Path(__file__).parent / 'funstudy' / 'EDRL-TAC-codes-v3'
sys.path.insert(0, str(FUNSTUDY_ROOT))

import jpype
from jpype import JString, JInt, JBoolean
from src.smb.proxy import MarioSurveyProxy
from src.utils.filesys import getpath


def encode_action(agent_event) -> int:
    """
    Encode action from AgentEvent to a unique integer.
    
    AgentEvent.getActions() returns a boolean array [LEFT, RIGHT, DOWN, JUMP, SPEED]
    - Index 0: LEFT
    - Index 1: RIGHT  
    - Index 2: DOWN
    - Index 3: JUMP
    - Index 4: SPEED/RUN
    
    Returns:
        Integer encoding of the action (0-31 for 5 buttons using bit encoding)
    """
    action_code = 0
    
    try:
        # AgentEvent has getActions() method that returns boolean array
        actions_array = agent_event.getActions()
        
        # Convert boolean array to bit-encoded integer
        # actions_array[0] = LEFT  -> bit 0
        # actions_array[1] = RIGHT -> bit 1
        # actions_array[2] = DOWN  -> bit 2
        # actions_array[3] = JUMP  -> bit 3
        # actions_array[4] = SPEED -> bit 4
        
        if actions_array[0]:  # LEFT
            action_code |= 0x01
        if actions_array[1]:  # RIGHT
            action_code |= 0x02
        if actions_array[2]:  # DOWN
            action_code |= 0x04
        if actions_array[3]:  # JUMP
            action_code |= 0x08
        if actions_array[4]:  # SPEED/RUN
            action_code |= 0x10
        
    except Exception as e:
        # Fallback: try to get actions field directly
        try:
            actions_field = agent_event.getClass().getDeclaredField("actions")
            actions_field.setAccessible(True)
            actions_array = actions_field.get(agent_event)
            
            if actions_array[0]:  # LEFT
                action_code |= 0x01
            if actions_array[1]:  # RIGHT
                action_code |= 0x02
            if actions_array[2]:  # DOWN
                action_code |= 0x04
            if actions_array[3]:  # JUMP
                action_code |= 0x08
            if actions_array[4]:  # SPEED/RUN
                action_code |= 0x10
        except:
            action_code = 0
    
    return action_code


def is_jump_action(agent_event) -> bool:
    """
    Check if action represents a jump.
    Directly checks the JUMP button (index 3) in actions array.
    """
    try:
        actions = agent_event.getActions()
        return bool(actions[3])  # JUMP is at index 3
    except:
        return False


def is_run_action(agent_event) -> bool:
    """Check if action represents running (SPEED button pressed)."""
    try:
        actions = agent_event.getActions()
        return bool(actions[4])  # SPEED/RUN is at index 4
    except:
        return False


def decode_action_code(action_value: int) -> Dict[str, bool]:
    """
    Decode numeric action encoding to button states.
    
    Based on Mario AI Framework action encoding:
    - Bit 0 (1): LEFT
    - Bit 1 (2): RIGHT
    - Bit 2 (4): DOWN
    - Bit 3 (8): JUMP
    - Bit 4 (16): SPEED/RUN
    
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
    Extract state trajectory from original JSON survey file (ACCURATE METHOD).
    
    Args:
        json_path: Path to original survey JSON file
        replay_stats: Optional dict with final stats from replay (kills, coins, lives, etc.)
    
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
    initial_lives = 3
    
    # Track positions for interpolating kills/coins based on progress
    positions = []
    valid_indices = []
    
    # First pass: collect all positions
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
        
        # Decode action from JSON
        action_value = elem['actions0'].get('0', 0)
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
        if x_range > 0:
            progress = (x - min_x) / x_range
            progress = max(0.0, min(1.0, progress))
        else:
            progress = (idx + 1) / len(valid_indices) if len(valid_indices) > 1 else 1.0
        
        current_kills = int(total_kills * progress)
        current_coins = int(total_coins * progress)
        
        # Create state tuple: (j, k, r, c, d, e)
        state = (
            cumulative_jumps,      # j
            current_kills,         # k
            cumulative_runs,       # r
            current_coins,         # c
            cumulative_deaths,     # d
            action_value           # e - raw numeric action encoding
        )
        state_trajectory.append(state)
    
    # Ensure final values match totals
    if state_trajectory and replay_stats:
        final_state = list(state_trajectory[-1])
        final_state[1] = total_kills
        final_state[3] = total_coins
        state_trajectory[-1] = tuple(final_state)
    
    metadata = {
        'source': 'json_survey_file',
        'json_path': str(json_path),
        'n_steps': len(state_trajectory),
        'total_kills': total_kills,
        'total_coins': total_coins,
        'estimated_deaths': cumulative_deaths,
    }
    
    return state_trajectory, metadata


def find_replay_file(json_name: str, reps_dir: Path) -> Optional[Path]:
    """Find corresponding .rep file for a JSON file."""
    rep_name = json_name.replace('.json', '.rep')
    rep_path = reps_dir / rep_name
    return rep_path if rep_path.exists() else None


def find_level_file(json_name: str, levels_dir: Path) -> Optional[Path]:
    """Find corresponding .lvl file for a JSON file."""
    parts = json_name.replace('.json', '').split('_', 1)
    if len(parts) < 2:
        return None
    level_name = parts[1]
    lvl_path = levels_dir / f"{level_name}.lvl"
    return lvl_path if lvl_path.exists() else None


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


def detect_jump_from_position(y_positions: List[float], idx: int, threshold: float = 1.0) -> bool:
    """
    Detect jump from position changes (fallback if action code unavailable).
    A jump typically causes upward Y movement (decrease in Y value).
    """
    if idx == 0 or idx >= len(y_positions):
        return False
    
    # If Y decreases (goes up) significantly, likely a jump
    y_change = y_positions[idx-1] - y_positions[idx]
    return y_change > threshold  # Mario goes up


def extract_state_trajectory(proxy: MarioSurveyProxy, lvl_path: str, rep_path: str) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Extract state trajectory from replay file.
    
    Returns:
        List of state tuples: [(j, k, r, c, d, e), ...] for each time step
    """
    # Reproduce the game to get AgentEvents
    # Convert paths to absolute
    lvl_path_abs = Path(lvl_path).absolute() if not Path(lvl_path).is_absolute() else Path(lvl_path)
    rep_path_abs = Path(rep_path).absolute() if not Path(rep_path).is_absolute() else Path(rep_path)
    
    jres = proxy._MarioSurveyProxy__proxy.reproduceGameResults(
        JString(str(lvl_path_abs)), 
        JString(str(rep_path_abs))
    )

    
    agent_events = jres.getAgentEvents()
    n_events = agent_events.size()
    
    # Get final statistics
    total_kills = int(jres.getKillsTotal())
    final_coins = int(jres.getCurrentCoins())
    initial_lives = 3  # Default starting lives, might need adjustment
    final_lives = int(jres.getLives())
    deaths = initial_lives - final_lives  # Approximate deaths
    
    # Track cumulative statistics and positions
    state_trajectory = []
    cumulative_jumps = 0
    cumulative_runs = 0
    prev_run_state = False
    prev_jump_state = False
    
    # Track deaths by detecting position resets
    cumulative_deaths = 0
    prev_x = None
    
    # Extract per-step information
    for i in range(n_events):
        item = agent_events.get(i)
        x = float(item.getMarioX())
        y = float(item.getMarioY())
        
        # Detect death: position resets significantly backward (Mario respawns)
        # A death typically causes X position to reset to near start (~8.0)
        if prev_x is not None and x < prev_x - 100:  # Moved backward significantly (likely death/respawn)
            cumulative_deaths += 1
        
        # Get action encoding
        action_code = encode_action(item)
        
        # Detect jump starts (transition from not-jumping to jumping)
        current_jump_state = is_jump_action(item)
        if current_jump_state and not prev_jump_state:
            cumulative_jumps += 1
        prev_jump_state = current_jump_state
        
        # Detect run starts (transition from not-running to running)
        current_run_state = is_run_action(item)
        if current_run_state and not prev_run_state:
            cumulative_runs += 1
        prev_run_state = current_run_state
        
        # Get cumulative kills and coins at this step
        # Try to extract from AgentEvent or interpolate
        current_kills = 0
        current_coins = 0
        
        # For kills and coins, AgentEvent doesn't store them directly
        # We'll need to track them by detecting changes or interpolate
        # For now, use linear interpolation as approximation
        # TODO: A more accurate approach would replay step-by-step and query game state
        if n_events > 1:
            progress = (i + 1) / n_events
            current_kills = int(total_kills * progress)
            current_coins = int(final_coins * progress)
        else:
            current_kills = total_kills
            current_coins = final_coins
        
        # Store previous position for death detection
        prev_x = x
        prev_y = y
        
        # Create state tuple: (j, k, r, c, d, e)
        state = (
            cumulative_jumps,      # j - cumulative jumps
            current_kills,          # k - cumulative kills at this step
            cumulative_runs,        # r - cumulative run starts
            current_coins,          # c - cumulative coins at this step
            cumulative_deaths,      # d - cumulative deaths
            action_code             # e - action encoding for this step
        )
        state_trajectory.append(state)
    
    # Ensure final values match totals
    if state_trajectory:
        final_state = list(state_trajectory[-1])
        final_state[1] = total_kills  # k
        final_state[3] = final_coins  # c
        state_trajectory[-1] = tuple(final_state)
    
    return state_trajectory


def inspect_agent_event(proxy: MarioSurveyProxy, lvl_path: str, rep_path: str):
    """
    Inspect AgentEvent object to see what methods/fields are available.
    This helps determine how to extract action information.
    """
    jres = proxy._MarioSurveyProxy__proxy.reproduceGameResults(
        JString(getpath(lvl_path)), 
        JString(getpath(rep_path))
    )
    
    agent_events = jres.getAgentEvents()
    if agent_events and agent_events.size() > 0:
        first_event = agent_events.get(0)
        
        print("AgentEvent methods:")
        methods = [m for m in dir(first_event) if not m.startswith('_') and not m.startswith('wait')]
        for m in sorted(methods):
            print(f"  - {m}")
        
        print("\nAgentEvent class:")
        print(f"  {first_event.getClass().getName()}")
        
        print("\nAgentEvent fields (via reflection):")
        fields = first_event.getClass().getDeclaredFields()
        for field in fields:
            field.setAccessible(True)
            try:
                value = field.get(first_event)
                print(f"  - {field.getName()}: {type(value).__name__} = {value}")
            except:
                print(f"  - {field.getName()}: (could not access)")


def save_state_trajectory(state_traj: List[Tuple[int, int, int, int, int, int]], 
                         output_path: Path, 
                         metadata: dict = None):
    """Save state trajectory to JSON file."""
    result = {
        'state_trajectory': [[j, k, r, c, d, e] for j, k, r, c, d, e in state_traj],
        'metadata': metadata or {}
    }
    
    with output_path.open('w') as f:
        json.dump(result, f, indent=2)
    
    print(f"[INFO] Saved state trajectory to {output_path}")


def main():
    """
    Main function with support for both JSON files (accurate) and .rep files (limited).
    
    For accurate results, use:
        --json_path <path_to_json_file> [--reps_dir <reps_dir>] [--levels_dir <levels_dir>]
    
    For .rep files (limited accuracy):
        --lvl_path <path_to_lvl> --rep_path <path_to_rep>
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract accurate state trajectories from JSON survey files or replay files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best approach: Use JSON files for accurate actions
  python generate_state_trajectories.py --json_path "jsons/file.json" --reps_dir "data/reps"
  
  # Limited approach: Use .rep files (actions will be inaccurate)
  python generate_state_trajectories.py --lvl_path "levels/file.lvl" --rep_path "reps/file.rep"
        """
    )
    
    # JSON-based approach (accurate)
    parser.add_argument("--json_path", type=str, default=None, help="Path to original survey JSON file (BEST for accuracy)")
    parser.add_argument("--reps_dir", type=str, default="data/reps", help="Directory containing .rep files (for stats)")
    parser.add_argument("--levels_dir", type=str, default="funstudy/EDRL-TAC-codes-v3/exp_data/survey data/formal/levels", help="Directory containing .lvl files")
    
    # .rep-based approach (limited)
    parser.add_argument("--lvl_path", type=str, default=None, help="Path to .lvl file (used with --rep_path)")
    parser.add_argument("--rep_path", type=str, default=None, help="Path to .rep file (limited accuracy)")
    
    # Common options
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--inspect", action="store_true", help="Inspect AgentEvent structure")
    
    args = parser.parse_args()
    
    # Use JSON-based approach if json_path is provided
    if args.json_path:
        json_path = Path(args.json_path)
        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {json_path}")
            return
        
        print(f"Processing JSON file: {json_path.name}")
        
        # Try to get replay stats
        replay_stats = None
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
        
        # Extract from JSON
        state_traj, metadata = extract_from_json(json_path, replay_stats)
        
        if not state_traj:
            print("[ERROR] Failed to extract state trajectory")
            return
        
    # Use .rep-based approach
    elif args.lvl_path and args.rep_path:
        proxy = MarioSurveyProxy()
        
        if args.inspect:
            print("Inspecting AgentEvent structure...")
            inspect_agent_event(proxy, args.lvl_path, args.rep_path)
            return
        
        print(f"Extracting state trajectory from {args.rep_path}...")
        print("[WARN] .rep files don't store actions accurately. Use --json_path for accurate results.")
        state_traj = extract_state_trajectory(proxy, args.lvl_path, args.rep_path)
        metadata = {
            'level': args.lvl_path,
            'replay': args.rep_path,
            'n_steps': len(state_traj),
            'note': 'Limited accuracy - actions are not stored in .rep files'
        }
    else:
        parser.print_help()
        print("\n[ERROR] Must provide either --json_path (recommended) or both --lvl_path and --rep_path")
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
        if args.json_path:
            json_name = Path(args.json_path).stem
            output_path = Path(f"data/state_trajectories/{json_name}_state.json")
        else:
            rep_name = Path(args.rep_path).stem
            output_path = Path(f"data/state_trajectories/{rep_name}_state.json")
    
    save_state_trajectory(state_traj, output_path, metadata)


if __name__ == '__main__':
    main()

