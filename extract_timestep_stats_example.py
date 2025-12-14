"""
Example script to extract per-time-step kills and coins from trajectory data.

Usage:
    python extract_timestep_stats_example.py
"""

import sys
from pathlib import Path

# Add funstudy to path
FUNSTUDY_ROOT = Path(__file__).parent / 'funstudy' / 'EDRL-TAC-codes-v3'
sys.path.insert(0, str(FUNSTUDY_ROOT))

from src.smb.proxy import MarioSurveyProxy


def extract_timestep_stats_from_replay(lvl_path: str, rep_path: str):
    """
    Extract per-time-step statistics from a replay file.
    
    Args:
        lvl_path: Path to .lvl file
        rep_path: Path to .rep file
    
    Returns:
        Dictionary with per-time-step statistics
    """
    proxy = MarioSurveyProxy()
    
    # Use get_timestep_stats=True to get per-step data
    result = proxy.reproduce(lvl_path, rep_path, get_timestep_stats=True)
    
    return result


def main():
    """Example usage."""
    # Example paths - adjust to your actual files
    lvl_path = "funstudy/EDRL-TAC-codes-v3/exp_data/survey data/formal/levels/lvl1.lvl"
    rep_path = "data/reps/example.rep"
    
    print("Extracting per-time-step statistics...")
    print(f"Level: {lvl_path}")
    print(f"Replay: {rep_path}")
    
    try:
        result = extract_timestep_stats_from_replay(lvl_path, rep_path)
        
        print(f"\nTotal stats:")
        print(f"  Status: {result['status']}")
        print(f"  Total kills: {result['#kills']}")
        print(f"  Final coins: {result['#coins']}")
        print(f"  Trace length: {len(result['trace'])} steps")
        
        if 'trace_with_stats' in result:
            print(f"\nPer-time-step stats (first 10 steps):")
            for i, step in enumerate(result['trace_with_stats'][:10]):
                x, y, kills, coins = step
                print(f"  Step {i}: pos=({x:.1f}, {y:.1f}), kills={kills}, coins={coins}")
            
            print(f"\nPer-time-step stats (last 10 steps):")
            for i, step in enumerate(result['trace_with_stats'][-10:], len(result['trace_with_stats'])-10):
                x, y, kills, coins = step
                print(f"  Step {i}: pos=({x:.1f}, {y:.1f}), kills={kills}, coins={coins}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Per-step kill/coin data may not be available in AgentEvent objects.")
        print("The code will try Java reflection, but if that fails, it uses interpolation.")


if __name__ == '__main__':
    main()

