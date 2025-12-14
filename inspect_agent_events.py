"""
Minimal script to inspect AgentEvent structure without requiring pygame.
"""

import sys
from pathlib import Path

# Add funstudy to path
FUNSTUDY_ROOT = Path(__file__).parent / 'funstudy' / 'EDRL-TAC-codes-v3'
sys.path.insert(0, str(FUNSTUDY_ROOT))

import jpype
from jpype import JString
from root import PRJROOT
from src.utils.filesys import getpath

# Initialize JVM
if not jpype.isJVMStarted():
    jarPath = getpath('smb/Mario-AI-Interface.jar')
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        f"-Djava.class.path={jarPath}"
    )
    jpype.JClass("java.lang.System").setProperty('user.dir', f'{PRJROOT}/smb')

# Get proxy
proxy_class = jpype.JClass("MarioSurveyProxy")
proxy = proxy_class()

# Test with a replay file
lvl_path = "exp_data/survey data/formal/levels/Collector-48.lvl"
rep_path = "data/reps/067d54d0-b2cc-4d65-9348-f1dae3fc9398_Collector-48.rep"

print(f"Testing with:\n  Level: {lvl_path}\n  Replay: {rep_path}\n")

try:
    jres = proxy.reproduceGameResults(
        JString(getpath(lvl_path)),
        JString(str(Path(rep_path).absolute()))
    )
    
    print("=== MarioResult Methods ===")
    print(f"getKillsTotal(): {jres.getKillsTotal()}")
    print(f"getCurrentCoins(): {jres.getCurrentCoins()}")
    print(f"getLives(): {jres.getLives()}")
    print(f"getGameStatus(): {jres.getGameStatus()}")
    print(f"getCompletionPercentage(): {jres.getCompletionPercentage()}")
    
    agent_events = jres.getAgentEvents()
    print(f"\nAgentEvents size: {agent_events.size()}")
    
    if agent_events.size() > 0:
        first_event = agent_events.get(0)
        
        print("\n=== AgentEvent Object ===")
        print(f"Class: {first_event.getClass().getName()}")
        
        print("\n=== AgentEvent Methods ===")
        methods = [m for m in dir(first_event) if not m.startswith('_') and not m.startswith('wait')]
        for m in sorted(methods):
            print(f"  - {m}")
        
        print("\n=== AgentEvent Fields (via reflection) ===")
        fields = first_event.getClass().getDeclaredFields()
        for field in fields:
            field.setAccessible(True)
            field_name = field.getName()
            try:
                value = field.get(first_event)
                value_str = str(value) if value is not None else "null"
                print(f"  - {field_name}: {field.getType().getName()} = {value_str}")
            except Exception as e:
                print(f"  - {field_name}: {field.getType().getName()} (could not access: {e})")
        
        print("\n=== Testing AgentEvent Methods ===")
        try:
            x = first_event.getMarioX()
            y = first_event.getMarioY()
            print(f"getMarioX(): {x}")
            print(f"getMarioY(): {y}")
        except Exception as e:
            print(f"Error getting position: {e}")
        
        # Try to find action-related methods/fields
        print("\n=== Looking for Action Information ===")
        action_methods = [m for m in methods if 'action' in m.lower() or 'button' in m.lower() or 'key' in m.lower()]
        if action_methods:
            print("Possible action-related methods:")
            for m in action_methods:
                try:
                    result = getattr(first_event, m)()
                    print(f"  {m}(): {result}")
                except Exception as e:
                    print(f"  {m}(): Error - {e}")
        else:
            print("No obvious action-related methods found")
        
        action_fields = [f.getName() for f in fields if 'action' in str(f.getName()).lower() or 'button' in str(f.getName()).lower() or 'key' in str(f.getName()).lower()]
        if action_fields:
            print("\nPossible action-related fields:")
            for fname in action_fields:
                try:
                    field = first_event.getClass().getDeclaredField(fname)
                    field.setAccessible(True)
                    value = field.get(first_event)
                    print(f"  {fname}: {value}")
                except Exception as e:
                    print(f"  {fname}: Error - {e}")
        
        # Show sample of events
        print(f"\n=== Sample Events (first 5) ===")
        for i in range(min(5, agent_events.size())):
            event = agent_events.get(i)
            x = event.getMarioX()
            y = event.getMarioY()
            print(f"Event {i}: x={x:.2f}, y={y:.2f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

