"""
=======================================================
  CLASH ROYALE AI — MATCH ROLLBACK UTILITY
=======================================================
Quickly rolls back neural network weights (rl_agent.pt),
replay buffer experiences (replay_buffer.pkl), and
training stats to the last git-committed clean checkpoint.
"""
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import config

def rollback():
    print("\n" + "=" * 55)
    print("  CLASH ROYALE AI — MATCH ROLLBACK UTILITY")
    print("=" * 55)
    
    targets = [
        config.REPLAY_BUFFER_PATH,
        config.RL_MODEL_PATH,
        config.RL_FINAL_PATH,
        config.TRAINING_STATS_PATH
    ]
    
    files_to_restore = []
    for t in targets:
        if os.path.exists(t):
            files_to_restore.append(t)
        else:
            files_to_restore.append(os.path.basename(t))
    
    cmd = ["git", "checkout", "--"] + files_to_restore
    res = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if res.returncode == 0:
        print("[SUCCESS] Successfully restored replay buffer and agent weights!")
        print("   The last match was completely purged from training memory.")
    else:
        # Fallback to base filenames
        fallback_cmd = ["git", "checkout", "--", "replay_buffer.pkl", "rl_agent.pt", "rl_agent_final.pt", "training_stats.json"]
        res2 = subprocess.run(fallback_cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        if res2.returncode == 0:
            print("[SUCCESS] Successfully restored replay buffer and agent weights (via git root)!")
        else:
            print(f"[ERROR] Error during rollback: {res.stderr or res2.stderr}")
            sys.exit(1)

if __name__ == '__main__':
    rollback()
