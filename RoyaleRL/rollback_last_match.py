"""
=======================================================
  CLASH ROYALE AI — MATCH ROLLBACK UTILITY
=======================================================
Quickly rolls back neural network weights (rl_agent.pt),
replay buffer experiences (replay_buffer.pkl), and
training stats to the last git-committed clean checkpoint.
"""
import subprocess
import sys

def rollback():
    print("\n" + "=" * 55)
    print("  CLASH ROYALE AI — MATCH ROLLBACK UTILITY")
    print("=" * 55)
    
    files_to_restore = [
        "replay_buffer.pkl",
        "rl_agent.pt",
        "rl_agent_final.pt",
        "training_stats.json"
    ]
    
    cmd = ["git", "checkout", "--"] + files_to_restore
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("✅ Successfully restored replay buffer and agent weights!")
        print("   The last match was completely purged from training memory.")
    else:
        print(f"❌ Error during rollback: {res.stderr}")
        sys.exit(1)

if __name__ == '__main__':
    rollback()
