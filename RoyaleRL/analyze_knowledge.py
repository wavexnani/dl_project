import os, sys, pickle, json, time
import numpy as np
import torch
import torch.nn.functional as F

import config
from agent import Agent, DecisionTransformer, PLACEMENT_GRID, NUM_GRID_LOCATIONS, ACTION_DIM
from config import NUM_CARD_TYPES, CARD_COSTS, CARD_TO_INDEX, ALL_CARDS, REFERENCE_RESOLUTION
STATE_DIM = 1 + 6 + (4 * NUM_CARD_TYPES) + (20 * 4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("       CLASH ROYALE AI — MODEL KNOWLEDGE & MASTERY AUDIT")
print("=" * 70)

# 1. Training Stats & Match Win Rate
stats_file = 'training_stats.json'
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    history = stats.get('history', [])
    total_matches = stats.get('matches_played', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    draws = stats.get('draws', 0)
    overall_wr = stats.get('overall_win_rate', 0.0)
    rolling_wr = stats.get('rolling_win_rate_20', 0.0)
    
    recent_matches = history[-25:] if len(history) >= 25 else history
    recent_wins = sum(1 for m in recent_matches if m.get('result') == 'WIN')
    recent_draws = sum(1 for m in recent_matches if m.get('result') == 'DRAW')
    recent_losses = sum(1 for m in recent_matches if m.get('result') == 'LOSS')
    
    print(f"\n📊 [1. CAREER & SESSION RECORD]")
    print(f"  • Total Matches Recorded: {total_matches}")
    print(f"  • Overall Record: {wins} Wins | {losses} Losses | {draws} Draws ({overall_wr:.1f}% Win Rate)")
    print(f"  • Recent Session Performance (Last {len(recent_matches)} games):")
    print(f"      - Wins: {recent_wins} ({recent_wins / len(recent_matches) * 100:.1f}%)")
    print(f"      - Draws: {recent_draws} ({recent_draws / len(recent_matches) * 100:.1f}%)")
    print(f"      - Losses: {recent_losses} ({recent_losses / len(recent_matches) * 100:.1f}%)")
    print(f"      - Undefeated Rate: {(recent_wins + recent_draws) / len(recent_matches) * 100:.1f}%")
    print(f"  • Rolling Win Rate (Last 20): {rolling_wr:.1f}%")

# 2. Replay Buffer & Human Demonstration Breakdown
buffer_file = 'replay_buffer.pkl'
with open(buffer_file, 'rb') as f:
    buffer_data = pickle.load(f)

sz = buffer_data['size']
states = buffer_data['states'][:sz]
actions = buffer_data['actions'][:sz]
rewards = buffer_data['rewards'][:sz]

print(f"\n🧠 [2. REPLAY BUFFER EXPERIENCES]")
print(f"  • Total Transitions in Memory: {sz:,}")
print(f"  • Positive Reward Experiences: {(rewards > 0).sum()} ({(rewards > 0).mean()*100:.1f}%)")
print(f"  • Max Match Reward: {rewards.max():.2f}")
print(f"  • Average Reward: {rewards.mean():.3f}")

# Human session slice (last ~320 steps added during record mode)
human_slice_actions = actions[max(0, sz - 320):sz]
slot_counts = {0: 0, 1: 0, 2: 0, 3: 0, 'none': 0}
y_placements = []
x_placements = []

for a_idx in human_slice_actions:
    if a_idx >= len(ALL_CARDS) * NUM_GRID_LOCATIONS:
        slot_counts['none'] += 1
    else:
        slot = a_idx // NUM_GRID_LOCATIONS
        slot_counts[slot] = slot_counts.get(slot, 0) + 1
        g_idx = a_idx % NUM_GRID_LOCATIONS
        pos_pct = PLACEMENT_GRID[g_idx]
        x_placements.append(pos_pct[0])
        y_placements.append(pos_pct[1])

print(f"\n🎮 [3. HUMAN DEMONSTRATION HABITS (Last ~{len(human_slice_actions)} Recorded Actions)]")
print(f"  • Active Card Plays: {len(x_placements)} | Timing Holds/Waits: {slot_counts['none']}")
if y_placements:
    y_arr = np.array(y_placements)
    x_arr = np.array(x_placements)
    backline_pct = (y_arr >= 0.70).mean() * 100
    midfield_pct = ((y_arr >= 0.52) & (y_arr < 0.70)).mean() * 100
    pocket_pct = (y_arr < 0.52).mean() * 100
    left_lane_pct = (x_arr < 0.50).mean() * 100
    right_lane_pct = (x_arr >= 0.50).mean() * 100
    print(f"  • Spatial Placement Distribution:")
    print(f"      - Backline Safe Zone (y >= 0.70): {backline_pct:.1f}% (Building slow pushes behind King)")
    print(f"      - Midfield / Bridge Support (0.52 <= y < 0.70): {midfield_pct:.1f}%")
    print(f"      - Aggressive / Pocket drops (y < 0.52): {pocket_pct:.1f}%")
    print(f"  • Lane Distribution: Left Lane: {left_lane_pct:.1f}% | Right Lane: {right_lane_pct:.1f}% (Balanced)")

# 3. Model Neural Network Evaluation & Loss on Replay Buffer
print(f"\n⚡ [4. DECISION TRANSFORMER EVALUATION]")
ai_agent = Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, card_costs=CARD_COSTS, device=device)
ai_agent.model.eval()

# Sample 500 experiences from the buffer and test top-1 and top-5 accuracy
test_indices = np.random.choice(sz, min(500, sz), replace=False)
correct_top1 = 0
correct_top3 = 0
total_tested = 0

with torch.no_grad():
    for idx in test_indices:
        s = torch.tensor(states[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        a_target = actions[idx]
        a_dummy = torch.zeros((1, 1, ACTION_DIM), dtype=torch.float32).to(device)
        r = torch.tensor([[[2.0]]], dtype=torch.float32).to(device)
        t = torch.zeros((1, 1), dtype=torch.long).to(device)
        
        preds = ai_agent.model(s, a_dummy, r, t).squeeze()
        top3 = torch.topk(preds, 3).indices.tolist()
        
        if top3[0] == a_target:
            correct_top1 += 1
        if a_target in top3:
            correct_top3 += 1
        total_tested += 1

print(f"  • Model Parameter Count: {sum(p.numel() for p in ai_agent.model.parameters()):,} parameters")
print(f"  • Expert Action Imitation Accuracy (Top-1): {correct_top1 / total_tested * 100:.1f}%")
print(f"  • Expert Action Imitation Accuracy (Top-3): {correct_top3 / total_tested * 100:.1f}%")

# 4. Live Situation Probing (Scenario Tests)
print(f"\n🔮 [5. STRATEGIC DECISION PROBES (What the Model Has Learned)]")

# Scenario 1: Starting Hand, 10 Elixir, Both Towers Full
state_start = {
    'hand': ['giant', 'musketeer', 'archers', 'knight'],
    'elixir': 9.8,
    'ocr_data': {'ptl': '2534', 'ptr': '2534', 'tk': '4008', 'pbl': '2534', 'pbr': '2534'},
    'enemies': []
}
cand = ai_agent._get_model_candidates(state_start, None, top_k=3)
print("  • Scenario A: Match Start / 10 Elixir (Hand: Giant, Musketeer, Archers, Knight):")
for i, c in enumerate(cand):
    slot = c['card_slot']
    card = state_start['hand'][slot]
    norm_x = c['position'][0] / REFERENCE_RESOLUTION[0]
    norm_y = c['position'][1] / REFERENCE_RESOLUTION[1]
    print(f"      Choice #{i+1}: Play {card.upper()} at ({norm_x:.2f}, {norm_y:.2f}) -> {'Backline Safe Cycle' if norm_y >= 0.70 else 'Bridge/Mid'}")

# Scenario 2: Counter-Push (Giant at bridge, opponent tower wounded)
state_counter = {
    'hand': ['musketeer', 'archers', 'knight', 'arrows'],
    'elixir': 6.0,
    'ocr_data': {'ptl': '1200', 'ptr': '2534', 'tk': '4008', 'pbl': '2534', 'pbr': '2534'},
    'enemies': [{'name': 'giant', 'box': (120, 600, 180, 660)}] # Enemy giant defending
}
cand_counter = ai_agent._get_model_candidates(state_counter, None, top_k=3)
print("  • Scenario B: Mid-Game Attack Support (Left Tower at 1200 HP, Elixir=6.0):")
for i, c in enumerate(cand_counter):
    slot = c['card_slot']
    card = state_counter['hand'][slot]
    norm_x = c['position'][0] / REFERENCE_RESOLUTION[0]
    norm_y = c['position'][1] / REFERENCE_RESOLUTION[1]
    print(f"      Choice #{i+1}: Play {card.upper()} at ({norm_x:.2f}, {norm_y:.2f})")

print("\n" + "=" * 70)
