import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import config

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Dark GitHub Style Theme matching presentation charts
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

ACCENT = '#58a6ff'
GREEN = '#3fb950'
RED = '#f85149'
ORANGE = '#d29922'
PURPLE = '#bc8cff'
CYAN = '#39d2c0'

stats_file = getattr(config, 'TRAINING_STATS_PATH', os.path.join(ROOT_DIR, 'data', 'training_stats.json'))
if not os.path.exists(stats_file) and os.path.exists('training_stats.json'):
    stats_file = 'training_stats.json'
with open(stats_file, 'r') as f:
    stats = json.load(f)

history = stats.get('history', [])
total_matches = stats.get('matches_played', 216)
wins = stats.get('wins', 125)
losses = stats.get('losses', 46)
draws = stats.get('draws', 45)
overall_wr = stats.get('overall_win_rate', 57.9)
rolling_wr = stats.get('rolling_win_rate_20', 75.0)

match_nums = [h['match_num'] for h in history]
results = [h['result'] for h in history]
rewards = [h['reward'] for h in history]
buffers = [h['buffer_size'] for h in history]
epsilons = [h.get('epsilon', 0.05) for h in history]

# Compute rolling win rate across history
window = 15
rolling_rates = []
for i in range(len(results)):
    start_idx = max(0, i - window + 1)
    sub = results[start_idx:i+1]
    w = sum(1 for r in sub if r == 'WIN')
    rolling_rates.append((w / len(sub)) * 100)

# Reward moving average
reward_ma = []
r_window = 10
for i in range(len(rewards)):
    start_idx = max(0, i - r_window + 1)
    sub = rewards[start_idx:i+1]
    reward_ma.append(np.mean(sub))

# Create 2x2 multi-panel presentation figure
fig = plt.figure(figsize=(16, 11))
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

# --- 1. Career Win / Loss / Draw Breakdown (Donut + Metrics) ---
ax1 = fig.add_subplot(gs[0, 0])
sizes = [wins, losses, draws]
labels = [f'Wins\n({wins})', f'Losses\n({losses})', f'Draws\n({draws})']
colors = [GREEN, RED, ORANGE]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=labels, autopct='%1.1f%%', colors=colors,
    startangle=140, pctdistance=0.75,
    textprops={'fontsize': 11, 'color': '#c9d1d9'}
)
for at in autotexts:
    at.set_color('#ffffff')
    at.set_fontweight('bold')
centre_circle = plt.Circle((0,0), 0.55, fc='#161b22')
ax1.add_artist(centre_circle)
ax1.text(0, 0.08, f"{overall_wr:.1f}%", ha='center', va='center', fontsize=20, fontweight='bold', color=GREEN)
ax1.text(0, -0.15, "Overall Win Rate", ha='center', va='center', fontsize=10, color='#8b949e')
ax1.set_title(f"Career Match Outcomes ({total_matches} Games Recorded)", pad=15)

# --- 2. Rolling Win Rate Progression Curve ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(match_nums, rolling_rates, color=CYAN, linewidth=2.5, label=f'Rolling Win Rate (window={window})')
ax2.axhline(overall_wr, color=ORANGE, linestyle='--', linewidth=1.8, label=f'Overall Career Baseline ({overall_wr:.1f}%)')
ax2.axhline(rolling_wr, color=GREEN, linestyle=':', linewidth=2, label=f'Recent 20-Game Peak ({rolling_wr:.1f}%)')
ax2.fill_between(match_nums, rolling_rates, overall_wr, where=(np.array(rolling_rates) >= overall_wr),
                 color=GREEN, alpha=0.18, interpolate=True)
ax2.fill_between(match_nums, rolling_rates, overall_wr, where=(np.array(rolling_rates) < overall_wr),
                 color=RED, alpha=0.18, interpolate=True)
ax2.set_xlabel('Match Number')
ax2.set_ylabel('Win Rate (%)')
ax2.set_ylim(20, 100)
ax2.set_title('Online Reinforcement Learning: Win Rate Trajectory')
ax2.grid(True, alpha=0.25)
ax2.legend(loc='lower right', fontsize=9.5)

# --- 3. Step & Match Reward Progression ---
ax3 = fig.add_subplot(gs[1, 0])
win_matches = [m for m, r in zip(match_nums, results) if r == 'WIN']
win_rewards = [rw for rw, r in zip(rewards, results) if r == 'WIN']
loss_matches = [m for m, r in zip(match_nums, results) if r == 'LOSS']
loss_rewards = [rw for rw, r in zip(rewards, results) if r == 'LOSS']
draw_matches = [m for m, r in zip(match_nums, results) if r == 'DRAW']
draw_rewards = [rw for rw, r in zip(rewards, results) if r == 'DRAW']

ax3.scatter(win_matches, win_rewards, color=GREEN, alpha=0.65, s=35, label='Win Reward (+1 to +6.4)')
ax3.scatter(loss_matches, loss_rewards, color=RED, alpha=0.65, s=35, label='Loss Penalty (-1 to -3.0)')
ax3.scatter(draw_matches, draw_rewards, color=ORANGE, alpha=0.65, s=35, label='Draw (0.0)')
ax3.plot(match_nums, reward_ma, color=ACCENT, linewidth=2.5, label='10-Match Moving Avg')
ax3.axhline(0, color='#8b949e', linestyle='-', linewidth=0.8)
ax3.set_xlabel('Match Number')
ax3.set_ylabel('Match Reward (Tower HP Δ + Crown Bonus)')
ax3.set_title('Match Reward Evolution (Autonomous Self-Play)')
ax3.grid(True, alpha=0.25)
ax3.legend(loc='upper left', fontsize=9)

# --- 4. Replay Buffer Growth & Experience Scaling ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(match_nums, buffers, color=PURPLE, linewidth=2.5, label='Replay Buffer Transitions')
ax4.set_xlabel('Match Number')
ax4.set_ylabel('Experience Transitions (s, a, r, s\')', color=PURPLE)
ax4.tick_params(axis='y', labelcolor=PURPLE)
ax4.grid(True, alpha=0.25)

# Twin axis for epsilon decay
ax4_twin = ax4.twinx()
ax4_twin.plot(match_nums, epsilons, color=ORANGE, linewidth=2, linestyle='-.', label='Exploration Rate (ε)')
ax4_twin.set_ylabel('Epsilon (Exploration)', color=ORANGE)
ax4_twin.tick_params(axis='y', labelcolor=ORANGE)
ax4_twin.set_ylim(0, 0.25)

# Combined legend for twin axis
lines_1, labels_1 = ax4.get_legend_handles_labels()
lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize=9.5)
ax4.set_title('Replay Memory Accumulation & Policy Convergence')

fig.suptitle('RoyaleRL — Deep Reinforcement Learning Agent Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
output_path = os.path.join(ROOT_DIR, 'presentation', '12_rl_training_performance.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Generated and saved: {output_path}")
