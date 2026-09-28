# Clash Royale AI — Autonomous 24/7 Reinforcement Learning & Self-Play System

An autonomous reinforcement learning system that plays and learns Supercell's *Clash Royale* in real-time on an emulator. The bot combines real-time computer vision (YOLOv9, MobileNetV2, and EasyOCR) to perceive the arena state and a **Decision Transformer** model that continuously learns optimal card deployments and macro-strategy through online self-play and offline replay buffer training.

---

## Key Features

* **Autonomous 24/7 Self-Play Loop**: Automatically starts matches from the main menu, navigates battles, evaluates outcomes, dismisses popups with a built-in watchdog, and loops indefinitely.
* **Decision Transformer RL Agent**: Sequence-modeling architecture predicting strategic card deployments across an $18 \times 30$ arena grid (`540` discrete placement zones) conditioned on state history, actions, and target rewards.
* **Continuous Online Learning**: Every played match automatically appends state transitions to a persistent replay buffer and triggers immediate reward-weighted policy optimization.
* **Multi-Modal Vision Pipeline**:
  * **YOLOv9** (`enemy_boundary_detector.pt`) for real-time enemy troop detection and bounding-box tracking.
  * **MobileNetV2** (`hand_classifier_best.pth`) for 4-slot hand card classification.
  * **EasyOCR & Template Matching** for tower health tracking, elixir counting, clock timers, and post-match victory/crown evaluation.
* **Human Teacher Demonstration Mode**: Allows human players to play manually (via mouse or keyboard 1–4) to record expert demonstrations directly into the replay buffer for imitation learning.
* **24/7 Power Management**: Built-in Windows sleep and lock prevention ([`power_manager.py`](RoyaleRL/power_manager.py)) to ensure uninterrupted long-term training sessions.
* **Performance Tracking & Analytics**: Automatic logging of win rates, match outcomes, and rewards in `training_stats.json`, with visual dashboard generators in `model_analysis.py`.

---

## System Architecture

The project is structured under the [`RoyaleRL/`](RoyaleRL/) directory:

```
dl_project/
├── README.md                      # Project documentation
├── runbot.bat                     # Windows 1-click launcher menu
└── RoyaleRL/
    ├── runbot.py                  # Main orchestrator & battle loop
    ├── agent.py                   # Decision Transformer & ReplayBuffer
    ├── vision.py                  # Vision suite (YOLOv9, MobileNetV2, EasyOCR)
    ├── game_state_manager.py      # UI navigation, crown detection & match outcome
    ├── controller.py              # Win32 PostMessage & PyAutoGUI click dispatcher
    ├── scaler.py                  # Resolution scaler & game area detector
    ├── power_manager.py           # Windows display & sleep prevention
    ├── human_recorder.py          # Real-time human demonstration listener
    ├── model_analysis.py          # Dashboard and performance chart generator
    ├── config.py                  # Grid, OCR offsets, and card cost definitions
    ├── requirements.txt           # Python package dependencies
    │
    ├── rl_agent.pt                # Primary trained Decision Transformer weights
    ├── rl_agent_final.pt          # Backup / final trained checkpoint
    ├── replay_buffer.pkl          # Compact persistent experience replay buffer
    ├── training_stats.json        # Match logs, cumulative and rolling win rates
    ├── enemy_boundary_detector.pt # Trained YOLOv9 unit detector weights
    ├── hand_classifier_best.pth   # Trained MobileNetV2 hand classifier weights
    ├── class_names.txt            # List of 11 recognized card classes
    └── sorted_data/               # UI anchors and elixir templates
```

### How the Learning Loop Works

1. **Perception**: At each step during battle, [`vision.py`](RoyaleRL/vision.py) reads the screen, identifies the cards in hand, checks elixir levels, registers tower hitpoints via OCR, and tracks enemy troop positions with YOLOv9.
2. **Decision**: [`agent.py`](RoyaleRL/agent.py) flattens this game state into a multi-modal feature vector. The Decision Transformer evaluates playable cards and arena coordinates, outputting the best action (or exploring based on current $\epsilon$).
3. **Execution**: [`controller.py`](RoyaleRL/controller.py) selects the card slot and clicks the target arena coordinate (clamped within valid arena boundaries).
4. **Outcome & Reward Evaluation**: When the battle concludes, [`game_state_manager.py`](RoyaleRL/game_state_manager.py) analyzes the end screen (victory/defeat banners and crown count) with fallback to tower damage deltas.
5. **Buffer Update & Policy Retraining**: The entire trajectory is committed to `replay_buffer.pkl`. The model immediately runs an online training step (default 5 epochs) with reward-weighted cross-entropy loss, updates `rl_agent.pt`, decays $\epsilon$, and updates `training_stats.json`.

---

## Setup & Installation

### Prerequisites

* **OS**: Windows 10/11 (required for BlueStacks emulator interaction and Win32 automation).
* **Python**: Python 3.10 to 3.12.
* **GPU**: NVIDIA GPU with CUDA support (CUDA 12.1 recommended for PyTorch models).
* **Emulator**: [BlueStacks 5](https://www.bluestacks.com/) with Clash Royale installed.

### 1. Clone the Repository & Switch to the Training Branch

Clone the repository and ensure you are on the `autonomous-24-7-selfplay` branch, which contains the complete self-play pipeline, checkpoints, and replay buffer:

```bash
git clone https://github.com/stanly363/Royale-RL-A-Reinforcement-Learning-Agent.git dl_project
cd dl_project
git checkout autonomous-24-7-selfplay
```

### 2. Create and Activate a Python Virtual Environment

Create a virtual environment named `venv` in the project root:

* **Windows (Command Prompt / PowerShell)**:
  ```cmd
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **Linux / WSL** (for development / analysis):
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the pinned dependencies (including CUDA-enabled PyTorch, TorchVision, YOLO/Ultralytics, OpenCV, and EasyOCR):

```bash
pip install -r RoyaleRL/requirements.txt
```

> **Note**: If you plan to use Human Teacher Mode (`--mode record`), ensure `pynput` is installed:
> ```bash
> pip install pynput
> ```

### 4. Configure BlueStacks & Clash Royale

1. Launch BlueStacks and open **Clash Royale**.
2. Set BlueStacks to **Fullscreen** (or Borderless Windowed).
3. Ensure the window title contains `BlueStacks App Player` (the default title looked up by [`scaler.py`](RoyaleRL/scaler.py)).
4. **Deck Setup**: The bot's vision system is trained on the following 11 cards out of the box (defined in [`class_names.txt`](RoyaleRL/class_names.txt)):
   * `archers`, `arrows`, `fireball`, `giant`, `goblin_cage`, `goblin_hut`, `knight`, `mini-pekka`, `minions`, `musketeer`.
   * For optimal performance, equip an 8-card deck selected from this pool.
5. Navigate to the Clash Royale **Main Menu** (the screen displaying the yellow "Battle" button).

---

## Starting and Continuing Training

All commands must be run from inside the `RoyaleRL/` directory (or via the root launcher script `runbot.bat`).

### Quick Start (Windows Launcher)

Double-click `runbot.bat` in the project root to open an interactive menu:

```text
===================================================
       CLASH ROYALE AI - 24/7 LAUNCHER
===================================================
 [1] Autonomous 24/7 Bot (Auto Self-Play)
 [2] Human Teacher Mode (Record your games to buffer)
 [3] Offline Train on Replay Buffer (50 epochs)
===================================================
```

---

### Mode 1: Autonomous 24/7 Self-Play Training (Recommended)

This is the primary operational mode. The bot runs continuously, playing matches, accumulating experiences, and training after every game.

```bash
cd RoyaleRL
python runbot.py --mode auto
```

#### What Happens During Autonomous Self-Play:
* **Automatic Matchmaking**: Clicks "Battle" from the main menu, waits for matchmaking, and detects when combat begins.
* **Real-Time Strategy**: Plays cards according to the Decision Transformer model with $\epsilon$-greedy exploration.
* **Match Conclusion**: Detects victory or defeat banners via OCR, calculates rewards based on crown differential and tower damage, and dismisses post-match rewards.
* **Online Model Updates**: Automatically appends the game's steps to `replay_buffer.pkl`, trains the model for 5 epochs on the buffer, saves updated weights to `rl_agent.pt`, and logs statistics to `training_stats.json`.
* **Self-Healing Watchdog**: If a screen hangs or an unknown popup appears, the watchdog triggers automated escape key and dismissal sequences to resume self-play.

#### Optional Arguments:
* `--epochs <int>`: Number of training epochs to execute after each game (default: `5`).
* `--batch_size <int>`: Training batch size (default: `64`).
* `--games <int>`: Stop after a fixed number of games (default: `0` for infinite 24/7 play).

Example:
```bash
python runbot.py --mode auto --epochs 5 --games 50
```

---

### Mode 2: Human Teacher Demonstration Recording

If you want to train the model on human gameplay and expert strategies (Imitation Learning):

```bash
cd RoyaleRL
python runbot.py --mode record
```

1. The bot navigates to matches as normal, but pauses autonomous clicking.
2. **You play manually**: Select cards by clicking them (or pressing keyboard keys `1`, `2`, `3`, or `4`), then click or drag onto the arena to deploy.
3. Every human move is tagged with `is_human: True` and saved directly into `replay_buffer.pkl` with full game state and reward attribution.
4. At the end of the match, the agent runs training epochs on the buffer, absorbing human play into the model weights.

---

### Mode 3: Dedicated Offline Training

You can train the Decision Transformer directly from your accumulated replay buffer without opening the emulator:

```bash
cd RoyaleRL
python runbot.py --mode train --epochs 50 --batch_size 64
```

* Loads existing experiences from `replay_buffer.pkl`.
* Loads existing weights from `rl_agent.pt`.
* Trains for 50 epochs (saving checkpoints every 10 epochs as `rl_agent_epoch_<N>.pt`).
* Updates `rl_agent.pt` and `rl_agent_final.pt` upon completion.

---

## Checkpoint & Persistence Mechanics

### Where Checkpoints and Experiences Are Stored

All model checkpoints, training data, and metrics are located directly in [`RoyaleRL/`](RoyaleRL/):

| File | Purpose |
| :--- | :--- |
| `rl_agent.pt` | **Primary Decision Transformer checkpoint**. Loaded automatically upon bot startup. |
| `rl_agent_final.pt` | Secondary/backup checkpoint updated at the conclusion of training runs. |
| `replay_buffer.pkl` | **Replay memory file**. Stores all collected game steps, state vectors, actions, and rewards. |
| `training_stats.json` | JSON database containing match history, rewards, overall win rate, and rolling win rate. |
| `enemy_boundary_detector.pt` | Pre-trained YOLOv9 enemy detection model. |
| `hand_classifier_best.pth` | Pre-trained MobileNetV2 hand card classification model. |

### How Continuing / Resuming Works

> [!IMPORTANT]
> **Continuing training is fully automatic.** Whenever you start `runbot.py`, the `Agent` class automatically inspects the directory:
> 1. If `rl_agent.pt` exists, it loads the saved weights directly into the Decision Transformer.
> 2. If `replay_buffer.pkl` exists, it loads the existing replay experiences into memory.
>
> You do **not** need a separate `--resume` command. Running `python runbot.py --mode auto` or `python runbot.py --mode train` will always pick up from the existing trained state.

### How to Resume After Stopping or Rebooting

1. You can stop the bot at any time by pressing `Ctrl + C` in the console. The bot captures the interrupt and cleanly writes all weights and buffer data to disk before terminating.
2. If your computer reboots or the process stops:
   * Open BlueStacks and launch Clash Royale (leave on Main Menu).
   * Activate your virtual environment (`.\venv\Scripts\activate`).
   * Navigate to `cd RoyaleRL`.
   * Run `python runbot.py --mode auto`.
   * The bot will print:
     ```text
     Agent model loaded from rl_agent.pt
     Replay buffer loaded with X items (compact).
     ```
   * Training seamlessly resumes from match `X + 1`.

### Avoiding Accidental Resets

* **DO NOT delete or move** `rl_agent.pt` or `replay_buffer.pkl`. If these files are absent, the agent will initialize with random weights and an empty memory buffer.
* **If you wish to start fresh intentionally**: First create a backup of your existing checkpoint files:
  ```bash
  copy rl_agent.pt rl_agent_backup.pt
  copy replay_buffer.pkl replay_buffer_backup.pkl
  copy training_stats.json training_stats_backup.json
  ```
  Only then delete `rl_agent.pt`, `replay_buffer.pkl`, and `training_stats.json`.

---

## Verifying Training Progress

You can verify that the model is actively learning and improving through several indicators:

### 1. Real-Time Console Outputs

During and after each game, the console outputs:

* **Buffer Growth**:
  ```text
  LEARNING: Adding 28 steps to replay buffer...
  LEARNING: Buffer size: 845. New epsilon: 0.941
  ```
* **Training Loss**:
  ```text
  Starting agent training (5 epochs on 845 experiences)...
  Epoch 1/5 | Loss: 2.1432
  Epoch 5/5 | Loss: 1.8720
  ```
* **Performance & Win Rate**:
  ```text
  📊 [STATS] Match #30: WIN | Overall Win Rate: 60.0% | Rolling (Last 20): 65.0%
  ```

### 2. Match Statistics File

Inspect [`RoyaleRL/training_stats.json`](RoyaleRL/training_stats.json) to view cumulative progress:
* `matches_played`: Total completed battles.
* `overall_win_rate`: Percentage of wins across all logged matches.
* `rolling_win_rate_20`: Win rate over the most recent 20 matches (a key metric for evaluating whether recent policy updates are translating into higher performance).
* `history`: Detailed log of individual match results, rewards, epsilon values, and buffer sizes.

### 3. Generate Analytical Dashboards

Run the analysis script to generate publication-grade diagnostic plots in [`RoyaleRL/presentation/`](RoyaleRL/presentation/):

```bash
cd RoyaleRL
python model_analysis.py
```

Generated charts include:
* `1_confusion_matrix.png` & `2_per_class_accuracy.png`: Vision classifier accuracy across card types.
* `8_decision_transformer_analysis.png`: Decision Transformer loss curves and token distributions.
* `9_action_space.png`: Heatmap of card placement density across the $18 \times 30$ arena grid.
* `10_system_overview.png`: Visual diagram of the multi-agent vision-action architecture.

---

## Adding New Cards

`config.py` contains elixir costs for all cards in Clash Royale. To train the bot to recognize cards beyond the default 11:

1. **Capture Images**: Run `python Mischelaneous/add_new_cards_images.py` to capture screenshots of new cards in your hand under varying elixir levels.
2. **Organize Folders**: Create a folder in `sorted_data/cards/<card_name>` using the exact name from `config.py` and place the captured crops there.
3. **Retrain Classifier**: Run `python optimized_train.py` to retrain the MobileNetV2 model.
4. **Alphabetize `class_names.txt`**: Ensure the updated `class_names.txt` has card names sorted in strict alphabetical order.

---

## Troubleshooting

* **`ValueError: BlueStacks window not found`**:
  * Ensure BlueStacks is running and the window title contains `BlueStacks App Player`.
  * Ensure BlueStacks is on your primary monitor and is not minimized.
* **`pygetwindow.PyGetWindowException` / Permissions**:
  * Run your Command Prompt / terminal as **Administrator**.
* **Model Layer Mismatch Notice**:
  * If you see `Notice: Checkpoint layer mismatch... Transferring compatible transformer layers`:
  * This is a built-in compatibility feature in [`agent.py`](RoyaleRL/agent.py). It transfers compatible transformer layers and re-initializes only the classification head when modifying action spaces.
* **Watchdog Screen Recovery**:
  * If the bot enters an unknown sub-menu or an event popup appears, allow up to 10 seconds. The watchdog will automatically send Android Escape and click center coordinates to clear the screen.

---

## License & Disclaimer

This project is developed for educational and machine learning research purposes. Automating gameplay may violate game Terms of Service. Use responsibly.
