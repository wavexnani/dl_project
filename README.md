# ⚔️ Royale-RL: Autonomous Tactical Reinforcement Learning Agent

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%20%2F%2011-0078d7.svg)](https://microsoft.com/windows)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%20Supported-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced, production-grade autonomous agent and training pipeline for **Clash Royale**. **Royale-RL** integrates real-time computer vision, a **Reward-Conditioned Decision Transformer** policy network, a **Deterministic Tactical Counter-AI Rule Engine**, and low-latency Windows API driver controls to achieve autonomous 24/7 self-play, human demonstration recording, and continuous offline reinforcement learning.

---

## 🏛️ System Architecture

### Pipeline Architecture Overview

```mermaid
graph TD
    %% Professional Slate & Accent Theme
    classDef input fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc;
    classDef model fill:#1e293b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc;
    classDef synth fill:#1e293b,stroke:#34d399,stroke-width:1.5px,color:#f8fafc;
    classDef brain fill:#1e293b,stroke:#a78bfa,stroke-width:1.5px,color:#f8fafc;
    classDef rule fill:#1e293b,stroke:#fbbf24,stroke-width:1.5px,color:#f8fafc;
    classDef action fill:#1e293b,stroke:#22d3ee,stroke-width:1.5px,color:#f8fafc;
    classDef training fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc;

    subgraph S1["1. Perception Pipeline (Every Frame in RAM)"]
        BS["BlueStacks Game Screen (565x1007)"]:::input --> GRAB["ImageGrab.grab() (In-Memory PIL Frame)"]:::input
        GRAB --> CROP["Card Slot Cropper"]:::input
        GRAB --> FULL["Full Arena Frame (640x640)"]:::input
        GRAB --> OCR["Tower & Clock ROIs"]:::input
        GRAB --> ELX["Elixir Crop ROI"]:::input
        
        CROP --> M1["Model 1: MobileNetV2\n(Card Classifier)"]:::model
        FULL --> M2["Model 2: YOLOv9-c\n(Enemy Detector)"]:::model
        OCR --> M_OCR["EasyOCR Engine\n(Tower Health)"]:::model
        ELX --> M_ELX["Template Matching\n(Elixir Gauge)"]:::model
        
        M1 -->|"4 Cards: ['giant', ...]"| STATE["State Assembler\n(_flatten_state in agent.py)"]:::synth
        M2 -->|"20 Troop BBoxes: [x1,y1,x2,y2]"| STATE
        M_OCR -->|"6 Tower HP Deltas"| STATE
        M_ELX -->|"Elixir Value (0-10)"| STATE
    end

    subgraph S2["2. Mathematical Representation"]
        STATE --> VEC["Unified State Vector s_t\n(Length: 439 Floats)"]:::synth
        VEC --> HIST["Trajectory Sequence Queue\n(Context Window K = 10)"]:::synth
    end

    subgraph S3["3. Neuro-Symbolic Decision Engine"]
        HIST --> DT["Model 3: Decision Transformer\n(Conditioned on Target Return R = +2.0)"]:::brain
        DT --> CAND["Action Candidate Proposals\n(Card Slot + 18x30 Grid)"]:::brain
        
        CAND --> TB{"TacticalBrain Filter\n(Symbolic Rules)"}:::rule
        TB -->|"Passes: Anti-Air, Center-Pull, King Safe"| EXEC["Validated Action: (slot, x, y)"]:::action
        TB -->|"Fails Tactical Check"| NEXT_CAND["Evaluate Next Candidate"]:::rule
        NEXT_CAND --> TB
        TB -->|"All Rejected & Elixir >= 9"| RELIEF["Elixir Relief Valve\n(Drop Tank behind King)"]:::rule
        RELIEF --> EXEC
    end

    subgraph S4["4. Execution & Continuous Learning"]
        EXEC --> POST["Win32 PostMessage API\n(No Cursor Hijacking)"]:::action
        POST --> BS
        EXEC --> BUFF["Replay Buffer (RAM / replay_buffer.pkl)\n(s_t, a_t, r_t, s_t+1)"]:::training
        BUFF --> TRAIN["Reward-Weighted\nGradient Update (Loss)"]:::training
    end
```

### End-to-End Decision & Real-Time Tick Loop

```mermaid
flowchart TD
    %% Professional Slate & Accent Theme
    classDef capture fill:#1e293b,stroke:#38bdf8,stroke-width:1.5px,color:#f8fafc;
    classDef vision fill:#1e293b,stroke:#818cf8,stroke-width:1.5px,color:#f8fafc;
    classDef synth fill:#1e293b,stroke:#34d399,stroke-width:1.5px,color:#f8fafc;
    classDef brain fill:#1e293b,stroke:#fbbf24,stroke-width:1.5px,color:#f8fafc;
    classDef model fill:#1e293b,stroke:#a78bfa,stroke-width:1.5px,color:#f8fafc;
    classDef action fill:#1e293b,stroke:#22d3ee,stroke-width:1.5px,color:#f8fafc;
    classDef training fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#f8fafc;

    subgraph Loop["1. Real-Time Tick Loop (runbot.py ~350ms cycle)"]
        A["Screen Capture (ImageGrab)<br/>Resolution Scaled via Scaler"]:::capture
        GS["GameStateManager.get_state()<br/>MatchTemplate on Anchors"]:::capture
    end

    subgraph Perception["2. Vision & State Extraction (core/vision.py)"]
        B["Vision.get_game_state()"]:::vision
        
        B1["<b>MobileNetV2 Classifier</b><br/>Input: 4 cropped card slots (128x128)<br/>Output: Active Hand Cards"]:::vision
        B2["<b>YOLOv9 Detector</b><br/>Input: Arena RGB frame (640x640)<br/>Output: Enemy Bounding Boxes & Names"]:::vision
        B3["<b>EasyOCR Reader</b><br/>Input: 6 Tower HP regions + Timer<br/>Output: Digits & Health Percentages"]:::vision
        B4["<b>Elixir Tracker & Vision</b><br/>Tick Accrual + CV2 Template Sync<br/>Output: Float Elixir [0.0 - 10.0]"]:::vision
    end

    subgraph StateSynth["3. State Vector Synthesis (core/agent.py)"]
        S["Flatten State (439-Dim Vector)<br/>• Elixir (1)<br/>• 6 Towers HP% (6)<br/>• Hand One-Hot (4 x 88 = 352)<br/>• Top 20 Enemy BBoxes (80)"]:::synth
    end

    subgraph Arbiter["4. Neuro-Symbolic Decision Engine (core/agent.py & tactical_brain.py)"]
        TB_Mandatory{"TacticalBrain.get_mandatory_action()<br/>Any Critical Threat or Lethal Win?"}:::brain
        
        M_Spell["Rule 1: Spell-Snipe Finisher<br/>(Tower HP <= 280 / 140)"]:::brain
        M_Threat["Rule 3 & 14: Emergency Threat Defense<br/>(Center-Pull Kiting vs Approaching Push)"]:::brain
        M_Pocket["Rule 10: Pocket King Assault<br/>(Enemy Princess Tower Down)"]:::brain
        M_Leak["Rule 7: 10-Elixir Leak Prevention<br/>(Cycle Card Behind King)"]:::brain
        
        TacticalLock{"TacticalBrain.has_unresolved_threat()<br/>High Threat Approaching?"}:::brain
        HoldElixir["HOLD ELIXIR<br/>Do not squander elixir on attack"]:::brain

        Epsilon{"Random < Epsilon?<br/>(Exploration vs Exploitation)"}:::brain
        
        RandAction["Sample Random Playable Card & Grid"]:::brain
        
        DT_Inference["<b>Decision Transformer Model</b><br/>Input: State (439d), Last Action, Target Return (+2.0), Step<br/>Output: 48,060 Action Logits<br/>(Masked to Playable Cards in Hand)"]:::model
        
        Validate{"TacticalBrain.validate_candidate_action()<br/>Check Negative Constraints:<br/>• Wake King early?<br/>• Ground melee vs Air?<br/>• Naked bridge giant?<br/>• Attack during defense?"}:::brain
        
        CorrectAction["Corrected / Shifted Action<br/>(e.g., Safe Pullback, Lane Shift)"]:::brain
        RejectAction["Reject Action (None)"]:::brain
        ReliefValve["Elixir Relief Valve (Elixir >= 9.0)"]:::brain
    end

    subgraph Execution["5. Motor Output & Driver (drivers/controller.py)"]
        EXEC["Controller.play_card()<br/>1. Click Card Slot<br/>2. Click Placement Grid Coordinate"]:::action
        DEDUCT["ElixirTracker.deduct(cost)<br/>Instant Elixir Deduction"]:::action
    end

    subgraph PostMatch["6. Replay Buffer & Training (core/agent.py)"]
        BUFF["ReplayBuffer.add(s, a, r, s')<br/>Experience Logging"]:::training
        TRAIN["Decision Transformer Training<br/>Reward-Weighted Cross-Entropy<br/>weights = clamp(1 + r, 0.2, 3.0)"]:::training
    end

    %% Wiring
    A --> GS
    GS -->|IN_BATTLE| B
    B --> B1 & B2 & B3 & B4
    B1 & B2 & B3 & B4 --> S
    S --> TB_Mandatory

    TB_Mandatory -->|Triggered| M_Spell & M_Threat & M_Pocket & M_Leak
    M_Spell & M_Threat & M_Pocket & M_Leak --> EXEC

    TB_Mandatory -->|None| TacticalLock
    TacticalLock -->|Yes & Elixir < 9.5| HoldElixir
    TacticalLock -->|No| Epsilon

    Epsilon -->|True| RandAction
    Epsilon -->|False| DT_Inference

    RandAction --> Validate
    DT_Inference --> Validate

    Validate -->|Valid| EXEC
    Validate -->|Violates Rule| CorrectAction --> EXEC
    Validate -->|Invalid/Blocked| RejectAction --> ReliefValve
    ReliefValve -->|Elixir >= 9.0| EXEC

    EXEC --> DEDUCT
    EXEC --> BUFF
    BUFF -->|Match Ends| TRAIN
```

---

## ✨ Key Features

- **Hybrid Intelligence Architecture**:
  - **Decision Transformer (Offline RL)**: Sequence modeling architecture predicting high-reward card deployments conditioned on multi-step trajectory history and cumulative return.
  - **Tactical Counter Engine**: High-speed, deterministic tactical rules ensuring optimal plays:
    - *Center-Pull Kiting*: Pulls win conditions (Giant, Hog Rider) toward the center kill-zone (Tile 4-3).
    - *Anti-Air & Ground Validation*: Strictly prevents invalid interactions (e.g. dropping ground melee on flying units).
    - *Lethal Finish Snipes*: Guaranteed instant spell victory snipes on critical Crown Towers.
    - *Dormant King Tower Protection*: Prevents accidental early King Tower activations.
    - *Pocket Exploitation & Breach Pushes*: Automatic spearhead offensive deployments upon destroying lane towers.
    - *Elixir Leak Prevention & Tactical Lock*: Never wastes 10-elixir cap; queues hard-counters against approaching threats.
- **Real-Time Perception Suite**:
  - **YOLOv9**: Identifies and bounds enemy troops, buildings, and champions across both lanes.
  - **MobileNetV2**: Classifies current 4 cards in hand with dynamic elixir state normalization.
  - **EasyOCR + Template Matching**: Live extraction of King/Princess tower health and elixir count.
  - **Dynamic Scaler**: DPI-aware viewport tracking that auto-adapts to emulator repositioning.
- **Windows Native Control Driver**:
  - Direct Win32 `PostMessage` mouse clicks without stealing user cursor focus.
  - Integrated `PowerManager` keeping Windows awake 24/7 without screen sleeps or lockouts.
  - Automated watchdog for match resets, chest popups, and disconnection recovery.

---

## 📁 Repository Directory Structure

```text
Project_DL/
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Python dependencies (root-level convenience)
├── .gitignore                             # Ignored build, cache, and runtime artifacts
├── RoyaleRL-Architecture.html             # Interactive architecture exploration report
├── RoyaleRL-MatchWorkflow.html            # Visual match workflow and lifecycle viewer
├── royale_rl_architecture_1080x1720.png   # High-resolution architectural diagram
├── royale_rl_architecture.svg            # Scalable vector architecture diagram
└── RoyaleRL/
    ├── config.py                          # Global coordinates, card costs, path resolvers
    ├── requirements.txt                   # Pinned PyTorch CUDA 12.1 dependencies
    ├── runbot.py                          # Main orchestrator (Autonomous / Record / Train)
    │
    ├── core/                              # Core AI intelligence modules
    │   ├── agent.py                       # Decision Transformer model & replay memory
    │   ├── tactical_brain.py              # 27+ deterministic tactical counter rules
    │   └── vision.py                      # Multi-model perception pipeline
    │
    ├── drivers/                           # Hardware, OS, and emulator drivers
    │   ├── controller.py                  # Win32 PostMessage mouse click dispatcher
    │   ├── game_state_manager.py          # Screen state navigation & template matcher
    │   ├── power_manager.py               # Win32 execution state sleep prevention
    │   └── scaler.py                      # Dynamic viewport & DPI resolution scaler
    │
    ├── weights/                           # Model checkpoints & classifier vocabularies
    │   ├── class_names.txt                # Alphabetical card class names
    │   ├── enemy_boundary_detector.pt     # YOLOv9 enemy detection weights
    │   ├── hand_classifier_best.pth       # MobileNetV2 hand classification model
    │   ├── rl_agent.pt                    # Active Decision Transformer weights
    │   └── rl_agent_final.pt              # Checkpoint of mature trained model
    │
    ├── data/                              # Datasets, experiences, and logs
    │   ├── replay_buffer.pkl              # Replay buffer containing match trajectories
    │   ├── training_stats.json            # Historical loss, rewards, and win rates
    │   └── sorted_data/                   # Image crops for cards and elixir numbers
    │
    ├── training/                          # Model training scripts
    │   ├── human_recorder.py              # Hotkey listener for human demonstration data
    │   └── optimized_train.py             # MobileNetV2 card classifier training
    │
    ├── evaluation/                        # Analysis and reporting utilities
    │   ├── analyze_knowledge.py           # Evaluation of state embeddings
    │   ├── model_analysis.py              # Visualizations of model predictions
    │   ├── generate_rl_presentation_chart.py # Matplotlib charts of agent metrics
    │   └── rollback_last_match.py         # Utility to prune corrupted match records
    │
    ├── tests/                             # Automated test suites
    │   ├── test_tactical_rules.py         # 27+ tactical rules verification suite
    │   ├── test_all_audit_fixes.py        # Comprehensive regression and audit tests
    │   └── test_bug_fixes.py              # Targeted fixes verification
    │
    └── tools_debug/                       # Calibration and diagnostic tools
        ├── calibrate.py                   # Viewport coordinate alignment tool
        ├── debug_crowns.py                # Crown tower OCR diagnostics
        └── debug_scaler.py                # Visual debug tool for scaler bounds
```

---

## 💻 Windows Setup & Installation Guide

### 1. Prerequisites

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Python**: Version **3.10.x** or **3.11.x** (Ensure `python` and `pip` are added to your System PATH).
- **GPU**: NVIDIA GPU with CUDA 12.1+ capability recommended. *(CPU fallback is supported but will increase inference latency).*
- **Android Emulator**: **BlueStacks 5** (Pie 64-bit or Android 11 instance).

---

### 2. BlueStacks 5 Configuration (Mandatory)

To ensure the computer vision and coordinate scalers align with the game elements, configure BlueStacks as follows:

1. Open **BlueStacks 5 Settings** (`Ctrl + Shift + I` or click the Gear icon).
2. **Display Settings**:
   - **Display Resolution**: Select **Custom** or **Phone**. Set resolution to **`1080 x 1920`** (Portrait) or **`565 x 1007`**.
   - **Pixel Density**: Set to **`240 DPI (Medium)`** or **`320 DPI (High)`**.
   - **Interface Scaling**: Set to **`100% (Default)`**.
3. **Graphics Settings**:
   - **Graphics Engine Mode**: Compatibility or Performance.
   - **Graphics Renderer**: **`DirectX`** or **`OpenGL`**.
   - **Interface Renderer**: Auto.
   - **GPU in use**: Enable **Prefer dedicated GPU** (NVIDIA).
4. **Clash Royale Settings**:
   - Install Clash Royale from the Google Play Store.
   - Open Clash Royale, ensure the language is set to **English** (for OCR accuracy).
   - Keep the BlueStacks window visible on your primary monitor.

---

### 3. Clone & Environment Setup

Open PowerShell or Command Prompt (running as Administrator is recommended to allow low-level window click messages):

```powershell
# 1. Clone repository
git clone https://github.com/wavexnani/dl_project.git
cd dl_project

# 2. Create isolated virtual environment
python -m venv venv

# 3. Activate virtual environment
# In PowerShell:
.\venv\Scripts\Activate.ps1
# Or in Command Prompt (cmd):
.\venv\Scripts\activate.bat

# 4. Upgrade pip, wheel, and setuptools
python -m pip install --upgrade pip setuptools wheel

# 5. Install dependencies (includes PyTorch with CUDA 12.1 support)
pip install -r requirements.txt
```

---

### 4. Calibration & Diagnostic Check

Before starting autonomous matches, calibrate the viewport to verify BlueStacks window detection:

```powershell
python RoyaleRL/tools_debug/calibrate.py
```

This utility will locate the active BlueStacks window, measure bounding box offsets, and verify that tower health regions, elixir readings, and card slots align accurately.

To run the automated rule engine tests:

```powershell
python RoyaleRL/tests/test_tactical_rules.py
```

---

## 🚀 Running the Agent

### Mode 1: Autonomous 24/7 Self-Play (Default)

Runs the fully autonomous loop: detects matches, executes tactical defenses and learned attacks, records trajectories into the replay buffer, and handles post-battle navigation.

```powershell
python RoyaleRL/runbot.py --mode auto
```

*Note: The built-in `power_manager.py` prevents Windows from going to sleep or activating the lock screen while the bot is active.*

### Mode 2: Human Demonstration Recording

Play manually on BlueStacks while the recorder captures state-action pairs into the replay buffer to provide expert demonstration data for behavioral cloning and offline RL:

```powershell
python RoyaleRL/runbot.py --mode record
```

### Mode 3: Offline Replay Buffer Training

Trains the Decision Transformer policy network using historical match trajectories saved in `RoyaleRL/data/replay_buffer.pkl`:

```powershell
python RoyaleRL/runbot.py --mode train
```

---

## 🤝 Contributing & Community Collaboration

We welcome contributions from reinforcement learning researchers, computer vision engineers, Clash Royale enthusiasts, and competitive players!

### 🎯 Areas Where You Can Contribute

#### 1. 🧠 Training the Models & Enhancing Strategy
- **Replay Data Sharing**: Share your `replay_buffer.pkl` containing high-trophy or grand challenge match trajectories to enrich our dataset.
- **YOLOv9 Enemy Unit Dataset**: Collect arena screenshots containing new card evolutions, champions, and tower troops to retrain `enemy_boundary_detector.pt`.
- **Card Classifier Expansion**: Add new card images to `RoyaleRL/data/sorted_data/cards/<card-name>` and retrain the MobileNetV2 classifier via `RoyaleRL/training/optimized_train.py`.
- **Decision Transformer Tuning**: Experiment with context lengths (`context_len`), multi-head attention configurations, and reward functions (evaluating elixir advantage, tempo, and counter-push efficiency).

#### 2. 🛡️ Tactical Counter Rule Engine
- Implement new counter-play rules in `RoyaleRL/core/tactical_brain.py` (e.g., Tornado king activations, Miner placement predictions, Bridge Spam defense).
- Add corresponding unit tests in `RoyaleRL/tests/test_tactical_rules.py` to maintain 100% test suite reliability.

#### 3. 🐛 Bug Fixes & Emulator Portability
- Enhance OCR accuracy across different arena ground textures and lighting effects.
- Add support for other emulators (LDPlayer 9, MuMu Player Pro, NoxPlayer) in `RoyaleRL/drivers/scaler.py` and `RoyaleRL/drivers/controller.py`.
- Optimize memory usage and buffer serialization for multi-day continuous runs.

---

### 📝 Contribution Workflow

1. **Fork the Repository**: Click the **Fork** button at the top right of this repository.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/tactical-tornado-pull
   ```
3. **Write Code & Add Tests**:
   - Implement your changes.
   - If adding tactical rules, add unit test cases to `RoyaleRL/tests/test_tactical_rules.py`.
4. **Validate Test Suite**:
   ```bash
   python RoyaleRL/tests/test_tactical_rules.py
   python RoyaleRL/tests/test_all_audit_fixes.py
   python RoyaleRL/tests/test_bug_fixes.py
   ```
5. **Commit with Clear Messages**:
   ```bash
   git commit -m "feat(tactical): add tornado center pull rule for Hog Rider"
   ```
6. **Push and Open a Pull Request**:
   ```bash
   git push origin feature/tactical-tornado-pull
   ```
   Open a PR against the `main` branch with a clear description of your changes and benchmark results.

---

## 📜 Legal Disclaimer

This project is developed strictly for **educational, academic, and artificial intelligence research purposes**. Automating gameplay may violate the Terms of Service of Supercell and Clash Royale. The authors and maintainers do not encourage the use of this software for unfair advantage or commercial purposes. Use responsibly and at your own discretion.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
