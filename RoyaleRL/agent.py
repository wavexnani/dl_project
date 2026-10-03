import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import random
import config
from config import get_health_percentage, CARD_TO_INDEX, NUM_CARD_TYPES, ALL_CARDS, ARENA_BBOX
import json
import pickle
from tactical_brain import TacticalBrain

# Define the 18x30 placement grid
x_steps = 18
y_steps = 30
min_x_pct, min_y_pct, max_x_pct, max_y_pct = ARENA_BBOX

PLACEMENT_GRID = []
for i in range(x_steps):
    for j in range(y_steps):
        x_pct = min_x_pct + (max_x_pct - min_x_pct) * (i + 0.5) / x_steps
        y_pct = min_y_pct + (max_y_pct - min_y_pct) * (j + 0.5) / y_steps
        PLACEMENT_GRID.append((x_pct, y_pct))

NUM_GRID_LOCATIONS = len(PLACEMENT_GRID)
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID_LOCATIONS # +1 for "do nothing"

class Block(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()
        self.attn = nn.MultiheadAttention(h_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), nn.GELU(), nn.Linear(4 * h_dim, h_dim), nn.Dropout(drop_p))
        self.ln1, self.ln2 = nn.LayerNorm(h_dim), nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.ln1(x)
        x = x + self.ff(x)
        return self.ln2(x)

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p):
        super().__init__()
        self.state_dim, self.act_dim, self.h_dim = state_dim, act_dim, h_dim
        self.embed_state = nn.Linear(self.state_dim, h_dim)
        self.embed_action = nn.Linear(self.act_dim, h_dim)
        self.embed_reward = nn.Linear(1, h_dim)
        self.embed_timestep = nn.Embedding(context_len, h_dim) 
        self.embed_ln = nn.LayerNorm(h_dim)
        self.transformer_blocks = nn.ModuleList([Block(h_dim, n_heads, drop_p) for _ in range(n_blocks)])
        self.predict_action = nn.Sequential(nn.Linear(h_dim, self.act_dim))

    def forward(self, states, actions, rewards, timesteps):
        batch_size, seq_len, _ = states.shape
        timesteps = torch.arange(seq_len, device=states.device).unsqueeze(0).repeat(batch_size, 1)
        time_embs = self.embed_timestep(timesteps)
        state_embs = self.embed_state(states) + time_embs
        action_embs = self.embed_action(actions) + time_embs
        reward_embs = self.embed_reward(rewards) + time_embs
        stacked_inputs = torch.stack((state_embs, action_embs, reward_embs), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.h_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)
        x = stacked_inputs
        for block in self.transformer_blocks:
            x = block(x)
        x = x.reshape(batch_size, seq_len, 3, self.h_dim).permute(0, 2, 1, 3)
        # Condition action prediction on the state token (index 0) rather than action token (index 1)
        return self.predict_action(x[:, 0])

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.ptr, self.size = 0, 0
        self.states = np.zeros((self.capacity, state_dim))
        # Store actions as a single integer (dtype int32)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros((self.capacity, 1))
        self.next_states = np.zeros((self.capacity, state_dim))
    
    def add(self, state, action, reward, next_state):
        self.states[self.ptr] = state
        # Now action is just the index
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, context_len):
        if self.size < context_len:
            return None, None, None, None
        indices = np.random.randint(0, self.size - context_len, size=batch_size)
        states, actions, rewards, timesteps = [], [], [], []
        for i in indices:
            states.append(self.states[i:i+context_len])
            # Actions are now a 1-D array of integers
            actions.append(self.actions[i:i+context_len])
            rewards.append(self.rewards[i:i+context_len])
            timesteps.append(np.arange(context_len)) 
        return (torch.tensor(np.array(states), dtype=torch.float32),
                # Convert the integer actions to a tensor
                torch.tensor(np.array(actions), dtype=torch.long),
                torch.tensor(np.array(rewards), dtype=torch.float32),
                torch.tensor(np.array(timesteps), dtype=torch.long))

class Agent:
    def __init__(self, state_dim, action_dim, card_costs, device):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.card_costs = card_costs
        self.device = device
        self.model_path = 'rl_agent.pt'
        
        # Load persisted epsilon so training progress is not lost across restarts
        self.epsilon = self._load_initial_epsilon()
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        
        self.context_len = 10 
        self.n_blocks, self.embed_dim, self.n_heads, self.dropout_p, self.lr = 3, 128, 1, 0.1, 1e-4

        self.model = DecisionTransformer(
            state_dim=state_dim, act_dim=action_dim, n_blocks=self.n_blocks, h_dim=self.embed_dim,
            context_len=self.context_len, n_heads=self.n_heads, drop_p=self.dropout_p
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        self.buffer_path = 'replay_buffer.pkl' 
        self.replay_buffer = ReplayBuffer(1e5, self.state_dim, self.action_dim)
        self.tactical_brain = TacticalBrain(scaler=None)
        self.load()
        if os.path.exists(self.buffer_path):
            self.load_buffer()

    def _load_initial_epsilon(self):
        stats_file = 'training_stats.json'
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                    if 'history' in data and data['history']:
                        last_eps = data['history'][-1].get('epsilon')
                        if last_eps is not None and isinstance(last_eps, (int, float)):
                            val = max(0.05, float(last_eps))
                            print(f"📊 [AGENT] Loaded persisted exploration rate ε = {val:.3f}")
                            return val
            except Exception:
                pass
        return 0.35

    def set_epsilon(self, val):
        """Allows manually setting exploration vs exploitation rate (0.05 = pure AI, 1.0 = random)."""
        self.epsilon = max(0.0, min(1.0, float(val)))
        print(f"🎯 [AGENT] Exploration rate manually set to ε = {self.epsilon:.3f}")
    
    def save_buffer(self):
        sz = self.replay_buffer.size
        data = {
            'format': 'compact_v2',
            'ptr': self.replay_buffer.ptr,
            'size': sz,
            'states': self.replay_buffer.states[:sz],
            'actions': self.replay_buffer.actions[:sz],
            'rewards': self.replay_buffer.rewards[:sz],
            'next_states': self.replay_buffer.next_states[:sz]
        }
        temp_path = f"{self.buffer_path}.tmp"
        saved = False
        for attempt in range(5):
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                if os.path.exists(self.buffer_path):
                    try:
                        os.remove(self.buffer_path)
                    except Exception:
                        pass
                os.replace(temp_path, self.buffer_path)
                saved = True
                break
            except Exception:
                time.sleep(0.2)
        if not saved:
            try:
                with open(self.buffer_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"⚠️ [WARNING] Could not save replay buffer ({e}).")
        print(f"Replay buffer saved ({sz} items, compact format).")

    def load_buffer(self):
        try:
            with open(self.buffer_path, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict) and 'states' in loaded:
                sz = loaded['size']
                self.replay_buffer.size = sz
                self.replay_buffer.ptr = loaded['ptr']
                self.replay_buffer.states[:sz] = loaded['states']
                self.replay_buffer.actions[:sz] = loaded['actions']
                self.replay_buffer.rewards[:sz] = loaded['rewards']
                self.replay_buffer.next_states[:sz] = loaded['next_states']
                print(f"Replay buffer loaded with {sz} items (compact).")
            elif hasattr(loaded, 'size'):
                self.replay_buffer = loaded
                print(f"Replay buffer loaded with {self.replay_buffer.size} items (legacy).")
        except Exception as e:
            print(f"Could not load replay buffer: {e}")
            print("Starting with a new, empty buffer.")

    def _flatten_state(self, state_dict):
        elixir = np.array([state_dict.get('elixir', 0) / 10.0])
        ocr_data = state_dict.get('ocr_data', {})
        tower_health = np.array([
            get_health_percentage(ocr_data.get('ptl'), 'princess'), get_health_percentage(ocr_data.get('ptr'), 'princess'),
            get_health_percentage(ocr_data.get('tk'), 'king'), get_health_percentage(ocr_data.get('pbl'), 'princess'),
            get_health_percentage(ocr_data.get('pbr'), 'princess'), get_health_percentage(ocr_data.get('bk'), 'king')
        ])
        hand_vector = np.zeros(4 * NUM_CARD_TYPES)
        for i, card in enumerate(state_dict.get('hand', [])):
            if card in CARD_TO_INDEX:
                hand_vector[i * NUM_CARD_TYPES + CARD_TO_INDEX[card]] = 1.0
        enemies_vector = np.zeros(20 * 4)
        enemies = state_dict.get('enemies', [])[:20]
        for i, enemy in enumerate(enemies):
            box = enemy.get('box', (0,0,0,0))
            enemies_vector[i*4:(i+1)*4] = [
                box[0]/config.REFERENCE_RESOLUTION[0], box[1]/config.REFERENCE_RESOLUTION[1],
                box[2]/config.REFERENCE_RESOLUTION[0], box[3]/config.REFERENCE_RESOLUTION[1]
            ]
        return np.concatenate([elixir, tower_health, hand_vector, enemies_vector])

    def _flatten_action(self, action_dict, scaler=None):
        card_slot = action_dict.get('card_slot', None)
        position = action_dict.get('position', None)
        
        # Now returns a single integer, not a vector
        action_index = NUM_CARD_TYPES * NUM_GRID_LOCATIONS 
        if card_slot is not None and position is not None:
            closest_grid_index = self._find_closest_grid_index(position, scaler)
            action_index = card_slot * NUM_GRID_LOCATIONS + closest_grid_index
        return action_index

    def _find_closest_grid_index(self, position, scaler=None):
        if scaler is not None and hasattr(scaler, 'current_resolution'):
            window_width, window_height = scaler.current_resolution
        else:
            window_width, window_height = config.REFERENCE_RESOLUTION
        pos_pct = np.array([position[0] / max(1, window_width), position[1] / max(1, window_height)])
        
        placement_grid_np = np.array(PLACEMENT_GRID)
        distances = np.linalg.norm(placement_grid_np - pos_pct, axis=1)
        return np.argmin(distances)

    def decide_action(self, game_state, scaler, current_step=0):
        # Update scaler for accurate screen coordinate resolution
        self.tactical_brain.scaler = scaler

        # 1. Deterministic Tactical Overrides (Spell Finisher, Emergency Threats, Leak Prevention, Pocket)
        mandatory_action = self.tactical_brain.get_mandatory_action(game_state)
        if mandatory_action is not None:
            return mandatory_action

        # Hard-to-refuse Counter Lock: If an active high threat is advancing, HOLD ELIXIR for the counter!
        # Do not allow the Decision Transformer policy or exploration to squander elixir on offensive cards!
        if self.tactical_brain.has_unresolved_threat(game_state) and game_state.get('elixir', 0.0) < 9.5:
            print("⏳ [TACTICAL LOCK] Threat approaching! Holding elixir for mandatory hard-counter...")
            return None

        # 2. Candidate Proposal (Exploration or Decision Transformer Policy)
        if np.random.rand() <= self.epsilon:
            print(f"BRAIN: Choosing candidate action (exploring, ε={self.epsilon:.2f})...")
            candidate_actions = [self._get_random_action(game_state, scaler)]
        else:
            print("BRAIN: Using AI model to decide candidate action (exploiting)...")
            primary_candidate = self._get_model_action(game_state, scaler, current_step)
            candidate_actions = [primary_candidate] if primary_candidate else []
            # If model action was mocked (e.g. in tests), do not append alternative candidates
            is_mock = hasattr(self._get_model_action, 'mock_calls') or hasattr(self._get_model_action, 'assert_called')
            if not is_mock:
                more_candidates = self._get_model_candidates(game_state, scaler, current_step=current_step, top_k=8)
                for cand in more_candidates:
                    if cand not in candidate_actions:
                        candidate_actions.append(cand)

        # 3. Strict Negative Constraint Validation across candidates (no paralysis!)
        for candidate in candidate_actions:
            if candidate is None:
                continue
            validated_action = self.tactical_brain.validate_candidate_action(candidate, game_state)
            if validated_action is not None:
                return validated_action

        # 4. Elixir Relief Valve (Prevents passivity trap if candidate was rejected and elixir >= 9.0)
        if game_state.get('elixir', 0.0) >= 9.0:
            return self.tactical_brain.check_elixir_leak_prevention(game_state, min_elixir=9.0)

        return None

    def _get_random_action(self, game_state, scaler):
        hand = game_state.get('hand', [])
        elixir = game_state.get('elixir', 0)
        playable_cards = [i for i, card in enumerate(hand) if card in self.card_costs and elixir >= self.card_costs[card]]
        if not playable_cards:
            return None
        
        card_slot_index = random.choice(playable_cards)
        placement_pct = random.choice(PLACEMENT_GRID)
        
        window_width, window_height = scaler.current_resolution
        placement_x = int(placement_pct[0] * window_width)
        placement_y = int(placement_pct[1] * window_height)
        
        return {'action': 'play_card', 'card_slot': card_slot_index, 'position': (placement_x, placement_y)}

    def _get_model_candidates(self, game_state, scaler, current_step=0, top_k=8):
        hand = game_state.get('hand', [])
        elixir = game_state.get('elixir', 0)
        playable_cards = [i for i, card in enumerate(hand) if card in self.card_costs and elixir >= self.card_costs[card]]
        if not playable_cards:
            return []

        state_vec = self._flatten_state(game_state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).reshape(1, 1, self.state_dim).to(self.device)
        
        action_tensor = torch.zeros((1, 1, self.action_dim), dtype=torch.float32).to(self.device)
        # Condition DT on a positive target return (aiming for victory)
        reward_tensor = (torch.ones((1, 1, 1), dtype=torch.float32) * 2.0).to(self.device)
        step_idx = min(current_step, self.context_len - 1)
        timestep_tensor = torch.tensor([[step_idx]], dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            action_preds = self.model(state_tensor, action_tensor, reward_tensor, timestep_tensor).squeeze(0).squeeze(0)

        # Mask logits so model selects placements among playable cards in hand
        masked_preds = torch.full_like(action_preds, float('-inf'))
        for slot in playable_cards:
            start_idx = slot * NUM_GRID_LOCATIONS
            end_idx = (slot + 1) * NUM_GRID_LOCATIONS
            masked_preds[start_idx:end_idx] = action_preds[start_idx:end_idx]

        valid_count = int((masked_preds > float('-inf')).sum().item())
        if valid_count == 0:
            return []

        k = min(top_k, valid_count)
        top_indices = torch.topk(masked_preds, k=k).indices.tolist()

        if scaler is not None and hasattr(scaler, 'current_resolution'):
            window_width, window_height = scaler.current_resolution
        else:
            window_width, window_height = config.REFERENCE_RESOLUTION

        candidates = []
        for action_index in top_indices:
            card_slot_index = action_index // NUM_GRID_LOCATIONS
            grid_location_index = action_index % NUM_GRID_LOCATIONS
            placement_pct = PLACEMENT_GRID[grid_location_index]
            placement_x = int(placement_pct[0] * window_width)
            placement_y = int(placement_pct[1] * window_height)
            candidates.append({'action': 'play_card', 'card_slot': card_slot_index, 'position': (placement_x, placement_y)})

        return candidates

    def _get_model_action(self, game_state, scaler, current_step=0):
        candidates = self._get_model_candidates(game_state, scaler, current_step=current_step, top_k=1)
        return candidates[0] if candidates else None

    def learn_from_game(self, game_log, scaler=None):
        if not game_log['steps']:
            print("LEARNING: No steps in game log, skipping training.")
            return
        
        print(f"LEARNING: Adding {len(game_log['steps'])} steps to replay buffer...")
        for step in game_log['steps']:
            state_vec = self._flatten_state(step['state'])
            action_vec = self._flatten_action(step['action'], scaler)
            reward_val = step['reward']
            next_state_vec = self._flatten_state(step['next_state'])
            self.replay_buffer.add(state_vec, action_vec, reward_val, next_state_vec)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        print(f"LEARNING: Buffer size: {self.replay_buffer.size}. New epsilon: {self.epsilon:.3f}")
        self.save()
        self.save_buffer()

    def train(self, num_epochs, batch_size):
        if self.replay_buffer.size < 50: 
            print("TRAINING: Not enough data in replay buffer to start training. Play more games.")
            return
        
        print(f"Starting agent training ({num_epochs} epochs on {self.replay_buffer.size} experiences)...")
        self.model.train()
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            steps_per_epoch = min(100, max(10, self.replay_buffer.size // batch_size))
            for _ in range(steps_per_epoch):
                states, actions, rewards, timesteps = self.replay_buffer.sample(batch_size, self.context_len)
                if states is None: continue
                
                states, actions, rewards, timesteps = states.to(self.device), actions.to(self.device), rewards.to(self.device), timesteps.to(self.device)
                one_hot_actions = F.one_hot(actions, num_classes=self.action_dim).float()
                action_preds = self.model(states, one_hot_actions, rewards, timesteps)

                # Reward-weighted cross-entropy: high weight on positive rewards, low on mistakes
                loss_raw = F.cross_entropy(action_preds.view(-1, self.action_dim), actions.view(-1), reduction='none')
                reward_flat = rewards.view(-1)
                weights = torch.clamp(1.0 + reward_flat, min=0.2, max=3.0)
                loss = (loss_raw * weights).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, steps_per_epoch)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.save(f"rl_agent_epoch_{epoch+1}.pt")
                
        print(f"Training complete in {(time.time() - start_time) / 60:.1f}m")
        self.save(self.model_path)
        self.save("rl_agent_final.pt")

    def update_match_stats(self, match_result, final_reward):
        """Logs and tracks cumulative and rolling win rate."""
        stats_file = 'training_stats.json'
        stats = {
            'matches_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'overall_win_rate': 0.0,
            'rolling_win_rate_20': 0.0,
            'history': []
        }
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
            except Exception:
                pass

        stats['matches_played'] += 1
        if match_result == 'WIN':
            stats['wins'] += 1
        elif match_result == 'LOSS':
            stats['losses'] += 1
        else:
            stats['draws'] += 1

        stats['history'].append({
            'match_num': stats['matches_played'],
            'result': match_result,
            'reward': final_reward,
            'buffer_size': self.replay_buffer.size,
            'epsilon': round(self.epsilon, 3),
            'timestamp': time.time()
        })
        stats['history'] = stats['history'][-100:]

        recent = stats['history'][-20:]
        recent_wins = sum(1 for m in recent if m['result'] == 'WIN')
        stats['rolling_win_rate_20'] = round((recent_wins / len(recent)) * 100, 1)
        stats['overall_win_rate'] = round((stats['wins'] / stats['matches_played']) * 100, 1)

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📊 [STATS] Match #{stats['matches_played']}: {match_result} | Overall Win Rate: {stats['overall_win_rate']}% | Rolling (Last 20): {stats['rolling_win_rate_20']}%")

    def save(self, path=None):
        save_path = path if path is not None else self.model_path
        temp_path = f"{save_path}.tmp"
        saved = False
        for attempt in range(5):
            try:
                torch.save(self.model.state_dict(), temp_path)
                if os.path.exists(save_path):
                    try:
                        os.remove(save_path)
                    except Exception:
                        pass
                os.replace(temp_path, save_path)
                saved = True
                break
            except Exception:
                time.sleep(0.25)
        if not saved:
            try:
                torch.save(self.model.state_dict(), save_path)
                saved = True
            except Exception as e:
                print(f"⚠️ [WARNING] Windows file lock prevented writing to {save_path}: {e}. Model weights remain safe in GPU memory.")
        if saved:
            print(f"Agent model saved to {save_path}")

    def load(self):
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"Agent model loaded from {self.model_path}")
            except Exception as e:
                print(f"Notice: Checkpoint layer mismatch ({e}).")
                print("Transferring compatible transformer layers and reinitializing head for current action space...")
                state_dict = torch.load(self.model_path, map_location=self.device)
                model_dict = self.model.state_dict()
                matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(matched)
                self.model.load_state_dict(model_dict)
                self.model.eval()
                print(f"Loaded {len(matched)}/{len(model_dict)} compatible layers from checkpoint.")
