# filename: runbot.py
"""
=================================================================
  Clash Royale AI — Autonomous 24/7 Bot & Training Pipeline
=================================================================
Modes:
  1. python runbot.py --mode auto     (24/7 autonomous self-play)
  2. python runbot.py --mode record   (Human teacher demonstration mode)
  3. python runbot.py --mode train    (Offline replay buffer training)

Features:
  - Windows sleep & screen lock prevention (Active 24/7)
  - Watchdog & popup auto-dismissal
  - Reward-weighted Decision Transformer learning
  - Rolling win-rate & performance tracker
=================================================================
"""

import os
import sys
import time
import cv2
import json
import argparse
import torch
import numpy as np
from PIL import ImageGrab

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
for sub in ["core", "drivers", "training", "evaluation"]:
    sub_path = os.path.join(BASE_DIR, sub)
    if sub_path not in sys.path:
        sys.path.insert(0, sub_path)

from drivers.scaler import Scaler
from drivers.game_state_manager import GameStateManager
from drivers.controller import Controller
from core.vision import Vision
from core.agent import Agent
from drivers.power_manager import keep_awake
from training.human_recorder import HumanRecorder
import config
from config import get_health_percentage, CARD_TO_INDEX, NUM_CARD_TYPES, CARD_COSTS, ALL_CARDS, STATE_DIM

x_steps = 18
y_steps = 30
NUM_GRID_LOCATIONS = x_steps * y_steps
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID_LOCATIONS

def find_cards_dynamically(scaler):
    game_area = scaler.game_area_rect
    gx1 = max(0, game_area[0])
    gy1 = max(0, game_area[1])
    gx2 = max(gx1 + 10, game_area[0] + game_area[2])
    gy2 = max(gy1 + 10, game_area[1] + game_area[3])
    bbox = (gx1, gy1, gx2, gy2)
    screenshot_pil = ImageGrab.grab(bbox=bbox)
    sx = getattr(scaler, 'x_scale', 1.0)
    sy = getattr(scaler, 'y_scale', 1.0)
    card_boxes = [
        (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
        for (x, y, w, h) in config.CARD_OFFSETS_WITH_SIZE
    ]
    return screenshot_pil, (0, 0), card_boxes

def wait_for_state_change(state_manager, initial_state, timeout=30):
    """Waits until the game state is different from the initial state with watchdog recovery."""
    print(f"Waiting for state to change from '{initial_state}'...")
    start_time = time.time()
    while True:
        state_manager.check_and_recover_if_stuck(unknown_timeout=8)
        current_state = state_manager.get_state()
        if current_state != initial_state and current_state != "UNKNOWN":
            print(f"State changed to '{current_state}'.")
            return current_state
        if time.time() - start_time > timeout:
            print("Wait timed out. Returning current state.")
            return current_state
        time.sleep(0.5)

def get_total_health_pct(ocr_data):
    enemy_hp = sum([get_health_percentage(ocr_data.get(k), t) for k, t in [('ptl', 'princess'), ('ptr', 'princess'), ('tk', 'king')]])
    friendly_hp = sum([get_health_percentage(ocr_data.get(k), t) for k, t in [('pbl', 'princess'), ('pbr', 'princess'), ('bk', 'king')]])
    return enemy_hp, friendly_hp

def calculate_reward(last_state, current_state):
    """
    Calculates the reward based on tower damage, enemies destroyed, 
    and a penalty for early King Tower activation.
    """
    ENEMY_DESTROYED_REWARD = 0.15
    KING_TOWER_ACTIVATION_PENALTY = 0.5

    last_ocr = last_state.get('ocr_data', {})
    current_ocr = current_state.get('ocr_data', {})

    last_enemy_hp, last_friendly_hp = get_total_health_pct(last_ocr)
    current_enemy_hp, current_friendly_hp = get_total_health_pct(current_ocr)
    damage_dealt = max(0, last_enemy_hp - current_enemy_hp)
    damage_taken = max(0, last_friendly_hp - current_friendly_hp)
    reward = (damage_dealt * 1.2) - (damage_taken * 1.0)

    # BUG 18 FIX: Raw detection count fluctuated every frame as troops walked behind towers
    # or were temporarily occluded, creating false-positive rewards that corrupted replay memory.
    # Rely on ground-truth tower damage (damage_dealt/damage_taken) and match outcome rewards.

    king_was_inactive = not last_ocr.get('tk')
    king_is_now_active = bool(current_ocr.get('tk'))
    ptl_was_alive = bool(last_ocr.get('ptl'))
    ptr_was_alive = bool(last_ocr.get('ptr'))
    
    if king_was_inactive and king_is_now_active and ptl_was_alive and ptr_was_alive:
        print(f"PENALTY: King Tower activated prematurely! Applying -{KING_TOWER_ACTIVATION_PENALTY} penalty.")
        reward -= KING_TOWER_ACTIVATION_PENALTY
        
    return reward

def run_offline_training(ai_agent, num_epochs=50, batch_size=64):
    """Performs dedicated offline training on current replay buffer."""
    print(f"\n{'='*60}")
    print(f"  OFFLINE REPLAY BUFFER TRAINING ({num_epochs} epochs)")
    print(f"{'='*60}")
    ai_agent.train(num_epochs=num_epochs, batch_size=batch_size)
    ai_agent.save()
    ai_agent.save_buffer()
    print("Offline training complete.")

def main():
    parser = argparse.ArgumentParser(description="Clash Royale AI — Autonomous Bot & Training")
    parser.add_argument('--mode', type=str, choices=['auto', 'record', 'train'], default='auto',
                        help="Operation mode: 'auto' (24/7 bot), 'record' (human teacher mode), 'train' (offline training)")
    # BUG-R4 FIX: Use default=None so explicit --epochs 5 is not treated as default
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs after each game (default 5, or 50 for train mode)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--games', type=int, default=0, help="Number of games to run (0 for infinite 24/7)")
    parser.add_argument('--epsilon', type=float, default=None, help="Exploration rate override (e.g. 0.05 for pure AI, 1.0 for random)")
    parser.add_argument('--no-train', action='store_true', help="Disable automatic post-battle training on match replay steps")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  CLASH ROYALE AI — {args.mode.upper()} MODE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=======================================================\n")

    ai_agent = Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, card_costs=CARD_COSTS, device=device)
    if args.epsilon is not None:
        ai_agent.set_epsilon(args.epsilon)

    # If pure offline training mode, run training and exit
    if args.mode == 'train':
        offline_epochs = args.epochs if args.epochs is not None else 50
        with keep_awake():
            run_offline_training(ai_agent, num_epochs=offline_epochs, batch_size=args.batch_size)
        return

    scaler = Scaler()
    controller = Controller(scaler=scaler)
    state_manager = GameStateManager(controller=controller, scaler=scaler)
    vision = Vision(scaler=scaler)
    recorder = HumanRecorder(scaler=scaler) if args.mode == 'record' else None

    matches_completed = 0

    # Wrap the entire live gameplay loop inside Windows keep_awake context
    with keep_awake():
        print(f"[BOT] Bot loop initialized in '{args.mode}' mode. Press Ctrl+C to stop.")
        try:
            while True:
                # Check for watchdog timeout or stuck popups
                recovery_flag = state_manager.check_and_recover_if_stuck(unknown_timeout=10, battle_timeout=280, in_battle=False)
                current_state_name = state_manager.get_state()
                print(f"\nCurrent game state is: {current_state_name}")

                if current_state_name == "MAIN_MENU":
                    if args.mode == 'record':
                        print("[HUMAN MODE] Main Menu detected. Starting match in 3 seconds (or click Battle yourself)...")
                        time.sleep(3)
                    state_manager.start_match()
                    wait_for_state_change(state_manager, "MAIN_MENU")
                
                elif current_state_name == "POST_BATTLE":
                    state_manager.end_match()
                    wait_for_state_change(state_manager, "POST_BATTLE")

                elif current_state_name == "POST_BATTLE_2":
                    state_manager.fix_bug()
                    wait_for_state_change(state_manager, "POST_BATTLE_2")

                elif current_state_name == "UNKNOWN":
                    # Sleep briefly; the watchdog in check_and_recover_if_stuck will auto-dismiss popups if stuck
                    time.sleep(2)
                
                elif current_state_name == "IN_BATTLE":
                    print(f"Battle in progress ({args.mode.upper()} mode)...")
                    current_game_log = {'steps': []}
                    battle_start_time = time.time()
                    last_state, action = None, None
                    battle_coords = {"cards": None}

                    if args.mode == 'record':
                        recorder.start()
                    
                    battle_loop_active = True
                    consecutive_non_battle = 0
                    while battle_loop_active:
                        now = time.time()
                        battle_duration = now - battle_start_time

                        # Watchdog check for stuck battle (safe in-battle mode: no random clicks in arena)
                        watchdog_res = state_manager.check_and_recover_if_stuck(battle_timeout=280, in_battle=True)
                        current_status = state_manager.get_state()

                        battle_has_ended = False
                        # Multi-signal battle end detector: templates, OCR, state priority
                        has_ended, end_reason = state_manager.check_battle_ended(
                            current_status=current_status, battle_duration=battle_duration, vision=vision
                        )
                        if has_ended:
                            print(f"[BATTLE] Battle ended ({end_reason}).")
                            battle_has_ended = True
                        elif current_status == "UNKNOWN":
                            consecutive_non_battle += 1
                            if consecutive_non_battle >= 4:  # ~1.8 seconds sustained UNKNOWN
                                print("[BATTLE] Battle ended (Left IN_BATTLE state).")
                                battle_has_ended = True
                        elif watchdog_res == "BATTLE_TIMEOUT":
                            print("[BATTLE] Battle ended (Watchdog timeout reached).")
                            battle_has_ended = True
                        elif battle_duration >= 35 and getattr(vision, 'consecutive_invalid_hand', 0) >= 8:
                            print("[BATTLE] Battle ended (Card deck absent for sustained duration).")
                            battle_has_ended = True
                        else:
                            consecutive_non_battle = 0

                        if battle_has_ended:
                            if args.mode == 'record':
                                recorder.stop()

                            # Accurately evaluate match result using OCR and tower damage
                            battle_duration = time.time() - battle_start_time
                            match_result, my_crowns, op_crowns, final_reward = state_manager.determine_match_outcome(
                                current_game_log['steps'], scaler, vision, battle_duration=battle_duration
                            )

                            if current_game_log['steps']:
                                current_game_log['steps'][-1]['reward'] += final_reward
                                ai_agent.learn_from_game(current_game_log, scaler=scaler)
                                if not args.no_train:
                                    post_epochs = args.epochs if args.epochs is not None else 5
                                    ai_agent.train(num_epochs=post_epochs, batch_size=args.batch_size)
                                    ai_agent.update_match_stats(match_result, final_reward)
                                else:
                                    # BUG-R5 FIX: Persist replay buffer and update match stats even when post-match training is disabled
                                    ai_agent.save_buffer()
                                    ai_agent.update_match_stats(match_result, final_reward)
                                    print("[SAFEGUARD] Training skipped (--no-train flag active). Replay buffer and match stats saved.")

                            matches_completed += 1
                            if args.games > 0 and matches_completed >= args.games:
                                print(f"Target of {args.games} games reached. Exiting.")
                                return
                            break

                        if battle_coords["cards"] is None:
                            time.sleep(2)
                            _, _, card_coords = find_cards_dynamically(scaler)
                            if card_coords:
                                battle_coords["cards"] = card_coords
                        
                        if battle_coords["cards"]:
                            screenshot, _, _ = find_cards_dynamically(scaler)
                            current_game_state = vision.get_game_state(screenshot, battle_coords["cards"])
                            
                            if last_state and action:
                                reward = calculate_reward(last_state, current_game_state)
                                current_game_log['steps'].append({
                                     'state': last_state, 'action': action,
                                     'reward': reward, 'next_state': current_game_state
                                })
                                action = None

                            if args.mode == 'record':
                                # Human Teacher Mode: User plays, recorder captures moves
                                human_action = recorder.pop_action()
                                if human_action:
                                    action = human_action
                                    last_state = current_game_state
                                    # Deduct elixir for the human player's card
                                    h_slot = human_action.get('card_slot')
                                    h_hand = current_game_state.get('hand', [])
                                    if h_slot is not None and 0 <= h_slot < len(h_hand):
                                        c_name = h_hand[h_slot]
                                        c_cost = config.CARD_COSTS.get(c_name, 3)
                                        vision.elixir_tracker.deduct(c_cost)
                                time.sleep(0.15)
                                continue

                            # Autonomous AI Mode: Decision Transformer & Tactical Brain decide action
                            step_idx = len(current_game_log['steps'])
                            action = ai_agent.decide_action(current_game_state, scaler, current_step=step_idx)
                            if action and action.get('action') == 'play_card':
                                slot, pos = action['card_slot'], action['position']
                                rule_name = action.get('tactical_rule', 'AI_DECISION')
                                
                                if 0 <= slot < len(battle_coords["cards"]):
                                    box = battle_coords["cards"][slot]
                                    click_x, click_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                                    print(f"[PLAY CARD] Executing {rule_name} (Slot {slot} at {pos})")
                                    controller.play_card((click_x, click_y), pos, slot=slot)
                                    # Deduct elixir for played card
                                    hand = current_game_state.get('hand', [])
                                    if 0 <= slot < len(hand):
                                        c_name = hand[slot]
                                        c_cost = config.CARD_COSTS.get(c_name, 3)
                                        vision.elixir_tracker.deduct(c_cost)
                                    # BUG-R2 FIX: Record last_state only when action was actually taken
                                    last_state = current_game_state
                                else:
                                    print(f"[WARNING] AI predicted invalid card slot: {slot}.")
                                    action = None
                            else:
                                action = None
                        
                        time.sleep(0.35)
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n[WARNING] Process interrupted by user. Saving model and buffer...")
            # BUG-R3 FIX: Guard against double recorder.stop() on KeyboardInterrupt
            if recorder and getattr(recorder, 'running', True):
                try:
                    recorder.stop()
                except Exception:
                    pass
            ai_agent.save()
            ai_agent.save_buffer()
            print("[SUCCESS] All weights and replay experiences saved cleanly.")

if __name__ == '__main__':
    main()
