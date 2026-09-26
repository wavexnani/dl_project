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

import time
import cv2
import json
import os
import sys
import argparse
import random
import torch
from PIL import ImageGrab

from scaler import Scaler
from game_state_manager import GameStateManager
from controller import Controller
from vision import Vision
from agent import Agent
from power_manager import keep_awake
from human_recorder import HumanRecorder
import config
from config import get_health_percentage, CARD_TO_INDEX, NUM_CARD_TYPES, CARD_COSTS

ALL_CARDS = list(CARD_COSTS.keys())
CARD_TO_INDEX = {name: i for i, name in enumerate(ALL_CARDS)}
NUM_CARD_TYPES = len(ALL_CARDS)

x_steps = 18
y_steps = 30
NUM_GRID_LOCATIONS = x_steps * y_steps
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID_LOCATIONS
STATE_DIM = 1 + 6 + (4 * NUM_CARD_TYPES) + (20 * 4)

def find_cards_dynamically(scaler):
    game_area = scaler.game_area_rect
    bbox = (game_area[0], game_area[1], game_area[0] + game_area[2], game_area[1] + game_area[3])
    screenshot_pil = ImageGrab.grab(bbox=bbox)
    card_boxes = [box for box in config.CARD_OFFSETS_WITH_SIZE]
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

    num_enemies_last = len(last_state.get('enemies', []))
    num_enemies_current = len(current_state.get('enemies', []))
    enemies_destroyed = num_enemies_last - num_enemies_current
    
    if enemies_destroyed > 0:
        enemy_reward = enemies_destroyed * ENEMY_DESTROYED_REWARD
        reward += enemy_reward

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
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs after each game (or total for train mode)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--games', type=int, default=0, help="Number of games to run (0 for infinite 24/7)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  CLASH ROYALE AI — {args.mode.upper()} MODE")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=======================================================\n")

    ai_agent = Agent(state_dim=STATE_DIM, action_dim=ACTION_DIM, card_costs=CARD_COSTS, device=device)

    # If pure offline training mode, run training and exit
    if args.mode == 'train':
        with keep_awake():
            run_offline_training(ai_agent, num_epochs=args.epochs if args.epochs != 5 else 50, batch_size=args.batch_size)
        return

    scaler = Scaler()
    controller = Controller(scaler=scaler)
    state_manager = GameStateManager(controller=controller, scaler=scaler)
    vision = Vision(scaler=scaler)
    recorder = HumanRecorder(scaler=scaler) if args.mode == 'record' else None

    matches_completed = 0

    # Wrap the entire live gameplay loop inside Windows keep_awake context
    with keep_awake():
        print(f"🚀 Bot loop initialized in '{args.mode}' mode. Press Ctrl+C to stop.")
        try:
            while True:
                # Check for watchdog timeout or stuck popups
                recovery_flag = state_manager.check_and_recover_if_stuck(unknown_timeout=10, battle_timeout=280)
                current_state_name = state_manager.get_state()
                print(f"\nCurrent game state is: {current_state_name}")

                if current_state_name == "MAIN_MENU":
                    if args.mode == 'record':
                        print("🎮 [HUMAN MODE] Main Menu detected. Starting match in 3 seconds (or click Battle yourself)...")
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
                    print(f"⚔️ Battle in progress ({args.mode.upper()} mode)...")
                    current_game_log = {'steps': []}
                    last_state, action = None, None
                    battle_coords = {"cards": None}

                    if args.mode == 'record':
                        recorder.start()
                    
                    battle_loop_active = True
                    consecutive_non_battle = 0
                    while battle_loop_active:
                        # Watchdog check for stuck battle
                        watchdog_res = state_manager.check_and_recover_if_stuck(battle_timeout=280)
                        current_status = state_manager.get_state()

                        battle_has_ended = False
                        if current_status in ("POST_BATTLE", "POST_BATTLE_2"):
                            print(f"🏁 Battle ended (Detected '{current_status}').")
                            battle_has_ended = True
                        elif current_status == "UNKNOWN":
                            consecutive_non_battle += 1
                            if consecutive_non_battle >= 5:  # ~2.5 seconds sustained UNKNOWN
                                print("🏁 Battle ended (Left IN_BATTLE state).")
                                battle_has_ended = True
                        elif watchdog_res == "BATTLE_TIMEOUT":
                            print("🏁 Battle ended (Watchdog timeout reached).")
                            battle_has_ended = True
                        else:
                            # Confirmed still IN_BATTLE
                            consecutive_non_battle = 0

                        if battle_has_ended:
                            if args.mode == 'record':
                                recorder.stop()

                            # Accurately evaluate match result using OCR and tower damage
                            match_result, my_crowns, op_crowns, final_reward = state_manager.determine_match_outcome(
                                current_game_log['steps'], scaler, vision
                            )

                            if current_game_log['steps']:
                                current_game_log['steps'][-1]['reward'] += final_reward
                                ai_agent.learn_from_game(current_game_log)
                                ai_agent.train(num_epochs=args.epochs, batch_size=args.batch_size)
                                ai_agent.update_match_stats(match_result, final_reward)

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

                            if args.mode == 'record':
                                # Human Teacher Mode: User plays, recorder captures moves
                                human_action = recorder.pop_action()
                                if human_action:
                                    action = human_action
                                    last_state = current_game_state
                                time.sleep(0.3)
                                continue

                            # Autonomous AI Mode: Decision Transformer decides action
                            action = ai_agent.decide_action(current_game_state, scaler)
                            if action and action.get('action') == 'play_card':
                                slot, pos = action['card_slot'], action['position']
                                
                                if 0 <= slot < len(battle_coords["cards"]):
                                    box = battle_coords["cards"][slot]
                                    click_x, click_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                                    controller.play_card((click_x, click_y), pos)
                                else:
                                    print(f"ERROR: AI predicted invalid card slot: {slot}. Using random fallback.")
                                    action = ai_agent._get_random_action(current_game_state, scaler)
                                    if action and action.get('action') == 'play_card':
                                        rand_slot = action['card_slot']
                                        if 0 <= rand_slot < len(battle_coords["cards"]):
                                            box = battle_coords["cards"][rand_slot]
                                            click_x, click_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                                            controller.play_card((click_x, click_y), action['position'])
                            
                            last_state = current_game_state
                        
                        time.sleep(1.8)
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user. Saving model and buffer...")
            if recorder:
                recorder.stop()
            ai_agent.save()
            ai_agent.save_buffer()
            print("✅ All weights and replay experiences saved cleanly.")

if __name__ == '__main__':
    main()
