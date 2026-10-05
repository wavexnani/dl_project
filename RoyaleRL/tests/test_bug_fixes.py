"""
=================================================================
  RoyaleRL — 20 Bug Audit Verification Test Suite
=================================================================
Validates fixes for all critical, high, and medium bugs identified
in the system audit:
- Bug 1: Internal Card Costs (BRAIN_CARD_COSTS)
- Bug 3: Screen-size coordinate normalization
- Bug 4: Valk in GROUND_ONLY_MELEE
- Bug 7: Multi-lane threat memory deduplication
- Bug 8: Score-5 gate on emergency defense
- Bug 11: Consecutive-None tower death detection
- Bug 12: Validator threat timestamp updates
- Bug 13: Counter-in-hand gate on unresolved threats
- Bug 15: royalhogs name normalization
- Bug 17: DT one-hot action conditioning
- Bug 18: Unstable YOLO reward removal
- Bug 19: Elixir sync window [0.5, 4.0]
- Bug 20: Pocket gate at 5.0 elixir
=================================================================
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for sub in ['core', 'drivers', 'training', 'evaluation']:
    sp = os.path.join(ROOT_DIR, sub)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import time
import unittest
import numpy as np
import torch

import config
from config import CARD_COSTS
from tactical_brain import (
    TacticalBrain, normalize_card_name, GROUND_ONLY_MELEE,
    BRAIN_CARD_COSTS, THREAT_LEVELS
)
from agent import Agent, NUM_GRID_LOCATIONS
from vision import ElixirTracker
from runbot import calculate_reward


class MockScaler:
    def __init__(self, res=(565, 1007)):
        self.current_resolution = res


class TestBugFixes(unittest.TestCase):
    def setUp(self):
        self.scaler = MockScaler()
        self.brain = TacticalBrain(scaler=self.scaler)
        state_dim = 1 + 6 + (4 * len(config.CARD_COSTS)) + (20 * 4)
        action_dim = (len(config.CARD_COSTS) + 1) * NUM_GRID_LOCATIONS
        self.agent = Agent(state_dim=state_dim, action_dim=action_dim, card_costs=CARD_COSTS, device='cpu')

    def test_bug1_brain_card_costs_mapping(self):
        """Bug 1: Internal card names map to correct elixir costs."""
        self.assertEqual(BRAIN_CARD_COSTS.get('valk'), 4)
        self.assertEqual(BRAIN_CARD_COSTS.get('hog'), 4)
        self.assertEqual(BRAIN_CARD_COSTS.get('mightyminer'), 4)
        self.assertEqual(BRAIN_CARD_COSTS.get('giant'), 5)
        self.assertEqual(BRAIN_CARD_COSTS.get('mini-pekka'), 4)
        self.assertEqual(BRAIN_CARD_COSTS.get('arrows'), 3)
        self.assertEqual(BRAIN_CARD_COSTS.get('fireball'), 4)
        print("  [PASS] Bug 1: BRAIN_CARD_COSTS correctly maps internal card names.")

    def test_bug3_screen_size_normalization(self):
        """Bug 3: TacticalBrain uses actual screenshot dimensions when provided in game_state."""
        game_state = {
            'screen_size': (600, 1050),
            'enemies': []
        }
        res = self.brain._get_res(game_state)
        self.assertEqual(res, (600, 1050))
        print("  [PASS] Bug 3: Screen-size coordinate normalization verified.")

    def test_bug4_valk_in_ground_only_melee(self):
        """Bug 4: Valkyrie is in GROUND_ONLY_MELEE and blocked against flying threats."""
        self.assertIn('valk', GROUND_ONLY_MELEE)
        game_state = {
            'hand': ['valk', 'knight', 'giant', 'arrows'],
            'enemies': [{'name': 'minions', 'box': (100, 300, 150, 350)}]  # Flying unit approaching
        }
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': (200, 600)}
        validated = self.brain.validate_candidate_action(candidate, game_state)
        self.assertIsNone(validated, "Valkyrie must be blocked against flying Minions!")
        print("  [PASS] Bug 4: Valkyrie ground-only melee restriction verified.")

    def test_bug7_multi_lane_threat_memory_dedup(self):
        """Bug 7: Two Giants in opposite lanes are both retained in threat memory."""
        now = time.time()
        game_state = {
            'enemies': [
                {'name': 'giant', 'box': (100, 450, 160, 510)},  # Left lane
                {'name': 'giant', 'box': (400, 450, 460, 510)}   # Right lane
            ],
            'elixir': 5.0,
            'hand': ['mini-pekka', 'knight', 'archers', 'giant']
        }
        self.brain.check_emergency_threats(game_state)
        giants_in_memory = [t for t in self.brain.active_threat_memory if t['name'] == 'giant']
        self.assertEqual(len(giants_in_memory), 2, "Both Left and Right Giants must be preserved in memory!")
        lanes = {t['lane'] for t in giants_in_memory}
        self.assertEqual(lanes, {'left', 'right'})
        print("  [PASS] Bug 7: Multi-lane threat memory deduplication verified.")

    def test_bug8_emergency_threat_score_gate(self):
        """Bug 8: Score-3 units (archers, bat) do NOT trigger emergency mandatory action."""
        game_state = {
            'enemies': [{'name': 'archers', 'box': (120, 450, 180, 510)}],  # Score 3
            'hand': ['knight', 'mini-pekka', 'musketeer', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '4000'}
        }
        action = self.brain.check_emergency_threats(game_state)
        self.assertIsNone(action, "Score-3 archers must not trigger emergency defense!")
        print("  [PASS] Bug 8: Threat score gate (score >= 5) on emergency response verified.")

    def test_bug11_consecutive_none_tower_dead_detection(self):
        """Bug 11: Single-frame OCR None on healthy tower is ignored; confirmed after 3 frames or when low."""
        # Setup tower as healthy (2500 HP)
        self.brain.last_tower_hp['ptl'] = 2500
        ocr_flicker = {'ptl': None, 'ptr': '2000'}

        # Frame 1 flicker: should NOT be dead
        left_dead, _ = self.brain._get_tower_status(ocr_flicker)
        self.assertFalse(left_dead, "Single OCR flicker on healthy tower must not declare it dead!")

        # Frame 2 flicker: should NOT be dead
        left_dead, _ = self.brain._get_tower_status(ocr_flicker)
        self.assertFalse(left_dead, "Two OCR flickers on healthy tower must not declare it dead!")

        # Frame 3 flicker: now declared dead
        left_dead, _ = self.brain._get_tower_status(ocr_flicker)
        self.assertTrue(left_dead, "Three consecutive None reads confirm tower destroyed!")
        print("  [PASS] Bug 11: Consecutive-None tower death detection verified.")

    def test_bug12_validator_threat_timestamp_update(self):
        """Bug 12: Validator updates timestamp of existing threat instead of ignoring it."""
        initial_time = time.time() - 2.0
        self.brain.active_threat_memory = [{
            'name': 'giant', 'score': 10, 'lane': 'left',
            'x': 0.25, 'y': 0.50, 'box': (100, 400, 160, 460), 'timestamp': initial_time
        }]
        game_state = {
            'hand': ['knight', 'musketeer', 'arrows', 'giant'],
            'enemies': [{'name': 'giant', 'box': (110, 420, 170, 480)}]
        }
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': (300, 600)}
        self.brain.validate_candidate_action(candidate, game_state)
        existing = next(t for t in self.brain.active_threat_memory if t['name'] == 'giant')
        self.assertGreater(existing['timestamp'], initial_time, "Timestamp of existing threat must be updated!")
        print("  [PASS] Bug 12: Threat memory timestamp updating verified.")

    def test_bug13_has_unresolved_threat_counter_in_hand_gate(self):
        """Bug 13: has_unresolved_threat only locks elixir if a counter IS actually in hand."""
        # Giant approaching, hand has NO counter (only spells and fragile units)
        game_state_no_counter = {
            'enemies': [{'name': 'giant', 'box': (120, 400, 180, 460)}],  # y >= 0.34, score >= 8
            'hand': ['arrows', 'fireball', 'empty', 'empty']  # No giant counter
        }
        self.assertFalse(
            self.brain.has_unresolved_threat(game_state_no_counter),
            "Should return False when no counter is in hand (prevents indefinite freezing)!"
        )

        # Giant approaching, hand HAS mini-pekka (hard counter)
        game_state_with_counter = {
            'enemies': [{'name': 'giant', 'box': (120, 400, 180, 460)}],
            'hand': ['mini-pekka', 'fireball', 'arrows', 'archers']
        }
        self.assertTrue(
            self.brain.has_unresolved_threat(game_state_with_counter),
            "Should return True when counter is in hand (holds elixir to afford it)!"
        )
        print("  [PASS] Bug 13: Counter-in-hand requirement on unresolved threats verified.")

    def test_bug15_royalhogs_normalization(self):
        """Bug 15: royalhogs is normalized to 'royalhogs', not generic 'hog'."""
        self.assertEqual(normalize_card_name('royalhogs'), 'royalhogs')
        self.assertEqual(normalize_card_name('royalhog'), 'royalhogs')
        self.assertEqual(normalize_card_name('hog'), 'hog')
        self.assertEqual(normalize_card_name('hogrider'), 'hog')
        print("  [PASS] Bug 15: Card name normalization preserves royalhogs distinct from hog.")

    def test_bug17_dt_one_hot_action_conditioning(self):
        """Bug 17: Decision Transformer conditions on one-hot last action index during inference."""
        self.agent.last_action_index = 5
        game_state = {
            'hand': ['knight', 'archers', 'musketeer', 'giant'],
            'elixir': 8.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '4000'},
            'enemies': []
        }
        candidates = self.agent._get_model_candidates(game_state, self.scaler, current_step=1, top_k=2)
        self.assertIsInstance(candidates, list)
        print("  [PASS] Bug 17: DT one-hot action conditioning executed successfully.")

    def test_bug18_yolo_unstable_reward_removed(self):
        """Bug 18: Fluctuating enemy count delta is not added to reward."""
        last_state = {
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'giant'}, {'name': 'knight'}]  # 2 enemies
        }
        current_state = {
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'giant'}]  # 1 enemy (1 dropped due to detector flicker)
        }
        reward = calculate_reward(last_state, current_state)
        # Without damage and without enemy reward, reward should be 0.0 (no false +0.15)
        self.assertEqual(reward, 0.0, "Fluctuating YOLO count must not inject false positive reward!")
        print("  [PASS] Bug 18: Unstable YOLO count reward removal verified.")

    def test_bug19_elixir_sync_window(self):
        """Bug 19: Elixir tracker accepts sync for 0.9 drift, rejects > 4.0 drift."""
        tracker = ElixirTracker()
        tracker.start()
        tracker.current_elixir = 5.0

        # Small drift of 0.9 (e.g. from 1-cost card): must sync
        tracker.sync_with_vision(5.9)
        self.assertAlmostEqual(tracker.current_elixir, 5.9, places=1)

        # Huge drift of 5.0 (optical noise e.g. 1 read as 10): must reject
        tracker.sync_with_vision(1.0)
        self.assertAlmostEqual(tracker.current_elixir, 5.9, places=1, msg="Out-of-range visual drift must be rejected!")
        print("  [PASS] Bug 19: Elixir sync window [0.5, 4.0] verified.")

    def test_bug20_pocket_gate_threshold(self):
        """Bug 20: Pocket deployment check fires at elixir >= 5.0 (not waiting for 6.0)."""
        game_state = {
            'hand': ['giant', 'musketeer', 'arrows', 'fireball'],
            'elixir': 5.0,  # Giant costs 5
            'ocr_data': {'ptl': None, 'ptr': '2000', 'tk': '4000'},  # Left breached
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'POCKET_ASSAULT')
        print("  [PASS] Bug 20: Pocket deployment fires promptly at 5.0 elixir.")


if __name__ == '__main__':
    unittest.main()
