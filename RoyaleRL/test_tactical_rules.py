"""
=================================================================
  Clash Royale AI — Tactical Rules & Zero-Tolerance Test Suite
=================================================================
Validates that the AI model and Tactical Brain NEVER cross any of
the fundamental rules under any game state or edge case.
=================================================================
"""

import sys
import unittest
import numpy as np
import torch

import config
from config import CARD_COSTS, REFERENCE_RESOLUTION
from tactical_brain import TacticalBrain, normalize_card_name, SPELL_TOWER_DAMAGE, KING_TOWER_ZONE
from agent import Agent, PLACEMENT_GRID, NUM_GRID_LOCATIONS


class MockScaler:
    def __init__(self, res=(565, 1007)):
        self.current_resolution = res


class TestTacticalRules(unittest.TestCase):
    def setUp(self):
        self.scaler = MockScaler()
        self.brain = TacticalBrain(scaler=self.scaler)
        # Create an Agent with real dimensions
        state_dim = 1 + 6 + (4 * len(config.CARD_COSTS)) + (20 * 4)
        action_dim = (len(config.CARD_COSTS) + 1) * NUM_GRID_LOCATIONS
        self.agent = Agent(state_dim=state_dim, action_dim=action_dim, card_costs=CARD_COSTS, device='cpu')

    # =================================================================
    # RULE 1: Tower Spell-Snipe Finisher (Instant Win Rule)
    # =================================================================
    def test_rule1_fireball_lethal_finish_left_tower(self):
        """Rule 1: If Left Tower HP <= 280 and player has Fireball, instantly cast on Left Tower."""
        game_state = {
            'hand': ['fireball', 'knight', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '210', 'ptr': '2400', 'tk': '4000'},
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Should execute spell finish when tower HP <= 280")
        self.assertEqual(action.get('tactical_rule'), 'SPELL_FINISH')
        self.assertEqual(action.get('card_slot'), 0)  # Fireball is at index 0
        # Expected position near Left Tower (0.23, 0.14)
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.23, delta=0.05)
        self.assertAlmostEqual(norm_y, 0.14, delta=0.05)
        print("  [PASS] Rule 1A: Lethal Fireball on Left Tower verified.")

    def test_rule1_arrows_lethal_finish_right_tower(self):
        """Rule 1: If Right Tower HP <= 140 and player has Arrows, instantly cast on Right Tower."""
        game_state = {
            'hand': ['knight', 'arrows', 'musketeer', 'giant'],
            'elixir': 3.5,
            'ocr_data': {'ptl': '1800', 'ptr': '115', 'tk': '4000'},
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'SPELL_FINISH')
        self.assertEqual(action.get('card_slot'), 1)  # Arrows is at index 1
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.76, delta=0.05)
        self.assertAlmostEqual(norm_y, 0.14, delta=0.05)
        print("  [PASS] Rule 1B: Lethal Arrows on Right Tower verified.")

    def test_rule1_king_tower_lethal_snipe(self):
        """Rule 1: If King Tower HP <= 280, Fireball must snipe it for instant 3-crown victory."""
        game_state = {
            'hand': ['fireball', 'archers', 'knight', 'mini-pekka'],
            'elixir': 4.0,
            'ocr_data': {'ptl': None, 'ptr': '1500', 'tk': '220'},  # 1 tower down, King low
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'SPELL_FINISH')
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.51, delta=0.05)
        print("  [PASS] Rule 1C: 3-Crown Lethal King Tower Snipe verified.")

    # =================================================================
    # RULE 2: King Tower Protection & Dormancy (Never Wake King Early)
    # =================================================================
    def test_rule2_king_tower_lockout_blocks_direct_spell(self):
        """Rule 2: Spells targeting King Tower while Princess towers are alive must be redirected."""
        game_state = {
            'hand': ['fireball', 'knight', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2200', 'tk': '4000'},
            'enemies': []
        }
        # Candidate action attempts to drop Fireball on King Tower (x=0.50, y=0.08)
        king_pos = (int(0.50 * 565), int(0.08 * 1007))
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': king_pos}

        validated = self.brain.validate_candidate_action(candidate, game_state)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.get('tactical_rule'), 'KING_TOWER_PROTECTION')
        # Position must have been redirected away from King Tower to Princess Tower
        pos = validated.get('position')
        norm_x = pos[0] / 565
        self.assertTrue(norm_x < 0.35 or norm_x > 0.65, f"Redirected position x={norm_x} still in King Zone!")
        print("  [PASS] Rule 2: King Tower Lockout and safe redirection verified.")

    # =================================================================
    # RULE 3: Emergency Threat Defense (Never Ignore Threats)
    # =================================================================
    def test_rule3_giant_threat_response_right_lane(self):
        """Rule 3: Giant on Right lane crossing river triggers immediate tank killer counter."""
        game_state = {
            'hand': ['mini-pekka', 'arrows', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '4000'},
            'enemies': [{
                'name': 'giant',
                'confidence': 0.85,
                # Right lane bridge area: x in [380, 440], y in [450, 520]
                'box': (380, 450, 440, 520)
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Must defend against Giant crossing river!")
        self.assertEqual(action.get('card_slot'), 0)  # Mini-Pekka is #1 hard counter
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        print("  [PASS] Rule 3A: Giant Threat Defense and Tank-Killer counter verified.")

    def test_rule3_mini_pekka_threat_countered_by_minions(self):
        """Rule 3: Mini-Pekka on Left lane countered by Minions (air unit takes 0 damage)."""
        game_state = {
            'hand': ['minions', 'knight', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '4000'},
            'enemies': [{
                'name': 'mini-pekka',
                'confidence': 0.90,
                'box': (120, 440, 180, 500)  # Left lane
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('card_slot'), 0)  # Minions is #1 counter to Mini-Pekka
        print("  [PASS] Rule 3B: Mini-Pekka Air Hard-Counter (Minions) verified.")

    # =================================================================
    # RULE 4: Anti-Air Counter Rule (Never Ground Melee vs Flying)
    # =================================================================
    def test_rule4_arrows_wipes_minions(self):
        """Rule 4: Minions approaching -> Arrows cast directly on them."""
        game_state = {
            'hand': ['arrows', 'knight', 'giant', 'mini-pekka'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'minions',
                'confidence': 0.88,
                'box': (140, 420, 200, 470)
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'AIR_SWARM_CLEAR')
        self.assertEqual(action.get('card_slot'), 0)  # Arrows
        # Target must be centered on the minion swarm box
        pos = action.get('position')
        self.assertEqual(pos, (170, 445))
        print("  [PASS] Rule 4A: Direct Arrows strike on Minions swarm verified.")

    def test_rule4_strictly_blocks_ground_melee_vs_minions(self):
        """Rule 4: Candidate trying to play Knight or Mini-Pekka against Minions MUST BE BLOCKED."""
        game_state = {
            'hand': ['knight', 'mini-pekka', 'giant', 'goblin_hut'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'minions',
                'confidence': 0.88,
                'box': (140, 420, 200, 470)
            }]
        }
        # Candidate tries to play Knight (slot 0) or Mini-Pekka (slot 1) against flying Minions
        candidate_knight = {'action': 'play_card', 'card_slot': 0, 'position': (200, 500)}
        candidate_pekka = {'action': 'play_card', 'card_slot': 1, 'position': (200, 500)}

        self.assertIsNone(self.brain.validate_candidate_action(candidate_knight, game_state),
                          "Ground melee (Knight) must NEVER be played against Minions!")
        self.assertIsNone(self.brain.validate_candidate_action(candidate_pekka, game_state),
                          "Ground melee (Mini-Pekka) must NEVER be played against Minions!")
        print("  [PASS] Rule 4B: Ground melee vs Flying Minions strictly blocked.")

    # =================================================================
    # RULE 5: Center-Pull / Kiting Geometry
    # =================================================================
    def test_rule5_center_pull_coordinates(self):
        """Rule 5: Left threat pulls center-left (x~0.47), Right threat pulls center-right (x~0.53)."""
        # Left lane threat
        state_left = {
            'hand': ['knight', 'musketeer', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'giant', 'box': (120, 450, 180, 510)}]
        }
        action_left = self.brain.get_mandatory_action(state_left)
        pos_l = action_left.get('position')
        norm_xl = pos_l[0] / 565
        norm_yl = pos_l[1] / 1007
        self.assertAlmostEqual(norm_xl, 0.47, delta=0.03)
        self.assertAlmostEqual(norm_yl, 0.63, delta=0.03)

        # Right lane threat
        state_right = {
            'hand': ['knight', 'musketeer', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'giant', 'box': (400, 450, 460, 510)}]
        }
        action_right = self.brain.get_mandatory_action(state_right)
        pos_r = action_right.get('position')
        norm_xr = pos_r[0] / 565
        norm_yr = pos_r[1] / 1007
        self.assertAlmostEqual(norm_xr, 0.53, delta=0.03)
        self.assertAlmostEqual(norm_yr, 0.63, delta=0.03)
        print("  [PASS] Rule 5: Center-Pull Kiting Geometry (Tile 4-3) verified.")

    # =================================================================
    # RULE 6: 10-Elixir Leak Prevention
    # =================================================================
    def test_rule6_elixir_leak_cycle(self):
        """Rule 6: At >= 9.5 elixir with no threats, must cycle a troop behind King Tower."""
        game_state = {
            'hand': ['giant', 'knight', 'arrows', 'fireball'],
            'elixir': 9.8,
            'ocr_data': {'ptl': '2000', 'ptr': '1500'},  # Right tower weaker
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Must not sit at 9.8 elixir leaking!")
        self.assertEqual(action.get('tactical_rule'), 'ELIXIR_LEAK_CYCLE')
        # Must cycle behind King Tower (y >= 0.80)
        pos = action.get('position')
        norm_y = pos[1] / 1007
        self.assertTrue(norm_y >= 0.80, f"Cycle play y={norm_y} is not safely behind King Tower!")
        print("  [PASS] Rule 6: 10-Elixir Leak Prevention verified.")

    # =================================================================
    # RULE 7: Minimum Spell Value Protection (No Wasting Spells)
    # =================================================================
    def test_rule7_spell_waste_blocked_on_empty_field(self):
        """Rule 7: Candidate action casting Arrows on empty grass must be blocked."""
        game_state = {
            'hand': ['arrows', 'giant', 'knight', 'archers'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []  # No enemies on field
        }
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': (300, 600)}  # Arrows on empty field
        validated = self.brain.validate_candidate_action(candidate, game_state)
        self.assertIsNone(validated, "Casting Arrows on 0 enemies must be blocked!")
        print("  [PASS] Rule 7: Wasting Arrows on empty field strictly blocked.")

    # =================================================================
    # RULE 8: Fragile Troop Protection (Tank-in-Front Sequencing)
    # =================================================================
    def test_rule8_squishy_bridge_drop_pulled_back(self):
        """Rule 8: Naked Musketeer or Archers dropped at bridge must be pulled back safely."""
        game_state = {
            'hand': ['musketeer', 'giant', 'knight', 'arrows'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []
        }
        # Candidate attempts to drop Musketeer right at the bridge (y=0.46)
        bridge_pos = (200, int(0.46 * 1007))
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': bridge_pos}
        validated = self.brain.validate_candidate_action(candidate, game_state)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.get('tactical_rule'), 'TANK_IN_FRONT_SEQUENCING')
        new_y = validated.get('position')[1] / 1007
        self.assertTrue(new_y >= 0.58, f"Pulled back position y={new_y} still exposed at bridge!")
        print("  [PASS] Rule 8: Fragile Troop Protection & Tank-in-Front sequencing verified.")

    # =================================================================
    # RULE 9: The Pocket Exploitation
    # =================================================================
    def test_rule9_pocket_assault_when_tower_destroyed(self):
        """Rule 9: Left Tower destroyed -> Deploys in Left Pocket (0.42, 0.42) to snipe Right Tower."""
        game_state = {
            'hand': ['musketeer', 'mini-pekka', 'giant', 'knight'],
            'elixir': 6.5,
            'ocr_data': {'ptl': None, 'ptr': '1800', 'tk': '4000'},  # Left dead, Right alive
            'enemies': []
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'POCKET_ASSAULT')
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.42, delta=0.04)
        self.assertAlmostEqual(norm_y, 0.42, delta=0.04)
        print("  [PASS] Rule 9: The Pocket Exploitation verified.")

    # =================================================================
    # RULE 10: Dynamic Lane Defense Adaptation (No Left Bias)
    # =================================================================
    def test_rule10_lane_adaptation_no_left_bias(self):
        """Rule 10: Enemy pushing Right lane -> AI placement adapted to Right lane."""
        game_state = {
            'hand': ['knight', 'archers', 'musketeer', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'knight',
                'confidence': 0.85,
                'box': (400, 420, 460, 480)  # Right lane
            }]
        }
        # Candidate action wants to play on Left lane (x=0.20, y=0.55)
        candidate = {'action': 'play_card', 'card_slot': 0, 'position': (int(0.20 * 565), int(0.55 * 1007))}
        validated = self.brain.validate_candidate_action(candidate, game_state)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.get('tactical_rule'), 'LANE_ADAPTATION')
        new_x = validated.get('position')[0] / 565
        self.assertTrue(new_x > 0.50, f"Adapted position x={new_x} did not move to Right lane!")
        print("  [PASS] Rule 10: Lane Defense Adaptation verified.")

    # =================================================================
    # RULE 12: Tank Bridge Drop Prohibition (Never Drop Giant at Bridge)
    # =================================================================
    def test_rule12_giant_bridge_drop_blocked_and_redirected(self):
        """Rule 12: Prohibits dropping Giant at river bridge (y < 0.55). Redirects to backline/center."""
        game_state = {
            'hand': ['giant', 'musketeer', 'arrows', 'knight'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []
        }
        # Candidate tries to drop Giant at the bridge (y = 0.23, identical to user incident (68, 241))
        bridge_giant = {'action': 'play_card', 'card_slot': 0, 'position': (68, 241)}
        validated = self.brain.validate_candidate_action(bridge_giant, game_state)
        self.assertIsNotNone(validated)
        self.assertEqual(validated.get('tactical_rule'), 'SAFE_BACKLINE_TANK')
        new_y = validated.get('position')[1] / 1007
        self.assertTrue(new_y >= 0.75, f"Giant redirected y={new_y} is not safely in the back!")
        print("  [PASS] Rule 12: Bridge Giant drop strictly blocked and redirected to backline.")

    # =================================================================
    # RULE 13: Threat Memory & Persistence (No Flicker Blindness)
    # =================================================================
    def test_rule13_threat_memory_persists_across_empty_frames(self):
        """Rule 13: If detector flickers for 1 frame, threat memory keeps defensive lock active."""
        # Frame 1: Giant detected crossing river
        frame1_state = {
            'hand': ['mini-pekka', 'archers', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'giant', 'confidence': 0.85, 'box': (120, 450, 180, 510)}]
        }
        self.brain.check_emergency_threats(frame1_state)
        self.assertEqual(self.brain.get_active_threat_lane(), 'left')

        # Frame 2: Detector momentarily returns empty enemies (flicker)
        frame2_state = {
            'hand': ['mini-pekka', 'archers', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []  # Empty!
        }
        # Brain must STILL defend against the Giant using Threat Memory!
        action = self.brain.get_mandatory_action(frame2_state)
        self.assertIsNotNone(action, "Threat Memory must prevent dropping defense when detector flickers!")
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        print("  [PASS] Rule 13: Threat Memory maintains defense through temporary detector flickers.")

    # =================================================================
    # RULE 14: Tower Damage Sensor (Instant Defense on Tower HP Drop)
    # =================================================================
    def test_rule14_tower_damage_sensor_triggers_emergency_defense(self):
        """Rule 14: Friendly Tower HP drop triggers emergency defense even if vision was blind."""
        # Frame 1: Tower healthy at 2500 HP
        state_healthy = {
            'hand': ['mini-pekka', 'archers', 'arrows', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'pbl': '2500', 'pbr': '2500'},
            'enemies': []
        }
        self.brain.check_emergency_threats(state_healthy)

        # Frame 2: Tower HP drops to 2350 HP (took 150 damage), no visual detection
        state_damaged = {
            'hand': ['mini-pekka', 'archers', 'arrows', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'pbl': '2350', 'pbr': '2500'},  # Left tower hit!
            'enemies': []  # Vision missed it!
        }
        action = self.brain.get_mandatory_action(state_damaged)
        self.assertIsNotNone(action, "Tower damage sensor must trigger emergency defense!")
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        self.assertEqual(self.brain.get_active_threat_lane(), 'left')
        print("  [PASS] Rule 14: Tower Damage Sensor successfully detected tower hit and locked defense.")

    # =================================================================
    # END-TO-END TEST: Agent.decide_action with real weights
    # =================================================================
    def test_agent_end_to_end_guarantees(self):
        """End-to-End: Verifies that decide_action strictly respects all rules under model & random modes."""
        # 1. Emergency Giant Threat with epsilon=0.0 (Pure AI Model)
        self.agent.set_epsilon(0.0)
        game_state_threat = {
            'hand': ['mini-pekka', 'arrows', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '4000'},
            'enemies': [{'name': 'giant', 'confidence': 0.9, 'box': (120, 450, 180, 510)}]
        }
        action_threat = self.agent.decide_action(game_state_threat, self.scaler)
        self.assertEqual(action_threat.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        self.assertEqual(action_threat.get('card_slot'), 0)  # Must be mini-pekka

        # 2. Lethal Finish with epsilon=1.0 (Pure Random Exploration)
        self.agent.set_epsilon(1.0)
        game_state_lethal = {
            'hand': ['fireball', 'knight', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '180', 'ptr': '2000', 'tk': '4000'},
            'enemies': []
        }
        action_lethal = self.agent.decide_action(game_state_lethal, self.scaler)
        self.assertEqual(action_lethal.get('tactical_rule'), 'SPELL_FINISH')
        self.assertEqual(action_lethal.get('card_slot'), 0)  # Must be fireball

        print("  [PASS] End-to-End: Agent strictly obeys Tactical Brain across both exploitation & exploration!")


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("  RUNNING CLASH ROYALE TACTICAL RULES & ZERO-TOLERANCE TEST SUITE")
    print(f"{'='*70}\n")
    unittest.main()
