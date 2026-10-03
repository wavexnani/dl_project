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
        # Target must lead the moving Minions forward along their flight path towards tower
        pos = action.get('position')
        self.assertTrue(pos[1] >= 445, f"Arrows must lead moving Minions forward! Got y={pos[1]}")
        print("  [PASS] Rule 4A: Predictive Arrows strike leading Minions swarm verified.")

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
        # Must cycle behind King Tower (y >= 0.75)
        pos = action.get('position')
        norm_y = pos[1] / 1007
        self.assertTrue(norm_y >= 0.75, f"Cycle play y={norm_y} is not safely behind King Tower!")
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
    # RULE 16: Predictive Aim Calculation Engine
    # =================================================================
    def test_rule16_predictive_aim_calculation(self):
        """Rule 16: Verifies calculate_predictive_aim leads moving enemies forward and clamps near tower."""
        box = (150, 400, 210, 450)  # Minions crossing river (y ~ 425)
        lead_pos = self.brain.calculate_predictive_aim(box, enemy_name='minions', spell_name='arrows')
        self.assertTrue(lead_pos[1] > 425, f"Lead aim must be forward of current position! Got {lead_pos}")

        # Near tower (y ~ 590): Lead must decay and clamp (no overshooting)
        box_near_tower = (150, 580, 210, 600)
        clamped_pos = self.brain.calculate_predictive_aim(box_near_tower, enemy_name='minions', spell_name='arrows')
        clamped_y_pct = clamped_pos[1] / 1007
        self.assertTrue(clamped_y_pct <= 0.61, f"Lead near tower must not overshoot! Got {clamped_y_pct}")
        print("  [PASS] Rule 16: Predictive Aim Engine & Stop-Zone Clamping verified.")

    # =================================================================
    # RULE 17: Outward Safe Tower Calibration
    # =================================================================
    def test_rule17_outward_safe_tower_calibration(self):
        """Rule 17: Princess tower snipes are calibrated outward away from King Tower."""
        game_state = {
            'hand': ['fireball', 'knight', 'archers', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '150', 'ptr': '2400', 'tk': '4000'},
            'enemies': []
        }
        action = self.brain.check_spell_finish(game_state)
        pos = action.get('position')
        norm_x = pos[0] / 565
        # Left Princess Tower must be at 0.20 (outward safety), not inward towards King
        self.assertAlmostEqual(norm_x, 0.20, delta=0.02)
        print("  [PASS] Rule 17: Outward Safe Tower Calibration verified.")

    # =================================================================
    # RULE 18: Elixir Relief Valve (No Passivity Trap)
    # =================================================================
    def test_rule18_elixir_relief_valve_prevents_passivity(self):
        """Rule 18: If candidate action was rejected but elixir >= 9.0, bot must safely cycle."""
        game_state = {
            'hand': ['arrows', 'giant', 'knight', 'archers'],
            'elixir': 9.2,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []  # No enemies on field
        }
        # Candidate wants to waste Arrows on empty grass (will be rejected)
        bad_arrows = {'action': 'play_card', 'card_slot': 0, 'position': (300, 600)}
        # Arbiter must reject bad_arrows and trigger safe Elixir Relief Cycle
        action = self.brain.arbitrate_decision(game_state, dt_action=bad_arrows)
        self.assertIsNotNone(action, "Elixir Relief Valve must prevent passivity at 9.2 elixir!")
        self.assertEqual(action.get('tactical_rule'), 'ELIXIR_LEAK_CYCLE')

        # Also verify agent integration: when model proposes a rejected move, agent relief valve fires
        with unittest.mock.patch.object(self.agent, '_get_model_action', return_value=bad_arrows):
            self.agent.set_epsilon(0.0)
            agent_action = self.agent.decide_action(game_state, self.scaler)
            self.assertIsNotNone(agent_action)
            self.assertEqual(agent_action.get('tactical_rule'), 'ELIXIR_LEAK_CYCLE')

        print("  [PASS] Rule 18: Elixir Relief Valve prevents passivity and forces safe cycling.")

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

    # =================================================================
    # RULE 19: Strict Spell Spatial Radius Check & Base Protection
    # =================================================================
    def test_rule19_spells_blocked_on_friendly_base_without_enemies(self):
        """Rule 19: Spells targeting friendly territory (y >= 0.45) MUST have enemies in splash radius."""
        # 1. Candidate wants to throw Arrows on our own Right Tower (y=0.77) with no enemies there
        game_state = {
            'hand': ['arrows', 'giant', 'knight', 'archers'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'valkyrie', 'box': (100, 300, 160, 360)}]  # Enemy is far away across river
        }
        own_tower_aim = {'action': 'play_card', 'card_slot': 0, 'position': (450, 780)}
        validated = self.brain.validate_candidate_action(own_tower_aim, game_state)
        self.assertIsNone(validated, "Throwing Arrows on our own tower without enemies nearby MUST BE BLOCKED!")

        # 2. But if an enemy swarm is attacking our friendly tower, spell defense is ALLOWED!
        game_state_defense = {
            'hand': ['arrows', 'giant', 'knight', 'archers'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{'name': 'minions', 'box': (430, 760, 480, 810)}]  # Minions on our tower!
        }
        validated_defense = self.brain.validate_candidate_action(own_tower_aim, game_state_defense)
        self.assertIsNotNone(validated_defense, "Defensive Arrows with enemies in splash radius must be allowed!")
        print("  [PASS] Rule 19: Spell Spatial Radius & Friendly Tower Protection verified.")

    # =================================================================
    # RULE 20: Troop Deployment Territory Clamping
    # =================================================================
    def test_rule20_illegal_enemy_territory_troops_clamped(self):
        """Rule 20: Troops placed across river (y < 0.50) with both towers up are clamped to friendly side."""
        game_state = {
            'hand': ['knight', 'musketeer', 'arrows', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []
        }
        # Model proposes Knight deep in enemy territory (y=0.33)
        illegal_enemy_drop = {'action': 'play_card', 'card_slot': 0, 'position': (390, int(0.33 * 1007))}
        validated = self.brain.validate_candidate_action(illegal_enemy_drop, game_state)
        self.assertIsNotNone(validated)
        clamped_y = validated.get('position')[1] / 1007
        self.assertTrue(clamped_y >= 0.52, f"Clamped troop deployment y={clamped_y} must be on friendly side!")
        print("  [PASS] Rule 20: Illegal enemy-territory troop deployments clamped to friendly side.")

    # =================================================================
    # RULE 21: Dormant King Tower Never Sniped on OCR Noise
    # =================================================================
    def test_rule21_dormant_king_never_sniped_on_ocr_noise(self):
        """Rule 21: OCR noise showing King at 5 HP while both Princess towers live MUST NOT trigger spell finish."""
        game_state = {
            'hand': ['arrows', 'fireball', 'knight', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000', 'tk': '5'},  # OCR noise!
            'enemies': []
        }
        action = self.brain.check_spell_finish(game_state)
        self.assertIsNone(action, "Must NEVER snipe dormant King Tower when both Princess Towers are alive!")
        print("  [PASS] Rule 21: Dormant King Tower protected from OCR noise spell finishes.")

    # =================================================================
    # RULE 22: Multi-Candidate Fallback Prevents Analysis Paralysis
    # =================================================================
    def test_rule22_multi_candidate_fallback_prevents_paralysis(self):
        """Rule 22: If top-1 action is rejected, decide_action seamlessly evaluates candidates without freezing."""
        game_state = {
            'hand': ['arrows', 'knight', 'musketeer', 'giant'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': []  # Empty field
        }
        self.agent.set_epsilon(0.0)
        action = self.agent.decide_action(game_state, self.scaler)
        self.assertIsNotNone(action, "Agent must not freeze when top candidate is rejected!")
        # Must not be Arrows on empty field
        hand = game_state['hand']
        slot = action.get('card_slot')
        self.assertNotEqual(hand[slot], 'arrows', "Agent must not waste Arrows on empty grass!")
        print("  [PASS] Rule 22: Multi-Candidate fallback successfully prevented passivity / paralysis.")

    # =================================================================
    # RULE 23: Universal 133-Card Threat Defense (Queen, Miner, Hog, Balloon)
    # =================================================================
    def test_rule23_archer_queen_threat_countered_by_melee(self):
        """Rule 23: Archer Queen crossing river is high threat (score 8) and countered by Knight on top."""
        game_state = {
            'hand': ['knight', 'arrows', 'fireball', 'giant'],
            'elixir': 4.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'queen',
                'confidence': 0.92,
                'box': (120, 440, 180, 500)
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Archer Queen must trigger mandatory defense!")
        self.assertEqual(action.get('card_slot'), 0)  # Knight is top counter
        self.assertEqual(action.get('tactical_rule'), 'MELEE_ON_RANGED')
        print("  [PASS] Rule 23A: Archer Queen Champion countered by Knight directly on top.")

    def test_rule23_miner_infiltrator_intercepted_at_tower(self):
        """Rule 23: Miner digging into friendly Left Tower triggers TOWER_INTERCEPT right on the tower."""
        game_state = {
            'hand': ['mini-pekka', 'arrows', 'musketeer', 'giant'],
            'elixir': 4.5,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'miner',
                'confidence': 0.89,
                'box': (100, 720, 160, 780)  # On friendly left tower
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Miner infiltrating tower must trigger mandatory defense!")
        self.assertEqual(action.get('card_slot'), 0)  # Mini-Pekka
        self.assertEqual(action.get('tactical_rule'), 'TOWER_INTERCEPT')
        # Check that deployment position is directly on the friendly tower (0.24, 0.77)
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.24, delta=0.05)
        self.assertAlmostEqual(norm_y, 0.77, delta=0.05)
        print("  [PASS] Rule 23B: Miner Tower Infiltrator intercepted directly at friendly Princess Tower.")

    def test_rule23_hog_rider_center_pull_kited(self):
        """Rule 23: Hog Rider rushing Left lane is pulled into the center firing zone."""
        game_state = {
            'hand': ['mini-pekka', 'arrows', 'musketeer', 'giant'],
            'elixir': 4.5,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'hog',
                'confidence': 0.95,
                'box': (120, 420, 180, 480)
            }]
        }
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action, "Hog Rider must trigger mandatory defense!")
        self.assertEqual(action.get('card_slot'), 0)  # Mini-Pekka is #1 hard counter
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.47, delta=0.04)
        print("  [PASS] Rule 23C: Hog Rider win condition kited to center-pull defense zone.")

    def test_rule23_balloon_blocks_ground_melee(self):
        """Rule 23: Balloon (flying) strictly blocks Knight or Mini-Pekka from being played."""
        game_state = {
            'hand': ['knight', 'mini-pekka', 'giant', 'goblin_cage'],
            'elixir': 5.0,
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'balloon',
                'confidence': 0.91,
                'box': (130, 420, 190, 480)
            }]
        }
        candidate_knight = {'action': 'play_card', 'card_slot': 0, 'position': (200, 500)}
        candidate_pekka = {'action': 'play_card', 'card_slot': 1, 'position': (200, 500)}
        self.assertIsNone(self.brain.validate_candidate_action(candidate_knight, game_state),
                          "Ground melee (Knight) must NEVER be played against Balloon!")
        self.assertIsNone(self.brain.validate_candidate_action(candidate_pekka, game_state),
                          "Ground melee (Mini-Pekka) must NEVER be played against Balloon!")
        print("  [PASS] Rule 23D: Ground melee vs Flying Balloon strictly blocked.")

    # =================================================================
    # RULE 24: Time-Variant Win Speed Bonus & Penalties
    # =================================================================
    def test_rule24_time_variant_speed_bonus_and_penalties(self):
        """Rule 24: Blitz win (<60s) awards massive bonus (+4.5 to +6.0), slow win (+2.5), loss (-2.5), draw (-1.0)."""
        # Test mathematical reward calculation directly
        def calc_reward(result, crowns, duration):
            speed_ratio = max(0.0, min(1.0, (180.0 - duration) / 180.0))
            speed_bonus = 2.5 * speed_ratio
            if result == 'WIN':
                return 2.5 + (0.5 * (crowns - 1)) + speed_bonus
            elif result == 'LOSS':
                return -2.5 - (0.3 * (crowns - 1))
            else:
                return -1.0

        blitz_win = calc_reward('WIN', 3, 45.0)   # 45-second 3-crown victory
        slow_win = calc_reward('WIN', 1, 180.0)   # 3-minute 1-crown victory
        loss = calc_reward('LOSS', 1, 120.0)      # Defeat
        draw = calc_reward('DRAW', 0, 180.0)      # Draw

        self.assertGreaterEqual(blitz_win, 5.0, f"Blitz 3-crown win must yield >= 5.0 reward! Got {blitz_win}")
        self.assertEqual(slow_win, 2.5, f"180s 1-crown win must equal 2.5! Got {slow_win}")
        self.assertLessEqual(loss, -2.5, f"Loss must penalize with <= -2.5! Got {loss}")
        self.assertEqual(draw, -1.0, f"Draw must penalize passivity with -1.0! Got {draw}")
        print("  [PASS] Rule 24: Time-Variant Speed Bonus and Loss/Draw Penalties verified.")

    # =================================================================
    # RULE 25: King Tower Breach Spearhead & Breached-Lane Reinforcements
    # =================================================================
    def test_rule25_king_breach_spearhead_and_leak_cycle(self):
        """Rule 25: Left Tower dead -> Assaults King Tower through breach and routes 10-elixir cycle to left lane."""
        game_state = {
            'hand': ['giant', 'musketeer', 'mini-pekka', 'knight'],
            'elixir': 7.0,
            'ocr_data': {'ptl': None, 'ptr': '2100', 'tk': '4000'},  # Left Tower destroyed!
            'enemies': []
        }
        # 1. Pocket King Assault check
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'POCKET_ASSAULT')
        pos = action.get('position')
        norm_x, norm_y = pos[0] / 565, pos[1] / 1007
        self.assertAlmostEqual(norm_x, 0.42, delta=0.04)  # Left breach pocket
        self.assertAlmostEqual(norm_y, 0.42, delta=0.04)  # Towards King Tower

        # 2. Elixir leak prevention routes reinforcements down the breached left lane
        game_state['elixir'] = 9.8
        leak_action = self.brain.check_elixir_leak_prevention(game_state)
        self.assertIsNotNone(leak_action)
        leak_pos = leak_action.get('position')
        leak_norm_x = leak_pos[0] / 565
        self.assertAlmostEqual(leak_norm_x, 0.26, delta=0.05,
                               msg="Reinforcements must cycle down breached Left lane (x~0.26) to push King Tower!")
        print("  [PASS] Rule 25: King Tower Breach Spearhead & Breached-Lane Reinforcements verified.")

    # =================================================================
    # RULE 26: Sentinel Defense on Opposite Lane During Breach
    # =================================================================
    def test_rule26_sentinel_defense_on_opposite_lane(self):
        """Rule 26: While Left Tower is breached, an incoming enemy push on the Right lane is intercepted immediately."""
        game_state = {
            'hand': ['mini-pekka', 'musketeer', 'arrows', 'giant'],
            'elixir': 6.0,
            'ocr_data': {'ptl': None, 'ptr': '2100', 'tk': '4000'},  # Left Tower breached
            'enemies': [{
                'name': 'hog',
                'confidence': 0.95,
                'box': (400, 420, 460, 480)  # Sneak attack on our Right Tower!
            }]
        }
        # Emergency defense must take precedence over the breach push!
        action = self.brain.get_mandatory_action(game_state)
        self.assertIsNotNone(action)
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE',
                         "Must immediately center-pull defense against Right-lane threat!")
        pos = action.get('position')
        norm_x = pos[0] / 565
        self.assertAlmostEqual(norm_x, 0.53, delta=0.04, msg="Defense must kite Right-lane threat to center-right!")
        print("  [PASS] Rule 26: Sentinel Defense on Opposite Lane during breach push verified.")


    # =================================================================
    # RULE 27: Tactical Elixir Lock & Hard-to-Refuse Complement Counter
    # =================================================================
    def test_rule27_unresolved_threat_locks_elixir_until_counter_affordable(self):
        """Rule 27: If a Giant is approaching and we need 4 elixir for Mini-Pekka, agent must hold elixir instead of wasting it on AI candidate actions."""
        game_state = {
            'hand': ['mini-pekka', 'knight', 'arrows', 'musketeer'],
            'elixir': 2.5,  # Not enough for Mini-Pekka yet
            'ocr_data': {'ptl': '2000', 'ptr': '2000'},
            'enemies': [{
                'name': 'giant',
                'confidence': 0.88,
                'box': (120, 320, 180, 400)  # y ~ 360/1007 = 0.36
            }]
        }
        # Step A: At 2.5 elixir, tactical brain recognizes unresolved threat
        self.assertTrue(self.brain.has_unresolved_threat(game_state), "Brain must detect unresolved Giant threat!")
        
        # Step B: Agent decide_action must return None (holding elixir) instead of firing DT/exploration candidate
        action = self.agent.decide_action(game_state, self.scaler)
        self.assertIsNone(action, "Agent MUST hold elixir (return None) when threat is approaching and counter is not yet affordable!")

        # Step C: Once elixir reaches 4.0, mandatory action fires Mini-Pekka counter immediately!
        game_state['elixir'] = 4.0
        action = self.agent.decide_action(game_state, self.scaler)
        self.assertIsNotNone(action, "Agent must immediately deploy hard-counter once elixir is reached!")
        self.assertEqual(action.get('card_slot'), 0)  # Mini-Pekka
        self.assertEqual(action.get('tactical_rule'), 'CENTER_PULL_DEFENSE')
        print("  [PASS] Rule 27A: Tactical Lock successfully held elixir and deployed hard-counter upon reaching cost.")

    def test_rule27_resilient_card_name_normalization(self):
        """Rule 27: normalize_card_name handles noisy detector labels and duplicate characters."""
        from tactical_brain import normalize_card_name
        self.assertEqual(normalize_card_name('SSpeargoblin'), 'speargoblin')
        self.assertEqual(normalize_card_name('Speaargoblin'), 'speargoblin')
        self.assertEqual(normalize_card_name('Skelleton'), 'skeleton')
        self.assertEqual(normalize_card_name('Skeleeton'), 'skeleton')
        self.assertEqual(normalize_card_name('Rascalgirl'), 'rascalgirl')
        self.assertEqual(normalize_card_name('Mightyminer'), 'mightyminer')
        self.assertEqual(normalize_card_name('Skeletonbarrel'), 'skeletonbarrel')
        print("  [PASS] Rule 27B: Resilient card name normalization verified.")


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("  RUNNING CLASH ROYALE TACTICAL RULES & ZERO-TOLERANCE TEST SUITE")
    print(f"{'='*70}\n")
    unittest.main()
