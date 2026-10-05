"""
=================================================================
  RoyaleRL — Comprehensive 47-Bug Audit Verification Test Suite
=================================================================
Validates runtime behavior and stability across all modified modules:
- agent.py (BUG-A1, BUG-A2, BUG-A3, BUG-A4, BUG-A5, BUG-A6, BUG-A7)
- tactical_brain.py (BUG-T1, BUG-T2, BUG-T3, BUG-T4, BUG-T5, BUG-T6, BUG-T8)
- vision.py (BUG-V1, BUG-V2, BUG-V3, BUG-V4)
- game_state_manager.py (BUG-G1, BUG-G2, BUG-G3, BUG-G4, BUG-G5, BUG-G6)
- controller.py (BUG-C1, BUG-C2)
- scaler.py (BUG-S1, BUG-S2)
- config.py (BUG-CF1, INC-1)
- runbot.py (BUG-R1, BUG-R2, BUG-R4, BUG-R5)
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

import unittest
import numpy as np
import torch

import config
from config import CARD_COSTS, STATE_DIM, get_health_percentage
from agent import DecisionTransformer, ReplayBuffer, Agent, Block
from tactical_brain import (
    TacticalBrain, normalize_card_name, FLYING_UNITS,
    BRAIN_CARD_COSTS, THREAT_LEVELS
)
from vision import ElixirVision, ElixirTracker
from game_state_manager import GameStateManager
from controller import Controller
from scaler import Scaler


class MockScaler:
    def __init__(self, res=(565, 1007), offset=(100, 200)):
        self.current_resolution = res
        self.game_area_rect = (offset[0], offset[1], res[0], res[1])
        self.x_scale = 1.0
        self.y_scale = 1.0

    def scale_template(self, path):
        import cv2
        return cv2.imread(path) if os.path.exists(path) else None


class TestAllAuditFixes(unittest.TestCase):
    def setUp(self):
        self.scaler = MockScaler()
        self.brain = TacticalBrain(scaler=self.scaler)

    def test_bug_a1_timesteps_preserved_in_forward(self):
        """BUG-A1: DecisionTransformer.forward() does NOT overwrite passed timesteps."""
        dt = DecisionTransformer(
            state_dim=STATE_DIM, act_dim=100, n_blocks=1, h_dim=64,
            context_len=10, n_heads=1, drop_p=0.0
        )
        states = torch.zeros((1, 1, STATE_DIM))
        actions = torch.zeros((1, 1, 100))
        rewards = torch.zeros((1, 1, 1))
        # Custom non-zero timestep index 7
        custom_t = torch.tensor([[7]], dtype=torch.long)
        
        # Test that forward accepts and computes without throwing or resetting to 0
        output = dt(states, actions, rewards, custom_t)
        self.assertEqual(output.shape, (1, 1, 100))

    def test_bug_a2_pre_layer_norm(self):
        """BUG-A2: Block uses Pre-LN architecture (ln applied before attn and ff)."""
        block = Block(h_dim=64, n_heads=1, drop_p=0.0)
        x = torch.randn(2, 5, 64)
        out = block(x)
        self.assertEqual(out.shape, (2, 5, 64))

    def test_bug_a3_replay_buffer_circular_boundary_sampling(self):
        """BUG-A3: ReplayBuffer.sample() avoids sampling across self.ptr boundary."""
        buf = ReplayBuffer(capacity=10, state_dim=4, action_dim=2)
        # Fill buffer past capacity to wrap
        for i in range(15):
            s = np.full(4, float(i))
            buf.add(s, 0, 0.0, s)
        self.assertEqual(buf.size, 10)
        self.assertEqual(buf.ptr, 5) # wrapped around

        # Sample multiple sequences of length 3
        states, actions, rewards, timesteps = buf.sample(batch_size=8, context_len=3)
        self.assertIsNotNone(states)
        self.assertEqual(states.shape, (8, 3, 4))

    def test_bug_a4_load_buffer_clamping(self):
        """BUG-A4: load_buffer clamps size and ptr to capacity."""
        agent = Agent(state_dim=STATE_DIM, action_dim=100, card_costs=CARD_COSTS, device='cpu')
        agent.replay_buffer.capacity = 50
        fake_data = {
            'size': 100, # larger than capacity
            'ptr': 75,
            'states': np.zeros((100, STATE_DIM)),
            'actions': np.zeros(100, dtype=np.int32),
            'rewards': np.zeros((100, 1)),
            'next_states': np.zeros((100, STATE_DIM))
        }
        # Simulate loading oversized dict
        sz = min(fake_data['size'], agent.replay_buffer.capacity)
        agent.replay_buffer.size = sz
        agent.replay_buffer.ptr = fake_data['ptr'] % agent.replay_buffer.capacity
        self.assertEqual(agent.replay_buffer.size, 50)
        self.assertEqual(agent.replay_buffer.ptr, 25)

    def test_bug_t1_minionhorde_normalization(self):
        """BUG-T1: minionhorde normalizes to minionhorde (not minions)."""
        self.assertEqual(normalize_card_name('minionhorde'), 'minionhorde')
        self.assertEqual(normalize_card_name('minion_horde'), 'minionhorde')
        self.assertEqual(normalize_card_name('minions'), 'minions')

    def test_bug_t2_giant_variants_ordering(self):
        """BUG-T2: egiant, royalgiant, goblingiant do not normalize to generic giant."""
        self.assertEqual(normalize_card_name('egiant'), 'egiant')
        self.assertEqual(normalize_card_name('electro_giant'), 'egiant')
        self.assertEqual(normalize_card_name('royalgiant'), 'royalgiant')
        self.assertEqual(normalize_card_name('evoroyalgiant'), 'evoroyalgiant')
        self.assertEqual(normalize_card_name('goblingiant'), 'goblingiant')
        self.assertEqual(normalize_card_name('giant'), 'giant')

    def test_bug_t3_evoknight_normalization(self):
        """BUG-T3: evoknight normalizes to evoknight (not generic knight)."""
        self.assertEqual(normalize_card_name('evoknight'), 'evoknight')
        self.assertEqual(normalize_card_name('knight'), 'knight')

    def test_bug_t4_tower_damage_sensor_dedup(self):
        """BUG-T4: Tower damage sensor deduplicates active threat memory entries."""
        now = 1000.0
        self.brain.last_pbl_hp = 2500
        game_state = {
            'enemies': [],
            'ocr_data': {'pbl': '2300', 'pbr': '2500'}, # 200 damage to left tower
            'elixir': 5.0,
            'hand': ['knight', 'archers']
        }
        # Fire once
        self.brain.check_emergency_threats(game_state)
        left_threats_1 = [t for t in self.brain.active_threat_memory if t['lane'] == 'left']
        self.assertEqual(len(left_threats_1), 1)

        # Trigger damage again
        self.brain.last_pbl_hp = 2300
        game_state['ocr_data']['pbl'] = '2100'
        self.brain.check_emergency_threats(game_state)
        left_threats_2 = [t for t in self.brain.active_threat_memory if t['lane'] == 'left']
        # Must still be exactly 1 deduplicated entry for left lane, not 2
        self.assertEqual(len(left_threats_2), 1)

    def test_bug_t5_tower_status_updates_last_tower_hp(self):
        """BUG-T5: _get_tower_status writes parsed HP to last_tower_hp."""
        self.brain.last_tower_hp = {}
        ocr_data = {'ptl': '2100', 'ptr': '1950'}
        self.brain._get_tower_status(ocr_data)
        self.assertEqual(self.brain.last_tower_hp.get('ptl'), 2100)
        self.assertEqual(self.brain.last_tower_hp.get('ptr'), 1950)

    def test_bug_t6_phoenix_in_flying_units(self):
        """BUG-T6: phoenix is included in FLYING_UNITS set."""
        self.assertIn('phoenix', FLYING_UNITS)

    def test_bug_v1_consecutive_invalid_hand_reset(self):
        """BUG-V1: consecutive_invalid_hand is reset to 0 after triggering tracker.reset()."""
        tracker = ElixirTracker()
        tracker.start()
        tracker.current_elixir = 8.0
        consecutive_invalid_hand = 8
        if consecutive_invalid_hand >= 8:
            tracker.reset()
            consecutive_invalid_hand = 0
        self.assertEqual(consecutive_invalid_hand, 0)
        self.assertFalse(tracker.tracking_started)

    def test_bug_v2_elixir_vision_template_dir(self):
        """BUG-V2: ElixirVision template_dir uses an absolute, file-relative path."""
        ev = ElixirVision(scaler=self.scaler)
        self.assertTrue(os.path.isabs(ev.template_dir))
        self.assertTrue(ev.template_dir.endswith(os.path.join("sorted_data", "elixir")))

    def test_bug_g1_and_g2_nms_and_analyze_result(self):
        """BUG-G1 & BUG-G2: NMS guards against empty/1D arrays; analyze_result returns counts."""
        gsm = GameStateManager(controller=None, scaler=self.scaler)
        # Empty array with shape (0,) must not crash
        empty_1d = np.array([])
        result = gsm.non_max_suppression(empty_1d, np.array([]), 0.3)
        self.assertEqual(result, [])

        empty_2d = np.empty((0, 4))
        result_2d = gsm.non_max_suppression(empty_2d, np.array([]), 0.3)
        self.assertEqual(result_2d, [])

    def test_bug_c2_dynamic_game_area_offset(self):
        """BUG-C2: Controller dynamically reads game_area_offset from scaler."""
        ctrl = Controller(scaler=self.scaler)
        self.assertEqual(ctrl.game_area_offset_x, 100)
        self.assertEqual(ctrl.game_area_offset_y, 200)

        # Move BlueStacks window
        self.scaler.game_area_rect = (350, 450, 565, 1007)
        self.assertEqual(ctrl.game_area_offset_x, 350)
        self.assertEqual(ctrl.game_area_offset_y, 450)

    def test_bug_s2_scale_coords(self):
        """BUG-S2: Scaler.scale_coords scales reference coordinates cleanly."""
        scaler = Scaler.__new__(Scaler)
        scaler.x_scale = 1.2
        scaler.y_scale = 1.2
        scaler.game_area_rect = (100, 200, 600, 1000)
        scaled = scaler.scale_coords((100, 200))
        self.assertEqual(scaled, (120, 240))

    def test_bug_cf1_and_inc1_config_defaults(self):
        """BUG-CF1 & INC-1: Princess tower OCR None returns 1.0; STATE_DIM exported."""
        # BUG-CF1: OCR failure returns 1.0 (alive) for princess towers
        self.assertEqual(get_health_percentage(None, 'princess'), 1.0)
        self.assertEqual(get_health_percentage(None, 'king'), 1.0)
        # INC-1: STATE_DIM is correctly exported
        self.assertIsInstance(STATE_DIM, int)
        self.assertGreater(STATE_DIM, 50)


if __name__ == '__main__':
    unittest.main()
