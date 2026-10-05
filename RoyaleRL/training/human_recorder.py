"""
=================================================================
  Clash Royale AI — Human Demonstration Recorder
=================================================================
Monitors human gameplay on BlueStacks in real-time.
Captures:
  1. Card selection (clicks or keyboard 1-4)
  2. Arena deployment coordinates
  3. Associated game states & rewards
Saves expert human demonstrations directly to the ReplayBuffer.
=================================================================
"""

import os
import sys
import time
import threading
import collections
from pynput import mouse, keyboard

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for sub in ['core', 'drivers', 'training', 'evaluation']:
    sp = os.path.join(ROOT_DIR, sub)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import config

class HumanRecorder:
    def __init__(self, scaler):
        self.scaler = scaler
        self.selected_slot = None
        self.last_selected_time = 0
        # BUG-H2 FIX: Use deque for O(1) popleft instead of O(n) list.pop(0)
        self.pending_actions = collections.deque()
        self.lock = threading.Lock()
        
        self.mouse_listener = None
        self.keyboard_listener = None
        self._is_recording = False

    def start(self):
        """Starts listeners for human input."""
        if self._is_recording:
            return

        self._is_recording = True
        self.selected_slot = None
        self.pending_actions = collections.deque()

        self.mouse_listener = mouse.Listener(
            on_click=self._on_mouse_click
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()
        print("[RECORDER] Human Teacher Mode active. Play your match manually!")
        print("    Tip: Click a card (or press 1-4) then click the arena to deploy.")

    def stop(self):
        """Stops listeners."""
        self._is_recording = False
        if self.mouse_listener:
            try:
                self.mouse_listener.stop()
            except Exception:
                pass
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
        print("[RECORDER] Recording stopped.")

    def pop_action(self):
        """Retrieves and clears any captured human action."""
        with self.lock:
            if self.pending_actions:
                # BUG-H2 FIX: popleft() is O(1) on deque vs O(n) pop(0) on list
                return self.pending_actions.popleft()
            return None

    def _get_relative_coords(self, abs_x, abs_y):
        """Converts absolute screen coordinates to game-area-relative coordinates."""
        gx, gy, gw, gh = self.scaler.game_area_rect
        if gx <= abs_x <= gx + gw and gy <= abs_y <= gy + gh:
            return abs_x - gx, abs_y - gy
        return None, None

    def _check_card_slot_clicked(self, rel_x, rel_y):
        """Checks if relative click landed on one of the 4 card slots (with margin)."""
        cur_w, cur_h = self.scaler.current_resolution
        ref_w, ref_h = config.REFERENCE_RESOLUTION
        sx = cur_w / ref_w
        sy = cur_h / ref_h

        for slot_idx, (bx, by, bw, bh) in enumerate(config.CARD_OFFSETS_WITH_SIZE):
            scaled_bx = bx * sx
            scaled_by = by * sy
            scaled_bw = bw * sx
            scaled_bh = bh * sy
            # Generous padding so clicking near card edge registers cleanly
            if (scaled_bx - 15) <= rel_x <= (scaled_bx + scaled_bw + 15) and (scaled_by - 25) <= rel_y <= (scaled_by + scaled_bh + 25):
                return slot_idx
        return None

    def _is_in_arena(self, rel_x, rel_y):
        """Checks if click is within the arena bounding box."""
        cur_w, cur_h = self.scaler.current_resolution
        min_x = config.ARENA_BBOX[0] * cur_w
        min_y = config.ARENA_BBOX[1] * cur_h
        max_x = config.ARENA_BBOX[2] * cur_w
        max_y = config.ARENA_BBOX[3] * cur_h
        return min_x <= rel_x <= max_x and min_y <= rel_y <= max_y

    def _record_deploy(self, slot, rel_x, rel_y):
        with self.lock:
            action_record = {
                'action': 'play_card',
                'card_slot': slot,
                'position': (int(rel_x), int(rel_y)),
                'timestamp': time.time(),
                'is_human': True
            }
            self.pending_actions.append(action_record)
            print(f"[HUMAN MOVE] Card {slot + 1} deployed at arena ({int(rel_x)}, {int(rel_y)})")

    def _on_mouse_click(self, x, y, button, pressed):
        if not self._is_recording or button != mouse.Button.left:
            return

        rel_x, rel_y = self._get_relative_coords(x, y)
        if rel_x is None:
            return

        if pressed:
            # Mouse DOWN: only check for card slot selection.
            # BUG-H1 FIX: Never record a deployment on DOWN in the arena.
            # Recording on DOWN captures the cursor-enter position for drags,
            # not the final release position. Only mouse-UP should record.
            slot_clicked = self._check_card_slot_clicked(rel_x, rel_y)
            if slot_clicked is not None:
                self.selected_slot = slot_clicked
                self.last_selected_time = time.time()
                print(f"[HUMAN] Selected card slot {slot_clicked + 1}")
                return
        else:
            # Mouse UP (Release): Records both clicks and drag-and-drop.
            # For a regular click, UP fires at the same spot as DOWN — correct.
            # For drag-and-drop, UP fires at the final release position — correct.
            if self.selected_slot is not None and self._is_in_arena(rel_x, rel_y):
                if (time.time() - self.last_selected_time) < 6.0:
                    self._record_deploy(self.selected_slot, rel_x, rel_y)
                    self.selected_slot = None
                    return

    def _on_key_press(self, key):
        if not self._is_recording:
            return
        try:
            char = getattr(key, 'char', None)
            if char in ('1', '2', '3', '4'):
                self.selected_slot = int(char) - 1
                self.last_selected_time = time.time()
                print(f"[HUMAN] Selected card slot {char}")
        except Exception:
            pass
