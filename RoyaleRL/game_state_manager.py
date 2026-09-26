import cv2
import numpy as np
from PIL import ImageGrab
import os
import time
from controller import Controller
import config

class GameStateManager:
    """Manages the game state using visual anchors."""
    def __init__(self, controller, scaler):
        self.controller = controller
        self.scaler = scaler
        self.anchors = self._load_anchors()
        self.state_start_time = time.time()
        self.current_state = "UNKNOWN"
        self.battle_start_time = None
        if self.anchors:
            print(f"Loaded {len(self.anchors)} state anchors.")

    def _load_anchors(self):
        anchor_path = "sorted_data/anchors/"
        # Screen-state anchors used in get_state() for navigation
        screen_state_files = {
            "MAIN_MENU":    "battle_anchor.png",
            "IN_BATTLE":    "game_anchor.png",
            "POST_BATTLE":  "ok_anchor.png",
            "POST_BATTLE_2": "ok_anchor2.png",
            "LUCKY_BOX":    "luckybox2.png"
        }
        # Crown templates — kept separate, NOT used in get_state()
        crown_files = {
            "WIN_CROWN":  "bluecrowns.png",
            "LOSE_CROWN": "redcrowns.png"
        }

        all_files = {**screen_state_files, **crown_files}
        anchors = {}
        for state, filename in all_files.items():
            path = os.path.join(anchor_path, filename)
            if not os.path.exists(path):
                base, ext = os.path.splitext(filename)
                alt = os.path.join(anchor_path, f"{base}{ext.upper()}")
                if os.path.exists(alt):
                    path = alt
            if os.path.exists(path):
                anchors[state] = self.scaler.scale_template(path)

        # Store screen-state anchors separately so get_state() never returns WIN_CROWN/LOSE_CROWN
        self._screen_state_keys = list(screen_state_files.keys())
        return anchors

    def get_state(self):
        """Returns screen navigation state (never returns WIN_CROWN/LOSE_CROWN)."""
        game_area_rect = self.scaler.game_area_rect
        bbox = (game_area_rect[0], game_area_rect[1], game_area_rect[0] + game_area_rect[2], game_area_rect[1] + game_area_rect[3])
        screen_pil = ImageGrab.grab(bbox=bbox)
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)

        STATE_THRESHOLDS = {
            "IN_BATTLE": 0.68,
            "POST_BATTLE": 0.75,
            "POST_BATTLE_2": 0.75,
            "MAIN_MENU": 0.80,
            "LUCKY_BOX": 0.80
        }

        # Select the highest-confidence matching state above threshold
        best_state = "UNKNOWN"
        best_val = 0.0

        for state in self._screen_state_keys:
            anchor_img = self.anchors.get(state)
            if anchor_img is None: continue
            thresh = STATE_THRESHOLDS.get(state, 0.80)
            res = cv2.matchTemplate(screen_cv_gray, anchor_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= thresh and max_val > best_val:
                best_val = max_val
                best_state = state

        self._update_internal_state(best_state)
        return best_state

    def _update_internal_state(self, state):
        if state != self.current_state:
            self.current_state = state
            self.state_start_time = time.time()
            if state == "IN_BATTLE":
                self.battle_start_time = time.time()
            elif state in ("MAIN_MENU", "POST_BATTLE", "POST_BATTLE_2"):
                self.battle_start_time = None

    def check_and_recover_if_stuck(self, unknown_timeout=12, battle_timeout=280):
        """
        Watchdog routine:
        1. If state is UNKNOWN for > unknown_timeout seconds, attempt popup dismissal.
        2. If battle has exceeded battle_timeout (4.6m), flag battle timeout.
        """
        now = time.time()
        elapsed = now - self.state_start_time

        if self.current_state == "UNKNOWN" and elapsed > unknown_timeout:
            print(f"🛡️ [WATCHDOG] Stuck in UNKNOWN state for {elapsed:.1f}s. Running popup dismissal...")
            self.controller.dismiss_popups()
            self.state_start_time = now # reset to give dismissal a chance
            return "RECOVERED"

        if self.current_state == "IN_BATTLE" and self.battle_start_time:
            battle_duration = now - self.battle_start_time
            if battle_duration > battle_timeout:
                print(f"🛡️ [WATCHDOG] Battle exceeded maximum duration ({battle_duration:.1f}s). Flagging timeout.")
                return "BATTLE_TIMEOUT"

        return None
    
    def start_match(self):
        print("Attempting to start match...")
        for name in ('sorted_data/anchors/battle_anchor.png', 'sorted_data/anchors/battle_anchor.PNG'):
            if os.path.exists(name) and self.controller.find_and_click(name):
                return True
        return False

    def end_match(self):
        print("Attempting to end match...")
        for name in ('sorted_data/anchors/ok_anchor.png', 'sorted_data/anchors/ok_anchor.PNG'):
            if os.path.exists(name) and self.controller.find_and_click(name):
                return True
        return False
    
    def fix_bug(self):
        print("Attempting to fix...")
        for _ in range(5):
            for name in ('sorted_data/anchors/luckybox2.png', 'sorted_data/anchors/luckybox2.PNG'):
                if os.path.exists(name) and self.controller.find_and_click(name):
                    break
        for name in ('sorted_data/anchors/ok_anchor2.png', 'sorted_data/anchors/ok_anchor2.PNG'):
            if os.path.exists(name) and self.controller.find_and_click(name):
                return True
        return False
    
    def non_max_suppression(self, boxes, scores, overlapThresh):
        if len(boxes) == 0:
            return []
        
        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
            
        return pick

    def analyze_result(self, screen_pil):
        print("Analyzing match result...")
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        
        win_crown_template = self.anchors.get("WIN_CROWN")
        lose_crown_template = self.anchors.get("LOSE_CROWN")

        if win_crown_template is None or lose_crown_template is None:
            print("Error: Crown anchor templates not found.")
            return
        
        win_w, win_h = win_crown_template.shape[::-1]
        lose_w, lose_h = lose_crown_template.shape[::-1]
        
        threshold = 0.9

        # Define the Region of Interest for your crowns at the bottom 60%
        game_height = screen_pil.height
        my_roi_y = int(game_height * 0.40) 
        my_roi = screen_cv_gray[my_roi_y:game_height, :]

        # Define the Region of Interest for your opponent's crowns at the top 40%
        opponent_roi_y = 0
        opponent_roi_height = int(game_height * 0.40) 
        opponent_roi = screen_cv_gray[opponent_roi_y : opponent_roi_y + opponent_roi_height, :]
        
        # Count my crowns in my ROI
        res_win_me = cv2.matchTemplate(my_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_me = np.where(res_win_me >= threshold)
        my_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_me[::-1])
        ]), res_win_me[win_locs_me], 0.3)
        
        # Count my opponent's crowns in my opponent's ROI
        res_win_opponent = cv2.matchTemplate(opponent_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
        win_locs_opponent = np.where(res_win_opponent >= threshold)
        opponent_win_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*win_locs_opponent[::-1])
        ]), res_win_opponent[win_locs_opponent], 0.3)

        # Count empty crowns on opponent's side
        res_lose_opponent = cv2.matchTemplate(opponent_roi, lose_crown_template, cv2.TM_CCOEFF_NORMED)
        lose_locs_opponent = np.where(res_lose_opponent >= threshold)
        opponent_lose_boxes = self.non_max_suppression(np.array([
            [pt[0], pt[1], pt[0] + lose_w, pt[1] + lose_h] for pt in zip(*lose_locs_opponent[::-1])
        ]), res_lose_opponent[lose_locs_opponent], 0.3)
        
        # The logic is now based on win crowns only
        my_crowns = len(my_win_boxes)
        opponent_crowns = len(opponent_win_boxes)

        print(f"Detected my crowns: {my_crowns}")
        print(f"Detected opponent's crowns: {opponent_crowns}")
        
        if my_crowns > opponent_crowns:
            print(f"You won! 🎉 ({my_crowns}-{opponent_crowns})")
        elif opponent_crowns > my_crowns:
            print(f"You lost. 😔 ({my_crowns}-{opponent_crowns})")
        else:
            print(f"The match was a draw. 🤝 ({my_crowns}-{opponent_crowns})")
    
    def get_crown_counts(self, screen_pil):
        """
        Detects crown counts for player and opponent using multi-threshold template matching.
        Returns: (my_crowns, opponent_crowns)
        """
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)
        win_crown_template = self.anchors.get("WIN_CROWN")
        lose_crown_template = self.anchors.get("LOSE_CROWN")

        if win_crown_template is None or lose_crown_template is None:
            return 0, 0

        win_w, win_h = win_crown_template.shape[::-1]
        lose_w, lose_h = lose_crown_template.shape[::-1]

        game_height = screen_pil.height
        my_roi_y = int(game_height * 0.35)
        my_roi = screen_cv_gray[my_roi_y:game_height, :]

        opponent_roi_height = int(game_height * 0.50)
        opponent_roi = screen_cv_gray[0:opponent_roi_height, :]

        my_crowns = 0
        opponent_crowns = 0

        # Try adaptive thresholds from 0.82 down to 0.65 to account for scaling & lighting
        for thresh in [0.82, 0.76, 0.70, 0.65]:
            # Player blue crowns
            res_me = cv2.matchTemplate(my_roi, win_crown_template, cv2.TM_CCOEFF_NORMED)
            locs_me = np.where(res_me >= thresh)
            if len(locs_me[0]) > 0:
                boxes_me = self.non_max_suppression(np.array([
                    [pt[0], pt[1], pt[0] + win_w, pt[1] + win_h] for pt in zip(*locs_me[::-1])
                ]), res_me[locs_me], 0.35)
                if len(boxes_me) > my_crowns:
                    my_crowns = min(3, len(boxes_me))

            # Opponent red crowns
            res_op = cv2.matchTemplate(opponent_roi, lose_crown_template, cv2.TM_CCOEFF_NORMED)
            locs_op = np.where(res_op >= thresh)
            if len(locs_op[0]) > 0:
                boxes_op = self.non_max_suppression(np.array([
                    [pt[0], pt[1], pt[0] + lose_w, pt[1] + lose_h] for pt in zip(*locs_op[::-1])
                ]), res_op[locs_op], 0.35)
                if len(boxes_op) > opponent_crowns:
                    opponent_crowns = min(3, len(boxes_op))

            if my_crowns > 0 or opponent_crowns > 0:
                break

        return my_crowns, opponent_crowns

    def determine_match_outcome(self, battle_steps, scaler, vision):
        """
        Determines whether the match was a WIN, LOSS, or DRAW using:
          1. OCR to reliably read "VICTORY", "DEFEAT", or "DRAW" on the screen.
          2. Tower damage and King Tower destruction ground-truth fallback from steps.
        Returns: (match_result, my_crowns, op_crowns, final_reward)
        """
        print("⏳ [RESULT] Waiting for battle end banner & animations (2.0s)...")
        time.sleep(2.0)

        ocr_result = None
        detected_all_texts = []

        # Phase 1: Use OCR across multiple frames to read end screen text
        print("🔍 [RESULT] Scanning screen with OCR for match result banner...")
        for attempt in range(5):
            game_area = scaler.game_area_rect
            bbox = (game_area[0], game_area[1], game_area[0] + game_area[2], game_area[1] + game_area[3])
            screen_pil = ImageGrab.grab(bbox=bbox)
            
            # Read text with EasyOCR
            texts = vision.reader.readtext(np.array(screen_pil), detail=0, text_threshold=0.55)
            detected_all_texts.extend(texts)
            text_str = " ".join(texts).upper()
            print(f"   [Frame {attempt+1}] OCR: {texts}")
            
            if any(w in text_str for w in ["VICTORY", "YOU WIN", "VICTOIRE", "WINNER"]):
                ocr_result = "WIN"
                break
            elif any(w in text_str for w in ["DEFEAT", "YOU LOSE", "LOST", "DEFAITE", "GAME OVER"]):
                ocr_result = "LOSS"
                break
            elif any(w in text_str for w in ["DRAW", "TIE", "TIEBREAKER"]):
                ocr_result = "DRAW"
                break
                
            time.sleep(0.5)

        # Phase 2: Analyze battle steps (damage and tower trajectory)
        total_damage_dealt = 0.0
        total_damage_taken = 0.0
        enemy_king_killed = False
        friendly_king_killed = False

        if battle_steps:
            for step in battle_steps:
                r = step.get('reward', 0.0)
                if r > 0:
                    total_damage_dealt += r
                elif r < 0:
                    total_damage_taken += abs(r)

            # Check if king towers were destroyed in the final steps
            for step in battle_steps[-8:]:
                ocr = step.get('next_state', {}).get('ocr_data', {})
                tk_val = ocr.get('tk', '')
                bk_val = ocr.get('bk', '')
                if tk_val in ('0', '00', '000') or (total_damage_dealt >= 2.0):
                    enemy_king_killed = True
                if bk_val in ('0', '00', '000') or (total_damage_taken >= 2.0):
                    friendly_king_killed = True

        print(f"📊 [RESULT] Battle Damage Analysis: Dealt={total_damage_dealt:.2f}, Taken={total_damage_taken:.2f}")

        # Combine OCR and tower damage metrics
        if ocr_result == "WIN":
            result = "WIN"
            if enemy_king_killed or total_damage_dealt >= 1.5 or "3" in detected_all_texts:
                max_my_crowns = 3
            elif total_damage_dealt >= 0.7:
                max_my_crowns = 2
            else:
                max_my_crowns = 1
            max_op_crowns = 0 if total_damage_taken < 0.6 else (1 if total_damage_taken < 1.2 else 2)

        elif ocr_result == "LOSS":
            result = "LOSS"
            max_my_crowns = 0 if total_damage_dealt < 0.6 else (1 if total_damage_dealt < 1.2 else 2)
            if friendly_king_killed or total_damage_taken >= 1.5 or "3" in detected_all_texts:
                max_op_crowns = 3
            elif total_damage_taken >= 0.7:
                max_op_crowns = 2
            else:
                max_op_crowns = 1

        elif ocr_result == "DRAW":
            result = "DRAW"
            max_my_crowns = 0
            max_op_crowns = 0

        else:
            print("⚠️ [RESULT] Banner text not detected by OCR, falling back to damage metrics.")
            if total_damage_dealt > total_damage_taken + 0.3:
                result = "WIN"
                max_my_crowns = 3 if total_damage_dealt >= 1.5 else (2 if total_damage_dealt >= 0.7 else 1)
                max_op_crowns = 0
            elif total_damage_taken > total_damage_dealt + 0.3:
                result = "LOSS"
                max_my_crowns = 0
                max_op_crowns = 3 if total_damage_taken >= 1.5 else (2 if total_damage_taken >= 0.7 else 1)
            else:
                result = "DRAW"
                max_my_crowns = 0
                max_op_crowns = 0

        # Calculate final reinforcement learning reward
        if result == "WIN":
            reward = 2.0 + (0.5 * (max_my_crowns - 1))
        elif result == "LOSS":
            reward = -1.5 - (0.3 * (max_op_crowns - 1))
        else:
            reward = 0.0

        print(f"🏆 [RESULT] Match Result: {result} (Player Crowns: {max_my_crowns} | Opponent Crowns: {max_op_crowns}) | Final Reward: {reward:+.2f}")
        return result, max_my_crowns, max_op_crowns, reward

    def get_crown_boxes(self, screen_pil):
        """Legacy compatibility wrapper."""
        c_me, c_op = self.get_crown_counts(screen_pil)
        return [None] * c_me, [None] * c_op
    
