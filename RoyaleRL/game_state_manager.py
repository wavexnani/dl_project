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

    def check_and_recover_if_stuck(self, unknown_timeout=12, battle_timeout=280, in_battle=False):
        """
        Watchdog routine:
        1. If state is UNKNOWN for > unknown_timeout seconds, attempt popup dismissal.
        2. If battle has exceeded battle_timeout (4.6m), flag battle timeout.
        """
        now = time.time()
        elapsed = now - self.state_start_time

        if self.current_state == "UNKNOWN" and elapsed > unknown_timeout:
            print(f"🛡️ [WATCHDOG] Stuck in UNKNOWN state for {elapsed:.1f}s. Running popup dismissal...")
            self.controller.dismiss_popups(in_battle=in_battle)
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
          1. OCR parsing of the versus screen tokens (splitting by VS to identify
             whether the local player (Mouli-p / right side) or the opponent (left side) has the Winner banner).
          2. Crown extraction from the winner's token indicators (e.g. '[3', digits)
             and Clash Royale 3-minute King Tower destruction rule.
          3. Tower damage ground-truth fallback from battle steps.
        Returns: (match_result, my_crowns, op_crowns, final_reward)
        """
        print("⏳ [RESULT] Waiting for battle end banner & animations (2.0s)...")
        time.sleep(2.0)

        ocr_result = None
        winner_tokens = []
        loser_tokens = []
        winner_keywords = ["WINNER", "WINNERL", "WIC'NER", "WICNER", "VICTORY", "YOU WIN", "GAGNANT"]

        # Phase 1: Use OCR across multiple frames to read end screen text
        print("🔍 [RESULT] Scanning screen with OCR for match result banner...")
        for attempt in range(5):
            game_area = scaler.game_area_rect
            bbox = (game_area[0], game_area[1], game_area[0] + game_area[2], game_area[1] + game_area[3])
            screen_pil = ImageGrab.grab(bbox=bbox)
            
            # Read text with EasyOCR
            texts = vision.reader.readtext(np.array(screen_pil), detail=0, text_threshold=0.50)
            print(f"   [Frame {attempt+1}] OCR: {texts}")

            if not texts:
                time.sleep(0.5)
                continue

            # Look for 'VS' separator on the result screen
            vs_idx = -1
            for i, t in enumerate(texts):
                t_up = t.upper().strip()
                if t_up in ('VS', 'V.S', 'V') or 'VS' in t_up:
                    vs_idx = i
                    break

            if vs_idx != -1:
                side_left = texts[:vs_idx]
                side_right = texts[vs_idx + 1:]

                # In Clash Royale: Opponent is on the left, Local Player is on the right
                # Also verify if player's name ('mouli') is explicitly present
                if any('mouli' in t.lower() for t in side_left):
                    player_side, opp_side = side_left, side_right
                elif any('mouli' in t.lower() for t in side_right):
                    player_side, opp_side = side_right, side_left
                else:
                    opp_side, player_side = side_left, side_right

                player_has_winner = any(any(kw in t.upper() for kw in winner_keywords) for t in player_side)
                opp_has_winner = any(any(kw in t.upper() for kw in winner_keywords) for t in opp_side)

                if player_has_winner and not opp_has_winner:
                    ocr_result = "WIN"
                    winner_tokens = player_side
                    loser_tokens = opp_side
                    break
                elif opp_has_winner and not player_has_winner:
                    ocr_result = "LOSS"
                    winner_tokens = opp_side
                    loser_tokens = player_side
                    break
                elif any('DEFEAT' in t.upper() for t in player_side):
                    ocr_result = "LOSS"
                    winner_tokens = opp_side
                    loser_tokens = player_side
                    break
                elif any('VICTORY' in t.upper() for t in player_side):
                    ocr_result = "WIN"
                    winner_tokens = player_side
                    loser_tokens = opp_side
                    break
            else:
                # No VS separator: check full screen text
                text_str = " ".join(texts).upper()
                if any(w in text_str for w in ["DEFEAT", "YOU LOSE", "LOST", "DEFAITE"]):
                    ocr_result = "LOSS"
                    break
                elif any(w in text_str for w in ["VICTORY", "YOU WIN", "VICTOIRE"]):
                    ocr_result = "WIN"
                    break
                elif any(w in text_str for w in ["DRAW", "TIE", "TIEBREAKER"]):
                    ocr_result = "DRAW"
                    break

            time.sleep(0.5)

        # Phase 2: Crown Evaluation
        # In Clash Royale: Destroying the middle main base (King Tower) = 3 CROWNS immediately.
        # Check if winner tokens explicitly contain crown indicator (e.g. '[3', '(3', or exact '3')
        winner_crowns = None
        for t in winner_tokens:
            cleaned = ''.join(c for c in t if c.isdigit())
            # Require exact single digit or bracketed crown indicator to avoid matching level 13 or trophies
            if cleaned == '3' or '[3' in t or '(3' in t or '3 CROWN' in t.upper():
                winner_crowns = 3
                break
            elif cleaned == '2' or '[2' in t or '(2' in t or '2 CROWN' in t.upper():
                winner_crowns = 2
                break
            elif cleaned == '1' or '[1' in t or '(1' in t or '1 CROWN' in t.upper():
                if winner_crowns is None:
                    winner_crowns = 1

        # Default to 1 crown (standard win) unless 2 or 3 is explicitly detected in winner tokens
        if winner_crowns is None:
            winner_crowns = 1

        # Phase 3: Final Outcome & Crown Assignment
        if ocr_result == "WIN":
            result = "WIN"
            max_my_crowns = winner_crowns
            max_op_crowns = 0  # Opponent got 0 crowns if King Tower taken
            reward = 2.0 + (0.5 * (max_my_crowns - 1))

        elif ocr_result == "LOSS":
            result = "LOSS"
            max_my_crowns = 0
            max_op_crowns = winner_crowns
            reward = -1.5 - (0.3 * (max_op_crowns - 1))

        elif ocr_result == "DRAW":
            result = "DRAW"
            max_my_crowns = 0
            max_op_crowns = 0
            reward = 0.0

        else:
            # Fallback to cumulative damage from steps if OCR failed to find anything
            print("⚠️ [RESULT] Banner text not detected by OCR, falling back to damage metrics.")
            total_damage_dealt = sum(s.get('reward', 0.0) for s in battle_steps if s.get('reward', 0.0) > 0)
            total_damage_taken = sum(abs(s.get('reward', 0.0)) for s in battle_steps if s.get('reward', 0.0) < 0)
            print(f"📊 [RESULT] Battle Damage Analysis: Dealt={total_damage_dealt:.2f}, Taken={total_damage_taken:.2f}")

            if total_damage_dealt > total_damage_taken + 0.3:
                result = "WIN"
                max_my_crowns = 3
                max_op_crowns = 0
                reward = 2.0
            elif total_damage_taken > total_damage_dealt + 0.3:
                result = "LOSS"
                max_my_crowns = 0
                max_op_crowns = 3
                reward = -1.5
            else:
                result = "DRAW"
                max_my_crowns = 0
                max_op_crowns = 0
                reward = 0.0

        print(f"🏆 [RESULT] Match Result: {result} (Player Crowns: {max_my_crowns} | Opponent Crowns: {max_op_crowns}) | Final Reward: {reward:+.2f}")
        return result, max_my_crowns, max_op_crowns, reward

    def get_crown_boxes(self, screen_pil):
        """Legacy compatibility wrapper."""
        c_me, c_op = self.get_crown_counts(screen_pil)
        return [None] * c_me, [None] * c_op
    
