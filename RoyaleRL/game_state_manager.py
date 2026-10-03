import cv2
import numpy as np
from PIL import ImageGrab
import os
import re
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
        gx1 = max(0, game_area_rect[0])
        gy1 = max(0, game_area_rect[1])
        gx2 = max(gx1 + 10, game_area_rect[0] + game_area_rect[2])
        gy2 = max(gy1 + 10, game_area_rect[1] + game_area_rect[3])
        bbox = (gx1, gy1, gx2, gy2)
        screen_pil = ImageGrab.grab(bbox=bbox)
        screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)

        STATE_THRESHOLDS = {
            "POST_BATTLE": 0.63,
            "POST_BATTLE_2": 0.63,
            "MAIN_MENU": 0.75,
            "LUCKY_BOX": 0.75,
            "IN_BATTLE": 0.68
        }

        # Select matching state above threshold with priority for post-battle / menu
        best_state = "UNKNOWN"
        best_val = 0.0

        for state in self._screen_state_keys:
            anchor_img = self.anchors.get(state)
            if anchor_img is None: continue
            thresh = STATE_THRESHOLDS.get(state, 0.75)
            res = cv2.matchTemplate(screen_cv_gray, anchor_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= thresh:
                # Prioritize post-battle and main-menu states over IN_BATTLE
                # (since an OK button never appears during actual fighting)
                if state in ("POST_BATTLE", "POST_BATTLE_2"):
                    if best_state not in ("POST_BATTLE", "POST_BATTLE_2") or max_val > best_val:
                        best_val = max_val
                        best_state = state
                elif state == "MAIN_MENU":
                    if best_state not in ("POST_BATTLE", "POST_BATTLE_2") and max_val > best_val:
                        best_val = max_val
                        best_state = state
                else:
                    if best_state not in ("POST_BATTLE", "POST_BATTLE_2", "MAIN_MENU") and max_val > best_val:
                        best_val = max_val
                        best_state = state

        self._update_internal_state(best_state)
        return best_state

    def check_battle_ended(self, current_status=None, battle_duration=0, vision=None):
        """
        Multi-signal check to determine if the battle has terminated:
        1. Checks if current_status is POST_BATTLE, POST_BATTLE_2, or MAIN_MENU.
        2. Direct template check on ok_anchor and ok_anchor2 with adaptive threshold (>= 0.60).
        3. If match duration >= 35s, runs quick OCR check on lower screen for 'OK', 'Play Again', etc.
        Returns: (has_ended: bool, reason: str or None)
        """
        if current_status in ("POST_BATTLE", "POST_BATTLE_2"):
            return True, f"Detected state '{current_status}'"
        if current_status == "MAIN_MENU":
            return True, "Returned to MAIN_MENU"

        # Check ok templates directly
        game_area_rect = self.scaler.game_area_rect
        gx1 = max(0, game_area_rect[0])
        gy1 = max(0, game_area_rect[1])
        gx2 = max(gx1 + 10, game_area_rect[0] + game_area_rect[2])
        gy2 = max(gy1 + 10, game_area_rect[1] + game_area_rect[3])
        bbox = (gx1, gy1, gx2, gy2)
        try:
            screen_pil = ImageGrab.grab(bbox=bbox)
            screen_cv_gray = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2GRAY)

            for anchor_name in ("POST_BATTLE", "POST_BATTLE_2"):
                anchor_img = self.anchors.get(anchor_name)
                if anchor_img is not None:
                    res = cv2.matchTemplate(screen_cv_gray, anchor_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val >= 0.60:
                        return True, f"Direct template match on {anchor_name} (conf={max_val:.2f})"

            # If battle has lasted more than 35s, check OCR for end-of-battle keywords
            if battle_duration >= 35 and vision and hasattr(vision, 'reader'):
                h = screen_pil.height
                bottom_crop = screen_pil.crop((0, int(h * 0.60), screen_pil.width, h))
                ocr_detections = vision.reader.readtext(np.array(bottom_crop), detail=0, text_threshold=0.4)
                for txt in ocr_detections:
                    clean = str(txt).upper().replace(' ', '')
                    if any(kw in clean for kw in ("OK", "PLAYAGAIN", "CONTINUE", "VICTORY", "DEFEAT")):
                        return True, f"Detected post-battle button text: '{txt}'"
        except Exception:
            pass

        return False, None

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
            if os.path.exists(name) and self.controller.find_and_click(name, confidence=0.70):
                return True
        return False

    def end_match(self):
        print("Attempting to end match...")
        for name in ('sorted_data/anchors/ok_anchor.png', 'sorted_data/anchors/ok_anchor.PNG'):
            if os.path.exists(name) and self.controller.find_and_click(name, confidence=0.65):
                return True
        return False
    
    def fix_bug(self):
        print("Attempting to fix...")
        for _ in range(5):
            for name in ('sorted_data/anchors/luckybox2.png', 'sorted_data/anchors/luckybox2.PNG'):
                if os.path.exists(name) and self.controller.find_and_click(name, confidence=0.70):
                    break
        for name in ('sorted_data/anchors/ok_anchor2.png', 'sorted_data/anchors/ok_anchor2.PNG'):
            if os.path.exists(name) and self.controller.find_and_click(name, confidence=0.65):
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

    def determine_match_outcome(self, battle_steps, scaler, vision, battle_duration=None):
        """
        Determines whether the match was a WIN, LOSS, or DRAW using:
          1. High-precision spatial OCR (detail=1): tracks exact X-coordinates of Winner banner
             and trophy indicators relative to screen center (Left = Opponent, Right = Player).
          2. Battle steps King Tower ground truth: checks if friendly or enemy King was destroyed.
          3. Conservative damage fallback: NEVER awards 3-crown blitzkrieg on ambiguous damage.
          4. Time-Variant Speed Bonus for genuine verified wins, penalties for losses/draws.
        Returns: (match_result, my_crowns, op_crowns, final_reward)
        """
        print("⏳ [RESULT] Waiting for battle end banner & animations (2.0s)...")
        time.sleep(2.0)

        ocr_result = None
        winner_crowns = None
        winner_keywords = ["WINNER", "WINNERL", "WIC'NER", "WICNER", "VICTORY", "VICTOIRE", "GAGNANT"]

        game_area = scaler.game_area_rect
        cur_w = max(1, game_area[2])
        cur_h = max(1, game_area[3])

        # Step 0: Check Battle Steps Ground Truth for King Tower Destruction
        friendly_king_destroyed = False
        enemy_king_destroyed = False

        if battle_steps:
            for s in battle_steps[-10:]:
                next_ocr = s.get('next_state', {}).get('ocr_data', {})
                bk = next_ocr.get('bk')
                if bk == '0' or bk == 0:
                    friendly_king_destroyed = True
                tk = next_ocr.get('tk')
                if tk == '0' or tk == 0:
                    enemy_king_destroyed = True

        # Phase 1: Spatial OCR across multiple frames (Vertical Clash Royale Layout: Opponent Top, Player Bottom)
        print("🔍 [RESULT] Scanning screen with spatial OCR for match result banner...")
        for attempt in range(5):
            gx1 = max(0, game_area[0])
            gy1 = max(0, game_area[1])
            gx2 = max(gx1 + 10, game_area[0] + game_area[2])
            gy2 = max(gy1 + 10, game_area[1] + game_area[3])
            bbox = (gx1, gy1, gx2, gy2)
            screen_pil = ImageGrab.grab(bbox=bbox)

            try:
                detections = vision.reader.readtext(np.array(screen_pil), detail=1, text_threshold=0.35)
            except Exception as e:
                print(f"   [Frame {attempt+1}] OCR Read Error: {e}")
                time.sleep(0.5)
                continue

            if not detections:
                time.sleep(0.5)
                continue

            token_logs = []
            player_winner_detected = False
            opp_winner_detected = False
            player_trophy_loss = False
            player_trophy_gain = False
            crown_candidate = None

            vs_ny = None
            player_ny = None
            winner_ny = None
            winner_text = ""

            # Pass 1: Discover vertical landmarks
            for box, text, conf in detections:
                t_str = str(text).strip()
                t_clean = t_str.upper().replace(' ', '').replace("'", "")
                cy = sum(p[1] for p in box) / 4.0
                ny = cy / cur_h
                
                if t_clean in ('VS', 'V.S', 'V'):
                    vs_ny = ny
                if 'mouli' in t_str.lower():
                    player_ny = ny
                if any(kw in t_clean for kw in winner_keywords):
                    winner_ny = ny
                    winner_text = t_str

            # Pass 2: Evaluate tokens with landmark context
            for box, text, conf in detections:
                t_str = str(text).strip()
                t_clean = t_str.upper().replace(' ', '').replace("'", "")
                cy = sum(p[1] for p in box) / 4.0
                ny = cy / cur_h
                token_logs.append(f"'{t_str}'(y={ny:.2f})")

                # Defeat text
                if any(kw in t_clean for kw in ["DEFEAT", "YOULOSE", "LOST", "DEFAITE"]):
                    opp_winner_detected = True
                    print(f"   🚨 Detected DEFEAT text on screen: '{t_str}'")

                # Victory text
                if any(kw in t_clean for kw in ["VICTORY", "YOUWIN", "VICTOIRE"]):
                    player_winner_detected = True
                    print(f"   🏆 Detected VICTORY text on screen: '{t_str}'")

                # Trophy detection: NEVER parse usernames like 'Mouli-3' as trophies!
                if not any(c.isalpha() for c in t_str):
                    m = re.search(r'([+\-])\s*(\d{1,3})', t_str)
                    if m:
                        sign = m.group(1)
                        val = int(m.group(2))
                        if 15 <= val <= 45:
                            if sign == '+':
                                player_trophy_gain = True
                                print(f"   🏆 Detected POSITIVE trophy gain (+{val}) at y={ny:.2f}: '{t_str}'")
                            elif sign == '-':
                                player_trophy_loss = True
                                print(f"   🚨 Detected NEGATIVE trophy loss (-{val}) at y={ny:.2f}: '{t_str}'")

                # Crown extraction
                if 0.35 <= ny <= 0.60:
                    cleaned_digit = ''.join(c for c in t_str if c.isdigit())
                    if cleaned_digit in ('1', '2', '3'):
                        crown_candidate = int(cleaned_digit)

            # Evaluate Winner Banner relative to VS and Player
            if winner_ny is not None:
                if vs_ny is not None:
                    if winner_ny > vs_ny:
                        player_winner_detected = True
                        print(f"   🏆 WINNER banner '{winner_text}' is BELOW VS (y={winner_ny:.2f} > vs_y={vs_ny:.2f}) -> Local Player WON!")
                    else:
                        opp_winner_detected = True
                        print(f"   🚨 WINNER banner '{winner_text}' is ABOVE VS (y={winner_ny:.2f} < vs_y={vs_ny:.2f}) -> Opponent WON!")
                elif player_ny is not None:
                    if abs(winner_ny - player_ny) < 0.22:
                        player_winner_detected = True
                        print(f"   🏆 WINNER banner '{winner_text}' is adjacent to Player '{player_ny:.2f}' -> Local Player WON!")
                    else:
                        opp_winner_detected = True
                        print(f"   🚨 WINNER banner '{winner_text}' is far above Player -> Opponent WON!")
                else:
                    if winner_ny >= 0.33:
                        player_winner_detected = True
                        print(f"   🏆 WINNER banner '{winner_text}' is in lower half (y={winner_ny:.2f}) -> Local Player WON!")
                    else:
                        opp_winner_detected = True
                        print(f"   🚨 WINNER banner '{winner_text}' is in upper half (y={winner_ny:.2f}) -> Opponent WON!")

            print(f"   [Frame {attempt+1}] Spatial OCR: {', '.join(token_logs[:12])}")

            # Synthesize Frame Verdict
            if player_winner_detected or player_trophy_gain:
                ocr_result = "WIN"
                winner_crowns = crown_candidate or 3
                break
            elif opp_winner_detected or player_trophy_loss:
                ocr_result = "LOSS"
                winner_crowns = crown_candidate or 3
                break

            time.sleep(0.5)

        # Step 1.5: If Friendly King was destroyed, override any OCR ambiguity -> 100% LOSS!
        if friendly_king_destroyed and not enemy_king_destroyed:
            print("🚨 [RESULT OVERRIDE] Friendly King Tower was destroyed in battle! Forcing LOSS.")
            ocr_result = "LOSS"
            winner_crowns = 3
        elif enemy_king_destroyed and not friendly_king_destroyed:
            print("🏆 [RESULT OVERRIDE] Enemy King Tower was destroyed in battle! Forcing 3-Crown WIN.")
            ocr_result = "WIN"
            winner_crowns = 3

        # Phase 2: Crown Evaluation
        if winner_crowns is None:
            winner_crowns = 1

        # Match Duration & Time-Variant Speed Bonus
        if battle_duration is not None and battle_duration > 0:
            match_time = battle_duration
        else:
            match_time = len(battle_steps) * 0.8

        speed_ratio = max(0.0, min(1.0, (180.0 - match_time) / 180.0))
        speed_bonus = 2.5 * speed_ratio

        # Phase 3: Final Outcome & Crown Assignment
        if ocr_result == "WIN":
            result = "WIN"
            max_my_crowns = winner_crowns
            max_op_crowns = 0
            base_reward = 2.5 + (0.5 * (max_my_crowns - 1))
            reward = base_reward + speed_bonus
            print(f"⚡ [TIME-REWARD] Blitzkrieg Win in {match_time:.1f}s! Base: +{base_reward:.2f}, Speed Bonus: +{speed_bonus:.2f} -> Total: +{reward:.2f}")

        elif ocr_result == "LOSS":
            result = "LOSS"
            max_my_crowns = 0
            max_op_crowns = winner_crowns
            reward = -2.5 - (0.3 * (max_op_crowns - 1))
            print(f"💀 [TIME-REWARD] Match Lost in {match_time:.1f}s. Penalty: {reward:.2f}")

        elif ocr_result == "DRAW":
            result = "DRAW"
            max_my_crowns = 0
            max_op_crowns = 0
            reward = -1.0
            print(f"⚖️ [TIME-REWARD] Match Draw in {match_time:.1f}s. Passivity Penalty: {reward:.2f}")

        else:
            # Conservative Damage Fallback: NEVER give 3-crown blitzkrieg win!
            print("⚠️ [RESULT] Banner text not definitively detected by OCR, checking damage metrics.")
            total_damage_dealt = sum(s.get('reward', 0.0) for s in battle_steps if s.get('reward', 0.0) > 0)
            total_damage_taken = sum(abs(s.get('reward', 0.0)) for s in battle_steps if s.get('reward', 0.0) < 0)
            print(f"📊 [RESULT] Battle Damage Analysis: Dealt={total_damage_dealt:.2f}, Taken={total_damage_taken:.2f}")

            if total_damage_dealt > (total_damage_taken * 1.5 + 1.0):
                # Strong verified damage lead: at most a modest 1-crown win (+2.0)
                result = "WIN"
                max_my_crowns = 1
                max_op_crowns = 0
                reward = 2.0
                print(f"⚡ [RESULT] Conservative Damage Win in {match_time:.1f}s! Reward: +{reward:.2f}")
            elif total_damage_taken >= total_damage_dealt:
                # Took more damage than dealt: Definite Loss
                result = "LOSS"
                max_my_crowns = 0
                max_op_crowns = 1
                reward = -2.5
                print(f"💀 [RESULT] Damage Loss in {match_time:.1f}s. Penalty: {reward:.2f}")
            else:
                result = "DRAW"
                max_my_crowns = 0
                max_op_crowns = 0
                reward = -1.0
                print(f"⚖️ [RESULT] Damage Draw in {match_time:.1f}s. Passivity Penalty: {reward:.2f}")

        print(f"🏆 [RESULT] Match Result: {result} (Player Crowns: {max_my_crowns} | Opponent Crowns: {max_op_crowns}) | Final Reward: {reward:+.2f}")
        return result, max_my_crowns, max_op_crowns, reward

    def get_crown_boxes(self, screen_pil):
        """Legacy compatibility wrapper."""
        c_me, c_op = self.get_crown_counts(screen_pil)
        return [None] * c_me, [None] * c_op
    
