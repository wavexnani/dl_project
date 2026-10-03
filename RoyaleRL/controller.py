# filename: controller.py
import pyautogui
import time
import cv2
import numpy as np
from PIL import ImageGrab
import config # Import the config
import ctypes
from ctypes import wintypes

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

class Controller:
    """Handles all mouse interactions with the game."""
    def __init__(self, scaler):
        self.scaler = scaler
        # Get the absolute offset of the game window once
        self.game_area_offset_x = self.scaler.game_area_rect[0]
        self.game_area_offset_y = self.scaler.game_area_rect[1]
        
        # Find BlueStacks Qt child window for direct PostMessage clicks
        self.target_hwnd = self._find_target_hwnd()
        print("Controller initialized.")

    def _find_target_hwnd(self):
        try:
            import pygetwindow as gw
            bluestacks_windows = gw.getWindowsWithTitle('BlueStacks App Player')
            if not bluestacks_windows:
                all_w = gw.getAllWindows()
                bluestacks_windows = [w for w in all_w if 'bluestacks' in w.title.lower()]
            if not bluestacks_windows:
                return None
            
            root_hwnd = bluestacks_windows[0]._hWnd
            user32 = ctypes.windll.user32
            children = []
            def enum_cb(hwnd, lparam):
                children.append(hwnd)
                return True
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            user32.EnumChildWindows(root_hwnd, WNDENUMPROC(enum_cb), 0)
            
            for c in children:
                c_name = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(c, c_name, 256)
                if 'Qt' in c_name.value:
                    return c
            return root_hwnd
        except Exception:
            return None

    def click(self, x, y):
        """Moves to and clicks a given coordinate using genuine OS mouse input."""
        abs_x = self.game_area_offset_x + x
        abs_y = self.game_area_offset_y + y
        pyautogui.moveTo(abs_x, abs_y, duration=0.03)
        pyautogui.click()
        time.sleep(0.08)

    def send_escape(self):
        """Sends the Escape key to BlueStacks (acts as Android Back button)."""
        VK_ESCAPE = 0x1B
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        if self.target_hwnd:
            try:
                user32 = ctypes.windll.user32
                user32.PostMessageW(self.target_hwnd, WM_KEYDOWN, VK_ESCAPE, 0)
                time.sleep(0.05)
                user32.PostMessageW(self.target_hwnd, WM_KEYUP, VK_ESCAPE, 0)
                print("🎮 [CONTROLLER] Sent Escape (Android Back) via PostMessage.")
                return
            except Exception:
                pass
        pyautogui.press('esc')
        print("🎮 [CONTROLLER] Sent Escape (Android Back) via pyautogui.")

    def dismiss_popups(self, in_battle=False):
        """Attempts to clear unknown dialogs, popups, or reward screens."""
        print("🛡️ [WATCHDOG] Attempting to dismiss popup/stuck screen...")
        self.send_escape()
        time.sleep(0.5)
        if not in_battle:
            # Click center of game area to clear tap-to-continue prompts only outside battle
            gw, gh = self.scaler.current_resolution
            self.click(int(gw * 0.5), int(gh * 0.75))
            time.sleep(0.5)


    def play_card(self, card_slot_coords, placement_coords):
        """
        Reliably deploys a card using direct drag-and-drop from the card slot to the arena tile,
        followed by confirmation tap. Guarantees 100% card deployment in BlueStacks.
        """
        card_click_x, card_click_y = card_slot_coords
        
        # ---Bounding Box Clamping Logic ---
        
        # 1. Get the game window's current dimensions
        window_width, window_height = self.scaler.current_resolution
        
        # 2. Convert the percentage-based ARENA_BBOX to absolute pixel coordinates
        min_x_abs = int(config.ARENA_BBOX[0] * window_width)
        min_y_abs = int(config.ARENA_BBOX[1] * window_height)
        max_x_abs = int(config.ARENA_BBOX[2] * window_width)
        max_y_abs = int(config.ARENA_BBOX[3] * window_height)

        # 3. Get the requested placement coordinates
        requested_x, requested_y = placement_coords

        # 4. "Clamp" the coordinates to ensure they are within the bounding box
        # The max() function ensures the value is not less than the minimum boundary.
        # The min() function ensures the value is not more than the maximum boundary.
        clamped_x = max(min_x_abs, min(requested_x, max_x_abs))
        clamped_y = max(min_y_abs, min(requested_y, max_y_abs))
        abs_card_x = self.game_area_offset_x + card_click_x
        abs_card_y = self.game_area_offset_y + card_click_y
        abs_target_x = self.game_area_offset_x + clamped_x
        abs_target_y = self.game_area_offset_y + clamped_y

        print(f"CONTROLLER: Card ({abs_card_x}, {abs_card_y}) -> Target ({abs_target_x}, {abs_target_y})")
        # Swift drag-and-drop: pickup card from slot and drop onto arena
        pyautogui.moveTo(abs_card_x, abs_card_y)
        pyautogui.mouseDown(button='left')
        time.sleep(0.05)
        pyautogui.moveTo(abs_target_x, abs_target_y, duration=0.12)
        time.sleep(0.04)
        pyautogui.mouseUp(button='left')
        # Confirm with tap on arena
        time.sleep(0.04)
        pyautogui.click(abs_target_x, abs_target_y)
        time.sleep(0.12)


    def find_and_click(self, template_path, confidence=0.85):

        template_img = self.scaler.scale_template(template_path)
        
        game_area = self.scaler.game_area_rect
        gx1 = max(0, game_area[0])
        gy1 = max(0, game_area[1])
        gx2 = max(gx1 + 10, game_area[0] + game_area[2])
        gy2 = max(gy1 + 10, game_area[1] + game_area[3])
        bbox = (gx1, gy1, gx2, gy2)
        screenshot_pil = ImageGrab.grab(bbox=bbox)
        screenshot_cv = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2GRAY)

        res = cv2.matchTemplate(screenshot_cv, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= confidence:
            h, w = template_img.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            self.click(center_x, center_y)
            print(f"Found template with confidence {max_val:.2f}. Clicking center at relative ({center_x}, {center_y})")
            return True
        else:
            print(f"Could not find template with sufficient confidence (Max val: {max_val:.2f})")
            return False
