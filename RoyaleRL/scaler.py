# filename: scaler.py
import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
from PIL import ImageGrab
import config
import os # Make sure this is imported

class Scaler:
    def __init__(self):
        # Attach thread to interactive desktop if running in background subshell
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hdesk = user32.OpenDesktopW('default', 0, False, 0x0100 | 0x0040 | 0x0001 | 0x0008 | 0x0002)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

        # Find the BlueStacks window by a flexible title search
        try:
            bluestacks_windows = gw.getWindowsWithTitle('BlueStacks App Player')
            if not bluestacks_windows:
                # Try finding any window with 'bluestacks' in the title
                all_windows = gw.getAllWindows()
                bluestacks_windows = [w for w in all_windows if 'bluestacks' in w.title.lower()]
                
            if not bluestacks_windows:
                available = [w.title for w in gw.getAllWindows() if w.title.strip()]
                raise ValueError(f"No BlueStacks window found. Detected windows: {available}")
            
            bluestacks_window = bluestacks_windows[0]
            print(f"Found BlueStacks window: '{bluestacks_window.title}'")
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.keybd_event(0x12, 0, 0, 0)
                user32.ShowWindow(bluestacks_window._hWnd, 9)
                user32.SetForegroundWindow(bluestacks_window._hWnd)
                user32.keybd_event(0x12, 0, 2, 0)
                import time
                time.sleep(1.0)
            except Exception as e:
                print(f"Notice: Could not bring window to foreground: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"BlueStacks window not found: {e}")

        # Find the game area within the BlueStacks window 
        game_area = self._find_game_area_by_combined_methods(bluestacks_window)
        
        if not game_area:
            raise ValueError("Could not find the game area within the BlueStacks window.")

        # The current resolution is the dimensions of the detected game area
        self.game_area_rect = game_area
        self.current_resolution = (self.game_area_rect[2], self.game_area_rect[3])
        
        # Calculate scaling factors based on the game area dimensions
        self.x_scale = self.current_resolution[0] / config.REFERENCE_RESOLUTION[0]
        self.y_scale = self.current_resolution[1] / config.REFERENCE_RESOLUTION[1]
        
        print(f"Scaler initialized. Detected Clash Royale resolution: {self.current_resolution}")
        print(f"Scaling factors: X={self.x_scale:.2f}, Y={self.y_scale:.2f}")

    def _find_game_area_by_combined_methods(self, bluestacks_window):

        bluestacks_rect = (bluestacks_window.left, bluestacks_window.top, bluestacks_window.width, bluestacks_window.height)
        # ImageGrab.grab needs (left, top, right, bottom)
        grab_bbox = (bluestacks_window.left, bluestacks_window.top, 
                     bluestacks_window.left + bluestacks_window.width, 
                     bluestacks_window.top + bluestacks_window.height)
        screenshot_pil = ImageGrab.grab(bbox=grab_bbox)
        screenshot_np = np.array(screenshot_pil)
        
        # --- Find the Top Boundary using Color Scan ---
        # The BlueStacks toolbar at the top is blue; scan down from center to find where it ends
        screenshot_hsv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(screenshot_hsv, lower_blue, upper_blue)
        center_x = screenshot_np.shape[1] // 2
        top_edge_y = 0
        try:
            for y in range(screenshot_np.shape[0]):
                if blue_mask[y, center_x] == 0:
                    top_edge_y = y
                    break
        except IndexError: pass
        if top_edge_y == 0:
            print("Could not find the top edge via color scan. Assuming fullscreen/borderless.")
            top_edge_y = 0

        # --- Try to Find Horizontal Boundaries using Contours ---
        # This works when BlueStacks is fullscreened on a widescreen monitor with black bars
        cropped_screenshot = screenshot_np[top_edge_y + 100:, :]
        cropped_screenshot_gray = cv2.cvtColor(cropped_screenshot, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(cropped_screenshot_gray, 10, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        x_start = 0
        width = bluestacks_window.width
        
        if len(contours) >= 2:
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            left_box = min(bounding_boxes, key=lambda b: b[0])
            right_box = max(bounding_boxes, key=lambda b: b[0])
            
            candidate_x_start = left_box[0] + left_box[2]
            candidate_width = right_box[0] - candidate_x_start
            
            # Only use contour-based boundaries if they make sense
            # (width should be at least 40% of window width to be a real game area)
            if candidate_width > bluestacks_window.width * 0.4:
                x_start = candidate_x_start
                width = candidate_width
                print(f"Game area detected via contour method: x_start={x_start}, width={width}")
            else:
                print(f"Contour detection found narrow area ({candidate_width}px). "
                      f"Falling back to full window mode.")
        else:
            print("Not enough contours found. Using full window mode.")
            
        # --- Calculate Final Bounding Box ---
        final_height = bluestacks_window.height - top_edge_y
        
        print(f"Game area: x={bluestacks_window.left + x_start}, y={bluestacks_window.top + top_edge_y}, "
              f"w={width}, h={final_height}")

        return (bluestacks_window.left + x_start, bluestacks_window.top + top_edge_y, width, final_height)

    def scale_coords(self, coords):
        """Scales coordinates relative to the found game area."""
        return (int((coords[0] - self.game_area_rect[0]) * self.x_scale), 
                int((coords[1] - self.game_area_rect[1]) * self.y_scale))
    
    def scale_box(self, box):
        """Scales a box tuple (x, y, w, h)."""
        x, y, w, h = box
        scaled_x = int(x * self.x_scale)
        scaled_y = int(y * self.y_scale)
        scaled_w = int(w * self.x_scale)
        scaled_h = int(h * self.y_scale)
        return (scaled_x, scaled_y, scaled_w, scaled_h)

    def scale_template(self, template_path):
        """Loads and scales a template image."""
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Template not found at {template_path}")
        
        new_w = int(template.shape[1] * self.x_scale)
        new_h = int(template.shape[0] * self.y_scale)
        
        scaled_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled_template
