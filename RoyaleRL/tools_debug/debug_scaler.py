"""Diagnostic script to debug BlueStacks game area detection."""
import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab
import os

print("=" * 60)
print("BLUESTACKS GAME AREA DIAGNOSTIC")
print("=" * 60)

# Step 1: Find BlueStacks window
all_windows = gw.getAllWindows()
bluestacks_windows = [w for w in all_windows if 'bluestacks' in w.title.lower() and w.title.strip()]

if not bluestacks_windows:
    print("ERROR: No BlueStacks window found!")
    print("Available windows:")
    for w in all_windows:
        if w.title.strip():
            print(f"  '{w.title}' -- pos=({w.left},{w.top}) size=({w.width}x{w.height})")
    exit(1)

win = bluestacks_windows[0]
print(f"\nFound: '{win.title}'")
print(f"  Position: left={win.left}, top={win.top}")
print(f"  Size: {win.width} x {win.height}")
print(f"  Visible: {win.visible}")
print(f"  Minimized: {win.isMinimized}")
print(f"  Maximized: {win.isMaximized}")

# Bring window to foreground and wait for paint
try:
    import ctypes
    user32 = ctypes.windll.user32
    user32.keybd_event(0x12, 0, 0, 0)
    user32.ShowWindow(win._hWnd, 9)
    user32.SetForegroundWindow(win._hWnd)
    user32.keybd_event(0x12, 0, 2, 0)
    import time
    time.sleep(1.0)
except Exception as e:
    print(f"Notice: Could not automatically activate window: {e}")

# Step 2: Take screenshot of BlueStacks area
bbox = (win.left, win.top, win.left + win.width, win.top + win.height)
print(f"\nCapturing screenshot of bbox: {bbox}")

screenshot_pil = ImageGrab.grab(bbox=bbox)
screenshot_np = np.array(screenshot_pil)
print(f"Screenshot shape: {screenshot_np.shape} (height x width x channels)")

# Save the raw screenshot
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_images")
os.makedirs(output_dir, exist_ok=True)
raw_path = os.path.join(output_dir, "debug_screenshot_raw.png")
cv2.imwrite(raw_path, cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR))
print(f"Saved raw screenshot to: {raw_path}")

# Step 3: Reproduce the scaler's detection logic
print("\n--- Reproducing Scaler Detection ---")

# Blue scan for top edge
screenshot_hsv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(screenshot_hsv, lower_blue, upper_blue)

center_x = screenshot_np.shape[1] // 2
top_edge_y = 0
for y in range(screenshot_np.shape[0]):
    if blue_mask[y, center_x] == 0:
        top_edge_y = y
        break

print(f"Blue scan: center_x={center_x}, top_edge_y={top_edge_y}")

# Contour detection
cropped_screenshot = screenshot_np[top_edge_y + 100:, :]
cropped_screenshot_gray = cv2.cvtColor(cropped_screenshot, cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(cropped_screenshot_gray, 10, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours total")
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

for i, c in enumerate(contours_sorted):
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    print(f"  Contour {i}: x={x}, y={y}, w={w}, h={h}, area={area}")

# The scaler takes top 2 contours
if len(contours_sorted) >= 2:
    top2 = contours_sorted[:2]
    bounding_boxes = [cv2.boundingRect(c) for c in top2]
    left_box = min(bounding_boxes, key=lambda b: b[0])
    right_box = max(bounding_boxes, key=lambda b: b[0])
    
    x_start = left_box[0] + left_box[2]
    width = right_box[0] - x_start
    final_height = win.height - top_edge_y
    
    print(f"\n  Left box: x={left_box[0]}, w={left_box[2]} -> right edge at {left_box[0]+left_box[2]}")
    print(f"  Right box: x={right_box[0]}")
    print(f"  Calculated game area: x_start={x_start}, width={width}, height={final_height}")
    
    if width < 300:
        print(f"\n  WARNING: WIDTH={width} IS TOO NARROW! Expected ~500-600px.")
        print("  The contour detection is not finding the correct game boundaries.")

# Save annotated debug image
debug_img = screenshot_np.copy()
cv2.line(debug_img, (center_x, 0), (center_x, screenshot_np.shape[0]), (0, 255, 0), 2)
cv2.line(debug_img, (0, top_edge_y), (screenshot_np.shape[1], top_edge_y), (255, 0, 0), 2)
for i, c in enumerate(contours_sorted[:5]):
    x, y, w, h = cv2.boundingRect(c)
    y_abs = y + top_edge_y + 100
    color = (0, 255, 255) if i < 2 else (128, 128, 128)
    cv2.rectangle(debug_img, (x, y_abs), (x + w, y_abs + h), color, 2)
    cv2.putText(debug_img, f"C{i}: {w}x{h}", (x, y_abs - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

debug_path = os.path.join(output_dir, "debug_screenshot_annotated.png")
cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
print(f"\nSaved annotated debug image to: {debug_path}")

thresh_path = os.path.join(output_dir, "debug_threshold.png")
cv2.imwrite(thresh_path, thresh)
print(f"Saved threshold image to: {thresh_path}")

print("\n" + "=" * 60)
print("Check the debug images in the RoyaleRL folder!")
print("=" * 60)
