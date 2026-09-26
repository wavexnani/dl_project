"""
=================================================================
  Crown Detection Debugger
=================================================================
Run this while the post-battle crown screen is visible in BlueStacks.
It will save annotated screenshots showing exactly what the template
matcher finds and print all matching scores.
=================================================================
Usage:
    python debug_crowns.py
=================================================================
"""
import cv2
import numpy as np
from PIL import ImageGrab
import os
from scaler import Scaler

def run_debug():
    print("Initializing scaler...")
    scaler = Scaler()

    game_area = scaler.game_area_rect
    bbox = (game_area[0], game_area[1], game_area[0] + game_area[2], game_area[1] + game_area[3])

    print("\n📸 Capturing current screen from BlueStacks game area...")
    screen_pil = ImageGrab.grab(bbox=bbox)
    screen_cv = cv2.cvtColor(np.array(screen_pil), cv2.COLOR_RGB2BGR)
    screen_cv_gray = cv2.cvtColor(screen_cv, cv2.COLOR_BGR2GRAY)

    print(f"   Screen size: {screen_pil.width}x{screen_pil.height}")

    # Save raw screenshot
    cv2.imwrite("crown_debug_raw.png", screen_cv)
    print("   Saved: crown_debug_raw.png")

    # Load crown templates (raw, unscaled)
    blue_raw = cv2.imread("sorted_data/anchors/bluecrowns.PNG", cv2.IMREAD_GRAYSCALE)
    red_raw  = cv2.imread("sorted_data/anchors/redcrowns.PNG",  cv2.IMREAD_GRAYSCALE)

    if blue_raw is None or red_raw is None:
        print("ERROR: Could not load crown templates!")
        return

    print(f"\n   Blue crown template (unscaled): {blue_raw.shape}")
    print(f"   Red crown template  (unscaled): {red_raw.shape}")
    print(f"   X scale: {scaler.x_scale:.3f}  Y scale: {scaler.y_scale:.3f}")

    # Scaled templates (same as scaler does)
    blue_t = scaler.scale_template("sorted_data/anchors/bluecrowns.PNG")
    red_t  = scaler.scale_template("sorted_data/anchors/redcrowns.PNG")
    print(f"\n   Blue crown template (scaled):   {blue_t.shape}")
    print(f"   Red crown template  (scaled):   {red_t.shape}")

    game_height = screen_pil.height
    game_width  = screen_pil.width

    print("\n" + "="*60)
    print("  SCANNING FULL IMAGE — Blue Crowns (your wins)")
    print("="*60)
    scan_template(screen_cv_gray, screen_cv, blue_t, "BLUE", (255, 165, 0), "crown_debug_blue_full.png")

    print("\n" + "="*60)
    print("  SCANNING FULL IMAGE — Red Crowns (opponent wins)")
    print("="*60)
    scan_template(screen_cv_gray, screen_cv.copy(), red_t, "RED", (0, 0, 255), "crown_debug_red_full.png")

    # --- ROI splits like the actual code uses ---
    my_roi_y = int(game_height * 0.35)
    op_roi_h = int(game_height * 0.50)

    my_roi_gray    = screen_cv_gray[my_roi_y:, :]
    my_roi_color   = screen_cv[my_roi_y:, :].copy()
    op_roi_gray    = screen_cv_gray[0:op_roi_h, :]
    op_roi_color   = screen_cv[0:op_roi_h, :].copy()

    print("\n" + "="*60)
    print(f"  ROI: Player side (y>{my_roi_y}) — Blue Crowns")
    print("="*60)
    my_count = scan_template(my_roi_gray, my_roi_color, blue_t, "BLUE", (255, 165, 0), "crown_debug_player_roi.png")

    print("\n" + "="*60)
    print(f"  ROI: Opponent side (y<{op_roi_h}) — Red Crowns")
    print("="*60)
    op_count = scan_template(op_roi_gray, op_roi_color, red_t, "RED", (0, 0, 255), "crown_debug_opponent_roi.png")

    print("\n" + "="*60)
    print(f"  RESULT: Player={my_count} crowns | Opponent={op_count} crowns")
    if my_count > op_count:
        print("  ✅ Correctly identifies as: WIN")
    elif op_count > my_count:
        print("  ✅ Correctly identifies as: LOSS")
    else:
        print("  ❌ Identifies as: DRAW — Crown templates may not match!")
        print("  → Check crown_debug_blue_full.png and crown_debug_red_full.png")
        print("  → Try lowering the threshold (currently at 0.65 minimum)")
    print("="*60)


def scan_template(gray, color_img, template, label, box_color, out_file, threshold=0.55):
    """Scans the entire gray image for the template, highlights all matches, prints scores."""
    if template is None:
        print("  Template is None — cannot scan.")
        return 0

    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    tw, th = template.shape[::-1]

    print(f"  Best match score: {max_val:.4f} at {max_loc}")
    print(f"  Threshold levels:  [0.82, 0.76, 0.70, 0.65, 0.55]")

    # Count at each threshold
    for thresh in [0.82, 0.76, 0.70, 0.65, 0.55]:
        locs = np.where(res >= thresh)
        count = len(locs[0])
        print(f"    thresh={thresh}: {count} pixel-level matches")

    # Annotate all detections at 0.55 threshold for visual debugging
    annotated = color_img.copy()
    locs_all = np.where(res >= threshold)
    for pt in zip(*locs_all[::-1]):
        cv2.rectangle(annotated, pt, (pt[0] + tw, pt[1] + th), box_color, 2)

    # Run NMS at 0.55
    boxes = [[pt[0], pt[1], pt[0] + tw, pt[1] + th] for pt in zip(*locs_all[::-1])]
    scores = [res[pt[1], pt[0]] for pt in zip(*locs_all[::-1])]

    count_nms = 0
    if boxes:
        boxes_np = np.array(boxes)
        scores_np = np.array(scores)
        pick = []
        x1, y1, x2, y2 = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores_np)
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
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.35)[0])))
        count_nms = len(pick)
        for i in pick:
            pt = (int(boxes_np[i, 0]), int(boxes_np[i, 1]))
            cv2.rectangle(annotated, pt, (pt[0] + tw, pt[1] + th), (0, 255, 0), 3)
            cv2.putText(annotated, f"{scores_np[i]:.2f}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"  After NMS at thresh=0.55: {count_nms} unique {label} crowns")
    cv2.imwrite(out_file, annotated)
    print(f"  Saved: {out_file}")
    return count_nms

if __name__ == "__main__":
    run_debug()
