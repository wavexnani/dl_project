"""
RoyaleRL Architecture Diagram Generator using Matplotlib
Configured for exact dimensions: 1080 x 1720 px (fully customizable)

Usage via Terminal:
    python generate_diagram.py --width 1080 --height 1720 --output royale_rl_architecture_1080x1720.png

Usage in Python:
    from generate_diagram import generate_royale_rl_diagram
    generate_royale_rl_diagram(width_px=1080, height_px=1720)
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from PIL import Image

def generate_royale_rl_diagram(
    output_path="royale_rl_architecture_1080x1720.png",
    width_px=1080,
    height_px=1720,
    dpi=100
):
    fig_w = width_px / dpi
    fig_h = height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    
    # Pixel-coordinate axes (0 to width_px, 0 to height_px)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis('off')
    
    # Primary Background (Ultra-dark sleek cyber theme)
    bg_color = "#070b14"
    ax.fill([0, width_px, width_px, 0], [0, 0, height_px, height_px], color=bg_color)
    
    # Outer container rounded frame
    frame_margin_x = 24
    frame_margin_y = 28
    frame_w = width_px - 2 * frame_margin_x
    frame_h = height_px - 2 * frame_margin_y
    
    outer_box = FancyBboxPatch(
        (frame_margin_x, frame_margin_y), frame_w, frame_h,
        boxstyle="round,pad=0,rounding_size=18",
        facecolor="#090f1e", edgecolor="#1e293b", linewidth=1.5, zorder=1
    )
    ax.add_patch(outer_box)
    
    # ==========================================
    # HEADER / TITLE BAR
    # ==========================================
    title_x = frame_margin_x + 28
    title_y = height_px - frame_margin_y - 36
    
    # Cyan glowing dot badge
    dot_glow = plt.Circle((title_x, title_y), 9, facecolor="#00e5ff", alpha=0.25, zorder=4)
    dot = plt.Circle((title_x, title_y), 6, facecolor="#00f0ff", edgecolor="#ffffff", linewidth=1.2, zorder=5)
    ax.add_patch(dot_glow)
    ax.add_patch(dot)
    
    # Title Text
    ax.text(
        title_x + 18, title_y,
        "RoyaleRL - Autonomous Match & Decision Workflow",
        color="#ffffff", fontsize=14, fontweight="bold", va="center", ha="left",
        fontfamily="sans-serif", zorder=5
    )
    
    # Dimension specs tag
    ax.text(
        frame_margin_x + frame_w - 24, title_y,
        f"{width_px} × {height_px} px • Target Canvas",
        color="#64748b", fontsize=9, fontweight="bold", va="center", ha="right",
        fontfamily="sans-serif", zorder=5
    )

    # ==========================================
    # COLOR THEME DEFINITIONS
    # ==========================================
    STYLES = {
        "external": {"border": "#3b82f6", "bg": "#0d1a33", "tag": "#60a5fa", "badge": "EXT", "icon": "◫"},
        "backend": {"border": "#10b981", "bg": "#09241d", "tag": "#34d399", "badge": "AGENT", "icon": "◈"},
        "database": {"border": "#8b5cf6", "bg": "#1e1338", "tag": "#c084fc", "badge": "STATE", "icon": "▤"},
        "security": {"border": "#f43f5e", "bg": "#2b101c", "tag": "#fb7185", "badge": "TACTIC", "icon": "▲"},
        "frontend": {"border": "#06b6d4", "bg": "#09232d", "tag": "#22d3ee", "badge": "ARBITER", "icon": "❖"},
    }

    # ==========================================
    # LAYOUT GRID DEFINITIONS
    # ==========================================
    grid_left = frame_margin_x + 18
    grid_w = frame_w - 36
    
    # Optimally calibrated column centers:
    # Generous gaps between col 3 & 4 (for policy action & threat level)
    # Balanced right margin for col 5 (so match diff is safely within frame)
    col_centers = [
        grid_left + 68,           # col 0: Screen Capture
        grid_left + 220,          # col 1: Vision YOLOv8
        grid_left + 376,          # col 2: State Manager
        grid_left + 546,          # col 3: Threat Matrix & Actor-Critic PPO
        grid_left + 746,          # col 4: Emergency Gate & Action Arbiter
        grid_left + 910,          # col 5: Reward Tracker & ADB Dispatcher
    ]
    
    # Phase bar above lanes
    phase_y = title_y - 36
    phase_h = 24
    
    phases = [
        (grid_left, (col_centers[2] + 70) - grid_left, "Observation", "#00e5ff", "#082133"),
        (col_centers[3] - 78, (col_centers[4] + 78) - (col_centers[3] - 78), "Decision & Arbitration", "#a855f7", "#1f1238"),
        (col_centers[5] - 78, grid_left + grid_w - (col_centers[5] - 78), "Execution & Feedback", "#ec4899", "#2d1028")
    ]
    
    for px, pw, plabel, pcolor, pbg in phases:
        pbox = FancyBboxPatch(
            (px, phase_y - phase_h/2), pw, phase_h,
            boxstyle="round,pad=0,rounding_size=6",
            facecolor=pbg, edgecolor=pcolor, linewidth=1, zorder=3, alpha=0.9
        )
        ax.add_patch(pbox)
        ax.text(
            px + pw/2, phase_y, plabel,
            color=pcolor, fontsize=9.2, fontweight="bold", va="center", ha="center", zorder=5
        )

    # 4 Horizontal Lanes
    lanes_top = phase_y - 20
    lanes_bottom = 575
    total_lanes_h = lanes_top - lanes_bottom
    lane_h = total_lanes_h / 4
    
    lane_defs = [
        ("perception", "01 / Perception & State Space", "#1e293b", "#0d1527", None),
        ("tactics", "EX / Tactical Rules & Defense", "#e11d48", "#1c0d18", "dashed"),
        ("policy", "03 / Deep RL Strategy & Planning", "#1e293b", "#0d1527", None),
        ("controller", "04 / Action Execution & ADB", "#1e293b", "#0d1527", None),
    ]
    
    lane_bounds = {}
    for idx, (lid, lname, lcolor, lbg, lstyle) in enumerate(lane_defs):
        ly = lanes_top - (idx + 1) * lane_h
        lane_bounds[lid] = (ly, lane_h)
        
        # Lane container box
        l_patch = FancyBboxPatch(
            (grid_left, ly + 5), grid_w, lane_h - 10,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor="#0b1220", edgecolor=lcolor,
            linewidth=1.2 if lstyle else 0.8,
            linestyle="--" if lstyle else ":",
            alpha=0.6 if not lstyle else 0.85,
            zorder=2
        )
        ax.add_patch(l_patch)
        
        # Lane label
        ax.text(
            grid_left + 14, ly + lane_h - 18, lname,
            color="#fb7185" if lstyle else "#64748b",
            fontsize=9.2, fontweight="bold", va="center", ha="left", zorder=4
        )

    # Sub-groups (dashed grouping boxes)
    groups = [
        ("Vision Pipeline", grid_left + 6, lane_bounds["perception"][0] + 10, (col_centers[1] - col_centers[0]) + 144, lane_h - 40, "#00e5ff"),
        ("Rule-Based Defense", col_centers[3] - 74, lane_bounds["tactics"][0] + 10, (col_centers[4] - col_centers[3]) + 148, lane_h - 40, "#f43f5e"),
        ("Policy Evaluation", col_centers[3] - 74, lane_bounds["policy"][0] + 10, 148, lane_h - 40, "#10b981"),
        ("Device Interaction", col_centers[5] - 74, lane_bounds["controller"][0] + 10, 148, lane_h - 40, "#10b981"),
    ]
    
    for gname, gx, gy, gw, gh, gcol in groups:
        gbox = FancyBboxPatch(
            (gx, gy), gw, gh,
            boxstyle="round,pad=0,rounding_size=8",
            facecolor="none", edgecolor=gcol, linewidth=0.9, linestyle="--",
            alpha=0.45, zorder=2
        )
        ax.add_patch(gbox)
        ax.text(
            gx + 10, gy + gh - 10, gname,
            color=gcol, fontsize=7.2, fontweight="bold", va="center", ha="left", zorder=3, alpha=0.95
        )

    # ==========================================
    # NODES DEFINITIONS & DRAWING
    # ==========================================
    node_w = 122
    node_h = 74
    
    def get_node_y(lane_id):
        ly, lh = lane_bounds[lane_id]
        return ly + (lh - node_h) / 2 - 8
        
    nodes = {
        "screencap": {
            "lane": "perception", "col_idx": 0, "type": "external",
            "title": "Screen Capture", "sub": "PIL / Scrcpy frame", "tag": None
        },
        "vision_yolo": {
            "lane": "perception", "col_idx": 1, "type": "backend",
            "title": "Vision YOLOv8", "sub": "troops, cards, HP", "tag": None
        },
        "state_mgr": {
            "lane": "perception", "col_idx": 2, "type": "database",
            "title": "State Manager", "sub": "114-D vector & elixir", "tag": "normalized"
        },
        "threat_eval": {
            "lane": "tactics", "col_idx": 3, "type": "security",
            "title": "Threat Matrix", "sub": "bridge push / tank", "tag": "reactive"
        },
        "emergency_gate": {
            "lane": "tactics", "col_idx": 4, "type": "security",
            "title": "Emergency Gate", "sub": "defense required?", "tag": None
        },
        "rl_eval": {
            "lane": "policy", "col_idx": 3, "type": "backend",
            "title": "Actor-Critic PPO", "sub": "action probs π(a|s)", "tag": "macro policy"
        },
        "arbitrator": {
            "lane": "policy", "col_idx": 4, "type": "frontend",
            "title": "Action Arbiter", "sub": "rule vs neural pick", "tag": None
        },
        "touch_dispatcher": {
            "lane": "controller", "col_idx": 5, "type": "backend",
            "title": "ADB Dispatcher", "sub": "drag card to (x,y)", "tag": None
        },
        "reward_tracker": {
            "lane": "policy", "col_idx": 5, "type": "database",
            "title": "Reward Tracker", "sub": "crown Δ & tower HP", "tag": "buffer store"
        },
    }
    
    node_coords = {}
    
    for nid, ninfo in nodes.items():
        cx = col_centers[ninfo["col_idx"]]
        nx = cx - node_w / 2
        ny = get_node_y(ninfo["lane"])
        node_coords[nid] = {
            "x": nx, "y": ny, "cx": cx, "cy": ny + node_h / 2,
            "w": node_w, "h": node_h,
            "left": nx, "right": nx + node_w,
            "top": ny + node_h, "bottom": ny,
            "type": ninfo["type"]
        }
        
        style = STYLES[ninfo["type"]]
        
        # Node Card
        card = FancyBboxPatch(
            (nx, ny), node_w, node_h,
            boxstyle="round,pad=0,rounding_size=8",
            facecolor=style["bg"], edgecolor=style["border"],
            linewidth=1.5, zorder=4
        )
        ax.add_patch(card)
        
        # Node Icon badge / pill
        badge_w = 18
        badge_box = FancyBboxPatch(
            (nx + 8, ny + node_h - 21), badge_w, 13,
            boxstyle="round,pad=0,rounding_size=3",
            facecolor="#000000", edgecolor=style["border"], linewidth=0.7, zorder=5
        )
        ax.add_patch(badge_box)
        ax.text(
            nx + 8 + badge_w/2, ny + node_h - 14.5, style["icon"],
            color=style["tag"], fontsize=7.2, fontweight="bold", va="center", ha="center", zorder=6
        )
        
        # Node Title
        ax.text(
            nx + 31, ny + node_h - 14.5, ninfo["title"],
            color="#ffffff", fontsize=8.6, fontweight="bold", va="center", ha="left", zorder=5
        )
        
        # Node Sublabel
        ax.text(
            nx + 10, ny + node_h - 36, ninfo["sub"],
            color="#94a3b8", fontsize=7.3, va="center", ha="left", zorder=5
        )
        
        # Optional tag badge (e.g. normalized, reactive, macro policy, buffer store)
        if ninfo["tag"]:
            tag_text = ninfo["tag"]
            tw = len(tag_text) * 5.2 + 10
            tbox = FancyBboxPatch(
                (nx + (node_w - tw)/2, ny + 8), tw, 14,
                boxstyle="round,pad=0,rounding_size=3",
                facecolor="#060c18", edgecolor=style["border"], linewidth=0.8, zorder=5
            )
            ax.add_patch(tbox)
            ax.text(
                nx + node_w/2, ny + 15, tag_text,
                color=style["tag"], fontsize=6.6, fontweight="medium", va="center", ha="center", zorder=6
            )

    # ==========================================
    # ARROWS & CONNECTORS
    # ==========================================
    def draw_straight_arrow(x1, y1, x2, y2, label="", color="#64748b", style="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=1.4, linestyle=style,
                shrinkA=2, shrinkB=3, mutation_scale=11
            ),
            zorder=3
        )
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            bbox_props = dict(boxstyle="round,pad=0.22", fc="#080e1a", ec=color, lw=0.7, alpha=0.96)
            ax.text(
                mx, my, label,
                color=color, fontsize=7.0, fontweight="bold",
                va="center", ha="center", bbox=bbox_props, zorder=6
            )
            
    def draw_step_arrow(pts, label="", color="#64748b", style="-", label_pt=None):
        for i in range(len(pts) - 2):
            ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]], color=color, linestyle=style, lw=1.3, zorder=3)
        p_pre = pts[-2]
        p_end = pts[-1]
        ax.annotate(
            "", xy=p_end, xytext=p_pre,
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=1.3, linestyle=style,
                shrinkA=0, shrinkB=3, mutation_scale=11
            ),
            zorder=3
        )
        if label and label_pt:
            bbox_props = dict(boxstyle="round,pad=0.22", fc="#080e1a", ec=color, lw=0.7, alpha=0.96)
            ax.text(
                label_pt[0], label_pt[1], label,
                color=color, fontsize=7.0, fontweight="bold",
                va="center", ha="center", bbox=bbox_props, zorder=6
            )

    # 1. Screen Capture -> Vision YOLOv8 [frame]
    c_cap = node_coords["screencap"]
    c_yolo = node_coords["vision_yolo"]
    draw_straight_arrow(c_cap["right"], c_cap["cy"], c_yolo["left"], c_yolo["cy"], label="frame", color="#64748b")
    
    # 2. Vision YOLOv8 -> State Manager [boxes]
    c_mgr = node_coords["state_mgr"]
    draw_straight_arrow(c_yolo["right"], c_yolo["cy"], c_mgr["left"], c_mgr["cy"], label="boxes", color="#64748b")
    
    # 3. State Manager -> Threat Matrix [board state]
    c_threat = node_coords["threat_eval"]
    sm_exit_x = c_mgr["cx"] + 18
    pts_threat = [
        (sm_exit_x, c_mgr["bottom"]),
        (sm_exit_x, c_threat["cy"] + 8),
        (c_threat["left"], c_threat["cy"] + 8)
    ]
    draw_step_arrow(pts_threat, label="board state", color="#f43f5e", style="--", label_pt=(sm_exit_x + 40, c_threat["cy"] + 8))
    
    # 4. Threat Matrix -> Emergency Gate [threat level]
    c_gate = node_coords["emergency_gate"]
    draw_straight_arrow(c_threat["right"], c_threat["cy"], c_gate["left"], c_gate["cy"], label="threat level", color="#f43f5e")
    
    # 5. State Manager -> Actor-Critic PPO [state vector]
    c_ppo = node_coords["rl_eval"]
    sm_exit_x2 = c_mgr["cx"] - 14
    pts_ppo = [
        (sm_exit_x2, c_mgr["bottom"]),
        (sm_exit_x2, c_ppo["bottom"] - 18),
        (c_ppo["cx"], c_ppo["bottom"] - 18),
        (c_ppo["cx"], c_ppo["bottom"])
    ]
    draw_step_arrow(pts_ppo, label="state vector", color="#10b981", style="-", label_pt=((sm_exit_x2 + c_ppo["cx"])/2, c_ppo["bottom"] - 18))
    
    # 6. Emergency Gate -> Action Arbiter [counter override]
    c_arb = node_coords["arbitrator"]
    draw_straight_arrow(c_gate["cx"], c_gate["bottom"], c_arb["cx"], c_arb["top"], label="counter override", color="#f43f5e", style="--")
    
    # 7. Actor-Critic PPO -> Action Arbiter [policy action]
    draw_straight_arrow(c_ppo["right"], c_ppo["cy"], c_arb["left"], c_arb["cy"], label="policy action", color="#10b981")
    
    # 8. Action Arbiter -> ADB Dispatcher [execute action]
    c_adb = node_coords["touch_dispatcher"]
    arb_exit_x = c_arb["cx"] + 15
    pts_dispatch = [
        (arb_exit_x, c_arb["bottom"]),
        (arb_exit_x, c_adb["cy"]),
        (c_adb["left"], c_adb["cy"])
    ]
    draw_step_arrow(pts_dispatch, label="execute action", color="#10b981", style="-", label_pt=(arb_exit_x + 42, c_adb["cy"]))
    
    # 9. ADB Dispatcher -> Reward Tracker [match diff]
    # Clean step up to Reward Tracker safely inside canvas with label centered directly on the vertical trace
    c_rew = node_coords["reward_tracker"]
    turn_x = c_adb["right"] + 18
    pts_diff = [
        (c_adb["right"], c_adb["cy"]),
        (turn_x, c_adb["cy"]),
        (turn_x, c_rew["cy"]),
        (c_rew["right"], c_rew["cy"])
    ]
    draw_step_arrow(pts_diff, label="match diff", color="#64748b", style="-", label_pt=(turn_x, (c_adb["cy"] + c_rew["cy"])/2))
    
    # 10. Reward Tracker -> State Manager [store transition]
    top_loop_y = lane_bounds["perception"][0] + lane_h - 14
    pts_loop = [
        (c_rew["cx"], c_rew["top"]),
        (c_rew["cx"], top_loop_y),
        (c_mgr["cx"] + 35, top_loop_y),
        (c_mgr["cx"] + 35, c_mgr["top"])
    ]
    for i in range(len(pts_loop) - 2):
        ax.plot([pts_loop[i][0], pts_loop[i+1][0]], [pts_loop[i][1], pts_loop[i+1][1]], color="#8b5cf6", linestyle="--", lw=1.3, zorder=3)
    ax.annotate(
        "", xy=pts_loop[-1], xytext=pts_loop[-2],
        arrowprops=dict(
            arrowstyle="-|>", color="#8b5cf6", linewidth=1.3, linestyle="--",
            shrinkA=0, shrinkB=3, mutation_scale=11
        ), zorder=3
    )
    bbox_loop = dict(boxstyle="round,pad=0.22", fc="#080e1a", ec="#8b5cf6", lw=0.7, alpha=0.96)
    ax.text((c_rew["cx"] + c_mgr["cx"])/2 + 35, top_loop_y, "store transition", color="#c084fc", fontsize=7.0, fontweight="bold", va="center", ha="center", bbox=bbox_loop, zorder=6)

    # ==========================================
    # BOTTOM SECTION: ARCHITECTURE HIGHLIGHT CARDS
    # ==========================================
    card_area_y = 114
    card_area_h = 425
    card_gap = 14
    card_w = (grid_w - 2 * card_gap) / 3
    
    cards_data = [
        {
            "color": "#00e5ff",
            "title": "Perception Loop",
            "badge": "VISION",
            "items": [
                "vision.py captures 30 FPS emulator frames via scrcpy/ADB screen stream.",
                "YOLOv8 detects troops, friendly towers, opponent crown HP, and card cooldowns.",
                "State Manager normalizes board state into 114-D numerical feature tensor for agents."
            ]
        },
        {
            "color": "#f43f5e",
            "title": "Tactical Prioritization",
            "badge": "DEFENSE",
            "items": [
                "TacticalBrain prioritizes immediate defensive threats (e.g. Hog Rider / Goblin Barrel).",
                "Emergency Gate overrides neural network when deterministic hard counters are mandatory.",
                "Maintains elixir advantage and enforces bridge push safety thresholds."
            ]
        },
        {
            "color": "#10b981",
            "title": "RL Policy & Feedback",
            "badge": "NEURAL PPO",
            "items": [
                "Actor-Critic network outputs macro play distributions pi(a|s) across valid tile coordinates.",
                "Reward Tracker measures incremental crown delta, tower preservation, and card cycling efficiency.",
                "Stores (s, a, r, s') experience transitions into replay buffer for PPO policy training."
            ]
        }
    ]
    
    for c_idx, cdata in enumerate(cards_data):
        cx = grid_left + c_idx * (card_w + card_gap)
        cbox = FancyBboxPatch(
            (cx, card_area_y), card_w, card_area_h,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor="#0c1424", edgecolor="#1e293b", linewidth=1.2, zorder=2
        )
        ax.add_patch(cbox)
        
        # Header inside card
        badge_w = len(cdata["badge"]) * 5.8 + 10
        p_badge = FancyBboxPatch(
            (cx + 14, card_area_y + card_area_h - 26), badge_w, 14,
            boxstyle="round,pad=0,rounding_size=3",
            facecolor="#050a14", edgecolor=cdata["color"], linewidth=0.8, zorder=3
        )
        ax.add_patch(p_badge)
        ax.text(
            cx + 14 + badge_w/2, card_area_y + card_area_h - 19, cdata["badge"],
            color=cdata["color"], fontsize=6.6, fontweight="bold", va="center", ha="center", zorder=4
        )
        
        # Title
        ax.text(
            cx + 18 + badge_w, card_area_y + card_area_h - 19, cdata["title"],
            color="#ffffff", fontsize=9.8, fontweight="bold", va="center", ha="left", zorder=3
        )
        
        # Accent separator line
        ax.plot([cx + 14, cx + card_w - 14], [card_area_y + card_area_h - 38, card_area_y + card_area_h - 38],
                color=cdata["color"], lw=1.2, alpha=0.5, zorder=3)
        
        # Card bullet items
        item_y = card_area_y + card_area_h - 60
        for item in cdata["items"]:
            ax.text(
                cx + 14, item_y, "▸",
                color=cdata["color"], fontsize=9.2, va="top", ha="left", zorder=3
            )
            words = item.split()
            lines = []
            curr_line = []
            for w in words:
                curr_line.append(w)
                if len(" ".join(curr_line)) > 30:
                    lines.append(" ".join(curr_line[:-1]))
                    curr_line = [w]
            if curr_line:
                lines.append(" ".join(curr_line))
            
            wrapped_text = "\n".join(lines)
            ax.text(
                cx + 26, item_y, wrapped_text,
                color="#cbd5e1", fontsize=8.2, va="top", ha="left", zorder=3, linespacing=1.38
            )
            item_y -= len(lines) * 17 + 22

    # ==========================================
    # LEGEND BAR (BOTTOM)
    # ==========================================
    legend_y = frame_margin_y + 36
    ax.text(
        grid_left + 15, legend_y, "Legend",
        color="#94a3b8", fontsize=9.8, fontweight="bold", va="center", ha="left", zorder=4
    )
    
    legend_items = [
        ("User UI", "#06b6d4"),
        ("Agent logic", "#10b981"),
        ("Policy", "#f43f5e"),
        ("Context / trace", "#8b5cf6"),
        ("External system", "#3b82f6")
    ]
    
    leg_start_x = grid_left + 115
    leg_gap = (grid_w - 115) / len(legend_items)
    for l_idx, (llabel, lcol) in enumerate(legend_items):
        lx = leg_start_x + l_idx * leg_gap
        pill = FancyBboxPatch(
            (lx, legend_y - 8), 18, 16,
            boxstyle="round,pad=0,rounding_size=4",
            facecolor="#071224", edgecolor=lcol, linewidth=1.5, zorder=4
        )
        ax.add_patch(pill)
        ax.text(
            lx + 26, legend_y, llabel,
            color="#cbd5e1", fontsize=8.4, fontweight="medium", va="center", ha="left", zorder=4
        )
        
    # ==========================================
    # SAVE FIGURE
    # ==========================================
    plt.savefig(output_path, dpi=dpi, facecolor=bg_color, edgecolor="none")
    plt.close()
    
    if output_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        im = Image.open(output_path)
        print(f"Generated diagram successfully at '{output_path}'")
        print(f"Dimensions: {im.size[0]} x {im.size[1]} pixels (Requested: {width_px} x {height_px})")
        assert im.size == (width_px, height_px), f"Expected {(width_px, height_px)}, got {im.size}"
    else:
        print(f"Generated vector diagram successfully at '{output_path}'")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RoyaleRL Architecture Diagram")
    parser.add_argument("--width", type=int, default=1080, help="Target image width in pixels (default: 1080)")
    parser.add_argument("--height", type=int, default=1720, help="Target image height in pixels (default: 1720)")
    parser.add_argument("--dpi", type=int, default=100, help="DPI resolution (default: 100)")
    parser.add_argument("--output", type=str, default="royale_rl_architecture_1080x1720.png", help="Output file path")
    args = parser.parse_args()
    
    generate_royale_rl_diagram(
        output_path=args.output,
        width_px=args.width,
        height_px=args.height,
        dpi=args.dpi
    )
