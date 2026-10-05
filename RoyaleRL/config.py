import os

# Project Root & Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DATA_DIR = os.path.join(BASE_DIR, "data")
DRIVERS_DIR = os.path.join(BASE_DIR, "drivers")
CORE_DIR = os.path.join(BASE_DIR, "core")

def resolve_path(filename, search_dirs=None):
    """Finds a file in preferred directory or root fallback."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if search_dirs is None:
        search_dirs = [WEIGHTS_DIR, DATA_DIR, BASE_DIR]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(search_dirs[0], filename)

# --- AI Model Paths ---
MODEL_PATH = resolve_path("hand_classifier_best.pth", [WEIGHTS_DIR, BASE_DIR])
CLASS_NAMES_PATH = resolve_path("class_names.txt", [WEIGHTS_DIR, BASE_DIR])
YOLO_MODEL_PATH = resolve_path("enemy_boundary_detector.pt", [WEIGHTS_DIR, BASE_DIR])
RL_MODEL_PATH = resolve_path("rl_agent.pt", [WEIGHTS_DIR, BASE_DIR])
RL_CHECKPOINT_PATH = resolve_path("rl_agent_checkpoint.pt", [WEIGHTS_DIR, BASE_DIR])
RL_FINAL_PATH = resolve_path("rl_agent_final.pt", [WEIGHTS_DIR, BASE_DIR])

# --- Data & Experience Paths ---
REPLAY_BUFFER_PATH = resolve_path("replay_buffer.pkl", [DATA_DIR, BASE_DIR])
TRAINING_STATS_PATH = resolve_path("training_stats.json", [DATA_DIR, BASE_DIR])
SORTED_DATA_PATH = os.path.join(DATA_DIR, "sorted_data") if os.path.exists(os.path.join(DATA_DIR, "sorted_data")) else os.path.join(BASE_DIR, "sorted_data")

REFERENCE_RESOLUTION = (565, 1007)
# --- Image Processing Constants ---
IMG_SIZE = 128
ARENA_BBOX = (0.04, 0.10, 0.96, 0.78)
# --- Offsets for OCR 
# These are the relative positions of all OCR regions
OCR_OFFSETS = [
    {'name': 'ptl', 'x_offset': 114, 'y_offset': 133, 'width': 39, 'height': 18},
    {'name': 'ptr', 'x_offset': 410, 'y_offset': 132, 'width': 41, 'height': 19},
    {'name': 'pbl', 'x_offset': 114, 'y_offset': 625, 'width': 32, 'height': 16},
    {'name': 'pbr', 'x_offset': 411, 'y_offset': 625, 'width': 33, 'height': 16},
    {'name': 'tk', 'x_offset': 267, 'y_offset': 14, 'width': 43, 'height': 19},
    {'name': 'bk', 'x_offset': 267, 'y_offset': 762, 'width': 45, 'height': 20},
    {'name': 'time', 'x_offset': 482, 'y_offset': 23, 'width': 70, 'height': 27},
]

# --- Card Offsets 
# These are the relative positions of the 4 cards
CARD_OFFSETS_WITH_SIZE = [
    (130, 837, 93, 120),
    (235, 837, 94, 119),
    (342, 838, 93, 118),
    (447, 836, 95, 120),
]

# --- Elixir Offset 
ELIXIR_OFFSET = {
    'x': 132, 'y': 958, 'width': 114, 'height': 45
}

KING_TOWER_MAX_HEALTH = 4008
PRINCESS_TOWER_MAX_HEALTH = 2534
CARD_COSTS = {
    'archers': 3, 'archer_queen': 5, 'arrows': 3, 'baby_dragon': 4, 'balloon': 5, 'bandit': 3,
    'barbarian': 5, 'barbarian_barrel': 2, 'barbarian_hut': 7, 'bat': 2, 'battle_healer': 4,
    'battle_ram': 4, 'bomb_tower': 4, 'bomber': 2, 'bowler': 5, 'cannon': 3, 'cannon_cart': 5,
    'clone': 3, 'dark_prince': 4, 'dart_goblin': 3, 'earthquake': 3, 'electro_dragon': 5,
    'electro_giant': 7, 'electro_spirit': 1, 'electro_wizard': 4, 'elite_barbarian': 6,
    'elixir_collector': 6, 'elixir_golem': 3, 'executioner': 5, 'fire_spirit': 1, 'fireball': 4,
    'firecracker': 3, 'fisherman': 3, 'flying_machine': 4, 'freeze': 4, 'furnace': 4, 'giant': 5,
    'giant_skeleton': 6, 'giant_snowball': 2, 'goblin': 2, 'goblin_barrel': 3, 'goblin_cage': 4,
    'goblin_drill': 4, 'goblin_gang': 3, 'goblin_hut': 5, 'golden_knight': 4, 'golem': 8,
    'graveyard': 5, 'guard': 3, 'heal_spirit': 1, 'hog_rider': 4, 'hunter': 4, 'ice_golem': 2,
    'ice_spirit': 1, 'ice_wizard': 3, 'inferno_dragon': 4, 'inferno_tower': 5, 'knight': 3,
    'lava_hound': 7, 'little_prince': 3, 'lumberjack': 4, 'magic_archer': 4, 'mega_knight': 7,
    'mega_minion': 3, 'mighty_miner': 4, 'miner': 3, 'minions': 3, 'minion_horde': 5,
    'mini-pekka': 4, 'mirror': 1, 'monk': 5, 'mortar': 4, 'mother_witch': 4, 'musketeer': 4,
    'night_witch': 4, 'pekka': 7, 'phoenix': 4, 'poison': 4, 'prince': 5, 'princess': 3,
    'rage': 2, 'ram_rider': 5, 'rascal': 5, 'rocket': 6, 'royal_delivery': 3, 'royal_ghost': 3,
    'royal_giant': 6, 'royal_hog': 5, 'royal_recruit': 7, 'skeleton': 1, 'skeleton_army': 3,
    'skeleton_barrel': 3, 'skeleton_king': 4, 'sparky': 6, 'spear_goblin': 2, 'tesla': 4,
    'the_log': 2, 'tombstone': 3, 'tornado': 3, 'valkyrie': 4, 'wall_breaker': 2, 'witch': 5,
    'wizard': 5, 'x_bow': 6, 'zap': 2, 'zappy': 4
}
ALL_CARDS = list(CARD_COSTS.keys())
CARD_TO_INDEX = {name: i for i, name in enumerate(ALL_CARDS)}
NUM_CARD_TYPES = len(ALL_CARDS)
# INC-1 FIX: Define STATE_DIM once here so all modules import it consistently.
STATE_DIM = 1 + 6 + (4 * NUM_CARD_TYPES) + (20 * 4)

# --- Shared Utility Functions ---
def get_health_percentage(ocr_value, tower_type):
    max_health = KING_TOWER_MAX_HEALTH if tower_type == 'king' else PRINCESS_TOWER_MAX_HEALTH
    # BUG-CF1 FIX: Returning 0.0 for a princess tower on OCR failure falsely
    # marks it as destroyed, triggering premature pocket/breach assaults.
    # Safe default = 1.0 (assume the tower is alive) for all types.
    if ocr_value is None or str(ocr_value).strip() == '':
        return 1.0
    try:
        health = min(int(ocr_value), max_health)
        return health / max_health
    except (ValueError, TypeError):
        return 1.0


# --- Debugging ---
# Set to True to show a live window with the OCR and card boxes drawn.
# BUG-CF2 NOTE: This flag is now wired into vision.py's get_game_state().
DEBUG_VISUALS = False
