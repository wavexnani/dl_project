import os
import sys

DRIVERS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(DRIVERS_DIR)
if DRIVERS_DIR not in sys.path:
    sys.path.insert(0, DRIVERS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from .controller import Controller
from .scaler import Scaler
from .power_manager import keep_awake
from .game_state_manager import GameStateManager

__all__ = [
    'Controller',
    'Scaler',
    'keep_awake',
    'GameStateManager',
]
