import os
import sys

# Ensure local imports within core and parent resolve cleanly
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CORE_DIR)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from .agent import Agent, DecisionTransformer, Block, ReplayBuffer, PLACEMENT_GRID, NUM_GRID_LOCATIONS, ACTION_DIM
from .tactical_brain import TacticalBrain
from .vision import Vision, EnemyDetector, CardClassifier, ElixirTracker, ElixirVision

__all__ = [
    'Agent',
    'DecisionTransformer',
    'Block',
    'ReplayBuffer',
    'PLACEMENT_GRID',
    'NUM_GRID_LOCATIONS',
    'ACTION_DIM',
    'TacticalBrain',
    'Vision',
    'EnemyDetector',
    'CardClassifier',
    'ElixirTracker',
    'ElixirVision',
]
