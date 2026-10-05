import os
import sys

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(TRAINING_DIR)
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from .human_recorder import HumanRecorder

__all__ = [
    'HumanRecorder',
]
