import os
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PKG_DIR)
for p in [PKG_DIR, PARENT_DIR, os.path.join(PARENT_DIR, 'core'), os.path.join(PARENT_DIR, 'drivers')]:
    if p not in sys.path:
        sys.path.insert(0, p)
