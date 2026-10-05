"""
=================================================================
  Clash Royale AI — Power & Sleep Management
=================================================================
Prevents Windows from:
  1. Turning off the display
  2. Entering sleep or hibernate
  3. Locking the screen due to user idle timeout
Restores normal Windows power settings upon exit.
=================================================================
"""

import ctypes
import contextlib
import sys

# Win32 Execution State Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040

class PowerManager:
    """Controls Windows power and sleep prevention."""
    _active = False

    @classmethod
    def prevent_sleep(cls, keep_display_on: bool = True):
        """Signals Windows to keep system and display awake."""
        if sys.platform != "win32":
            return False

        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if keep_display_on:
            flags |= ES_DISPLAY_REQUIRED

        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            if res != 0:
                cls._active = True
                print("[POWER] Windows sleep & screen lock prevented (Active 24/7 Mode).")
                return True
            else:
                print("[POWER] Warning: SetThreadExecutionState returned 0.")
                return False
        except Exception as e:
            print(f"[POWER] Failed to prevent sleep: {e}")
            return False

    @classmethod
    def restore_sleep(cls):
        """Restores normal Windows power and sleep behavior."""
        if sys.platform != "win32" or not cls._active:
            return

        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            cls._active = False
            print("[POWER] Windows sleep & power settings restored to normal.")
        except Exception as e:
            print(f"[POWER] Error restoring sleep: {e}")

@contextlib.contextmanager
def keep_awake(keep_display_on: bool = True):
    """Context manager for keeping Windows awake during a block of work."""
    PowerManager.prevent_sleep(keep_display_on=keep_display_on)
    try:
        yield
    finally:
        PowerManager.restore_sleep()
