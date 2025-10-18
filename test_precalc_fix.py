#!/usr/bin/env python
"""Test that precalc status works without errors"""

from gui_app import InteractiveLadderGUI

def test_precalc_status():
    """Test that precalc status works correctly"""
    try:
        gui = InteractiveLadderGUI()
        status = gui.get_precalc_status()
        print(f"Precalc status: {status}")
        print(f"Precalc running: {gui.precalc_running}")
        print(f"Cache size: {len(gui.precalc_cache)}")
        print("SUCCESS: Precalc status works")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_precalc_status()
