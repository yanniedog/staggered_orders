#!/usr/bin/env python3
"""
Simple GUI test script
"""
import sys
import traceback

try:
    print("Testing GUI imports...")
    from gui_app import InteractiveLadderGUI
    print("✅ GUI imports successful")
    
    print("Initializing GUI...")
    gui = InteractiveLadderGUI()
    print("✅ GUI initialized successfully")
    
    print("Starting GUI server...")
    print("The GUI should open at: http://localhost:8050")
    print("Press Ctrl+C to stop")
    
    gui.run(debug=False, port=8050)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
