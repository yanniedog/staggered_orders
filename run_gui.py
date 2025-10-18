#!/usr/bin/env python3
"""
Interactive Ladder GUI Launcher
Simple launcher script for the GUI application with centralized logging.
"""
import sys
import os
import subprocess
import requests
import time
import signal
import psutil

# Import centralized logging
from logger import create_analysis_logger, LoggingContext

def _kill_process_on_port(port=8050):
    """Kill process using specified port"""
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
        pids = set()
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        pids.add(int(parts[-1]))
                    except ValueError:
                        pass
        
        for pid in pids:
            subprocess.run(['taskkill', '/PID', str(pid), '/F'], shell=True, stderr=subprocess.DEVNULL)
        
        if pids:
            time.sleep(1)
            return True
        return False
    except Exception:
        return False

def kill_existing_sessions():
    """Kill any existing GUI sessions running on port 8050"""
    try:
        requests.get('http://localhost:8050', timeout=2)
        print("Found existing session, closing...")
        if _kill_process_on_port(8050):
            print("Previous session closed.")
    except requests.exceptions.RequestException:
        pass
    except Exception as e:
        print(f"Warning: Could not check for existing sessions: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['dash', 'plotly', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'yaml', 'requests', 'psutil']
    missing = [pkg for pkg in required if not __import__('importlib').util.find_spec(pkg)]
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def main():
    """Main launcher function with centralized logging"""
    with LoggingContext(output_dir="output", symbol="GUI_LAUNCHER") as logger:
        logger.log_analysis_step("Starting GUI launcher", "STARTED")
        print("=" * 60)
        print("    INTERACTIVE STAGGERED ORDER LADDER GUI")
        print("=" * 60 + "\n")

        # Check dependencies
        if not check_dependencies():
            logger.log_problem("Missing required dependencies", "CRITICAL")
            sys.exit(1)
        logger.log_analysis_step("Dependencies OK", "SUCCESS")

        # Kill existing sessions and check files
        kill_existing_sessions()
        
        gui_files = ['gui_app.py', 'gui_calculator.py', 'gui_visualizations.py', 'gui_historical.py']
        if missing := [f for f in gui_files if not os.path.exists(f)]:
            logger.log_problem(f"Missing GUI files: {missing}", "CRITICAL")
            print(f"Missing files: {', '.join(missing)}")
            sys.exit(1)
        
        if not os.path.exists('config.yaml'):
            print("Warning: config.yaml not found, using defaults.")
        if not os.path.exists('cache_SOLUSDT_1h_1095d.csv'):
            print("Warning: No cache found. Run 'python main.py' first for real data.\n")

        print("Starting GUI on http://localhost:8050")
        print("Press Ctrl+C to stop.\n")

        try:
            from gui_app import InteractiveLadderGUI
            gui = InteractiveLadderGUI()
            gui.run(debug=False, port=8050)
            logger.log_analysis_step("GUI started", "SUCCESS")
        except KeyboardInterrupt:
            print("\nShutting down...")
            sys.exit(0)
        except Exception as e:
            logger.log_error(e, "GUI startup")
            print(f"Error: {e}")
            print("Try: pip install -r requirements.txt")
            sys.exit(1)

if __name__ == "__main__":
    main()
