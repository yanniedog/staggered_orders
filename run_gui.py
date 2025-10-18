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

def kill_existing_sessions():
    """Kill any existing GUI sessions running on port 8050"""
    try:
        # Try to connect to existing session
        response = requests.get('http://localhost:8050', timeout=2)
        if response.status_code == 200:
            print("Found existing GUI session on port 8050, closing it...")

            # Method 1: Use netstat to find and kill all processes using port 8050
            try:
                import subprocess

                # Get all processes using port 8050
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
                lines = result.stdout.split('\n')
                pids_to_kill = []

                for line in lines:
                    if ':8050' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            try:
                                pids_to_kill.append(int(pid))
                            except ValueError:
                                continue

                # Kill all processes found
                killed_any = False
                for pid in pids_to_kill:
                    print(f"Found process {pid} using port 8050, killing it...")
                    subprocess.run(['taskkill', '/PID', str(pid), '/F'], shell=True)
                    killed_any = True
                    time.sleep(1)  # Give it time to close

                if killed_any:
                    # Wait for cleanup
                    time.sleep(2)

                    # Verify no processes remain using port 8050
                    print("Verifying no processes remain on port 8050...")
                    verify_result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
                    verify_lines = verify_result.stdout.split('\n')
                    remaining_processes = []

                    for line in verify_lines:
                        if ':8050' in line and 'LISTENING' in line:
                            remaining_processes.append(line.strip())

                    if remaining_processes:
                        print("WARNING: Processes still using port 8050:")
                        for proc in remaining_processes:
                            print(f"  {proc}")
                        print("Attempting to kill remaining processes...")

                        # Try to kill any remaining processes
                        for line in remaining_processes:
                            parts = line.split()
                            if len(parts) >= 5:
                                pid = parts[-1]
                                try:
                                    pid_int = int(pid)
                                    print(f"Killing remaining process {pid_int}...")
                                    subprocess.run(['taskkill', '/PID', str(pid_int), '/F'], shell=True)
                                except ValueError:
                                    continue

                        # Final verification
                        time.sleep(2)
                        final_result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True, shell=True)
                        final_lines = final_result.stdout.split('\n')
                        final_remaining = [line.strip() for line in final_lines if ':8050' in line and 'LISTENING' in line]

                        if final_remaining:
                            print("ERROR: Could not kill all processes using port 8050:")
                            for proc in final_remaining:
                                print(f"  {proc}")
                            print("You may need to manually restart your system or close browser tabs.")
                        else:
                            print("[OK] All processes successfully terminated from port 8050")
                    else:
                        print("[OK] No processes remain on port 8050")

                    print("Previous session closed.")
                    return

            except Exception as e:
                print(f"Netstat method failed: {e}")

            # Method 2: Fallback to psutil
            killed_any = False
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Get connections for this process (use net_connections instead of deprecated connections)
                    connections = proc.net_connections()
                    for conn in connections:
                        if conn.laddr.port == 8050:
                            print(f"Killing process {proc.info['pid']} ({proc.info['name']}) using port 8050")
                            proc.kill()
                            killed_any = True
                            time.sleep(1)  # Give it time to close
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, AttributeError):
                    # Process might not have connections or other issues
                    pass

            if killed_any:
                # Wait a moment for cleanup
                time.sleep(2)
                print("Previous session closed.")
            else:
                print("Could not find process using port 8050, but session exists.")
                print("You may need to manually close your browser tab or restart.")

    except requests.exceptions.RequestException:
        # No existing session found, which is fine
        pass
    except Exception as e:
        print(f"Warning: Could not check for existing sessions: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'dash', 'plotly', 'pandas', 'numpy', 'scipy',
        'matplotlib', 'seaborn', 'yaml', 'requests', 'psutil'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages with:")
        print("pip install -r requirements.txt")
        return False

    return True

def main():
    """Main launcher function with centralized logging"""
    # Initialize logging
    with LoggingContext(output_dir="output", symbol="GUI_LAUNCHER") as logger:
        logger.log_analysis_step("Starting GUI launcher", "STARTED")

        print("=" * 60)
        print("    INTERACTIVE STAGGERED ORDER LADDER GUI")
        print("=" * 60)
        print()

        # Log configuration check
        logger.log_analysis_step("Checking dependencies", "STARTED")

        # Check dependencies
        if not check_dependencies():
            logger.log_problem("Missing required dependencies", "CRITICAL")
            logger.log_analysis_step("Dependency check", "FAILED")
            sys.exit(1)

        logger.log_analysis_step("Dependency check", "SUCCESS")

        # Kill any existing sessions
        logger.log_analysis_step("Killing existing sessions", "STARTED")
        kill_existing_sessions()
        logger.log_analysis_step("Killing existing sessions", "SUCCESS")

        # Check if GUI files exist
        gui_files = ['gui_app.py', 'gui_calculator.py', 'gui_visualizations.py', 'gui_historical.py']
        missing_files = [f for f in gui_files if not os.path.exists(f)]

        if missing_files:
            logger.log_problem(f"Missing GUI files: {missing_files}", "CRITICAL")
            print("Missing GUI files:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nPlease ensure all GUI files are present.")
            sys.exit(1)

        logger.log_analysis_step("GUI files check", "SUCCESS")

        # Check if config exists
        if not os.path.exists('config.yaml'):
            logger.log_problem("config.yaml not found - using default configuration", "WARNING")
            print("Warning: config.yaml not found. Using default configuration.")

        # Check if historical data exists
        if not os.path.exists('cache_SOLUSDT_1h_1095d.csv'):
            logger.log_problem("Historical data cache not found - GUI will use mock data", "WARNING")
            print("Warning: Historical data cache not found.")
            print("The GUI will use mock data for demonstration.")
            print("Run 'python main.py' first to generate historical data.")
            print()

        logger.log_analysis_step("Configuration validation", "SUCCESS")

        print("Starting Interactive Ladder GUI...")
        print("The GUI will open in your default web browser.")
        print("If it doesn't open automatically, navigate to: http://localhost:8050")
        print()
        print("Press Ctrl+C to stop the server.")
        print()

        try:
            # Import and run the GUI
            logger.log_analysis_step("Starting GUI server", "STARTED")
            from gui_app import InteractiveLadderGUI
            gui = InteractiveLadderGUI()
            gui.run(debug=False, port=8050)
            logger.log_analysis_step("GUI server started", "SUCCESS")
        except KeyboardInterrupt:
            logger.log_analysis_step("GUI shutdown requested", "SUCCESS")
            print("\nShutting down GUI...")
            sys.exit(0)
        except Exception as e:
            logger.log_error(e, "GUI startup")
            logger.log_analysis_step("GUI startup", "FAILED")
            print(f"Error starting GUI: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
            print("2. Check that all GUI files are present")
            print("3. Try running 'python main.py' first to generate data")
            sys.exit(1)

if __name__ == "__main__":
    main()
