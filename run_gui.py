#!/usr/bin/env python3
"""
Interactive Ladder GUI Launcher
Simple launcher script for the GUI application.
"""
import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'dash', 'plotly', 'pandas', 'numpy', 'scipy', 
        'matplotlib', 'seaborn', 'pyyaml', 'requests'
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
    """Main launcher function"""
    print("=" * 60)
    print("    INTERACTIVE STAGGERED ORDER LADDER GUI")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if GUI files exist
    gui_files = ['gui_app.py', 'gui_calculator.py', 'gui_visualizations.py', 'gui_historical.py']
    missing_files = [f for f in gui_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing GUI files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all GUI files are present.")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists('config.yaml'):
        print("Warning: config.yaml not found. Using default configuration.")
    
    # Check if historical data exists
    if not os.path.exists('cache_SOLUSDT_1h_1095d.csv'):
        print("Warning: Historical data cache not found.")
        print("The GUI will use mock data for demonstration.")
        print("Run 'python main.py' first to generate historical data.")
        print()
    
    print("Starting Interactive Ladder GUI...")
    print("The GUI will open in your default web browser.")
    print("If it doesn't open automatically, navigate to: http://localhost:8050")
    print()
    print("Press Ctrl+C to stop the server.")
    print()
    
    try:
        # Import and run the GUI
        from gui_app import InteractiveLadderGUI
        gui = InteractiveLadderGUI()
        gui.run(debug=False, port=8050)
    except KeyboardInterrupt:
        print("\nShutting down GUI...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that all GUI files are present")
        print("3. Try running 'python main.py' first to generate data")
        sys.exit(1)

if __name__ == "__main__":
    main()
