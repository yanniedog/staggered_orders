#!/usr/bin/env python3
"""
Simple GUI Monitor - Shows website status every 5 seconds
"""
import requests
import time
from datetime import datetime

def simple_monitor():
    """Simple monitor that shows website status every 5 seconds"""
    url = "http://localhost:8050"
    
    print("GUI Status Monitor - Checking every 5 seconds")
    print("=" * 50)
    print(f"URL: {url}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"[{timestamp}] [OK] GUI RUNNING - Status: {response.status_code} - Size: {len(response.text)} chars")
                else:
                    print(f"[{timestamp}] [ERROR] Status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"[{timestamp}] [FAIL] CONNECTION FAILED - {str(e)[:50]}...")
                
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Monitor stopped")

if __name__ == "__main__":
    simple_monitor()
