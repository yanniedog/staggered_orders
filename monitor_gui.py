#!/usr/bin/env python3
"""
GUI Monitor - Outputs website content every 5 seconds to command line
"""
import requests
import time
import os
from datetime import datetime

def monitor_gui():
    """Monitor the GUI website and output content every 5 seconds"""
    url = "http://localhost:8050"
    
    print("=" * 80)
    print("GUI MONITOR - Fetching website content every 5 seconds")
    print("=" * 80)
    print(f"Monitoring: {url}")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        while True:
            try:
                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Fetch the website
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    print(f"\n[{timestamp}] SUCCESS - Website is running")
                    print(f"Status Code: {response.status_code}")
                    print(f"Content Length: {len(response.text)} characters")
                    print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
                    
                    # Show first 500 characters of HTML
                    html_preview = response.text[:500]
                    print(f"\nHTML Preview (first 500 chars):")
                    print("-" * 50)
                    print(html_preview)
                    if len(response.text) > 500:
                        print("...")
                    print("-" * 50)
                    
                else:
                    print(f"\n[{timestamp}] ERROR - Status Code: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] CONNECTION ERROR: {e}")
                
            except Exception as e:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] UNEXPECTED ERROR: {e}")
            
            # Wait 5 seconds
            print(f"\nWaiting 5 seconds...")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitor stopped by user")
        print("=" * 80)

if __name__ == "__main__":
    monitor_gui()
