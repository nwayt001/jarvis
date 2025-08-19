#!/usr/bin/env python3
"""
Setup script for JARVIS Google Calendar integration
This will guide you through setting up Google Calendar API access
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50 + "\n")

def main():
    print_header("JARVIS Calendar Setup")
    
    print("This script will help you set up Google Calendar integration for JARVIS.\n")
    print("You'll need:")
    print("  1. A Google account")
    print("  2. Access to Google Cloud Console")
    print("  3. About 5-10 minutes\n")
    
    input("Press Enter to continue...")
    
    # Step 1: Install dependencies
    print_header("Step 1: Installing Dependencies")
    print("Installing Google Calendar API libraries...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "google-auth", "google-auth-oauthlib", 
            "google-auth-httplib2", "google-api-python-client"
        ], check=True)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please run:")
        print("   pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return 1
    
    # Step 2: Guide through Google Cloud Console
    print_header("Step 2: Google Cloud Console Setup")
    
    print("I'll open the Google Cloud Console for you.")
    print("\nFollow these steps:")
    print("1. Click 'Select a project' → 'New Project'")
    print("2. Name it 'JARVIS-Calendar' and create it")
    print("3. Once created, make sure it's selected\n")
    
    open_browser = input("Open Google Cloud Console? (y/n): ").lower()
    if open_browser == 'y':
        webbrowser.open("https://console.cloud.google.com/")
    
    input("\nPress Enter when you've created and selected your project...")
    
    # Step 3: Enable Calendar API
    print_header("Step 3: Enable Google Calendar API")
    
    print("Now we need to enable the Calendar API:")
    print("1. Click the menu (☰) → 'APIs & Services' → 'Library'")
    print("2. Search for 'Google Calendar API'")
    print("3. Click on it and press 'ENABLE'\n")
    
    open_browser = input("Open the API Library? (y/n): ").lower()
    if open_browser == 'y':
        webbrowser.open("https://console.cloud.google.com/apis/library")
    
    input("\nPress Enter when you've enabled the Calendar API...")
    
    # Step 4: Create credentials
    print_header("Step 4: Create OAuth 2.0 Credentials")
    
    print("Now let's create credentials:")
    print("1. Go to 'APIs & Services' → 'Credentials'")
    print("2. Click '+ CREATE CREDENTIALS' → 'OAuth client ID'")
    print("3. If prompted, configure OAuth consent screen:")
    print("   - Choose 'External' user type")
    print("   - Fill in App name: 'JARVIS'")
    print("   - Add your email as support email")
    print("   - Add your email to test users")
    print("4. Back in credentials, create OAuth client ID:")
    print("   - Application type: 'Desktop app'")
    print("   - Name: 'JARVIS Desktop Client'")
    print("5. Click 'CREATE'\n")
    
    open_browser = input("Open the Credentials page? (y/n): ").lower()
    if open_browser == 'y':
        webbrowser.open("https://console.cloud.google.com/apis/credentials")
    
    input("\nPress Enter when you've created the OAuth client...")
    
    # Step 5: Download credentials
    print_header("Step 5: Download Credentials")
    
    print("After creating the OAuth client:")
    print("1. Click the download button (⬇) next to your client ID")
    print("2. Save the file as 'credentials.json'")
    print(f"3. Move it to: {os.path.abspath('.')}")
    print("\nThe file should be in the same directory as this script.\n")
    
    # Check if credentials file exists
    creds_path = Path("credentials.json")
    
    while not creds_path.exists():
        print("❌ credentials.json not found in current directory")
        retry = input("\nHave you placed credentials.json in this directory? (y/n): ").lower()
        if retry != 'y':
            print("\nPlease download and place credentials.json in:")
            print(f"  {os.path.abspath('.')}")
            print("\nThen run this script again.")
            return 1
    
    print("✅ Found credentials.json!")
    
    # Step 6: Test the connection
    print_header("Step 6: Test Calendar Connection")
    
    print("Let's test the calendar connection...")
    print("This will open a browser window for you to authorize JARVIS.\n")
    
    try:
        from jarvis_calendar import JARVISCalendar
        
        print("Initializing calendar connection...")
        cal = JARVISCalendar()
        
        print("\nTrying to fetch today's events...")
        events = cal.get_today_events()
        
        if events:
            print(f"✅ Success! Found {len(events)} event(s) today:")
            for event in events[:3]:  # Show first 3 events
                print(f"  - {event['summary']} at {event['start'].strftime('%I:%M %p')}")
        else:
            print("✅ Connection successful! (No events found for today)")
        
    except Exception as e:
        print(f"❌ Error connecting to calendar: {e}")
        print("\nPlease check your setup and try again.")
        return 1
    
    # Success!
    print_header("Setup Complete!")
    
    print("🎉 JARVIS Calendar integration is ready!")
    print("\nYou can now use these commands:")
    print("  - 'JARVIS, what's on my calendar today?'")
    print("  - 'JARVIS, when is my next meeting?'")
    print("  - 'JARVIS, give me my morning briefing'")
    print("\nNote: The token is saved, so you won't need to authorize again.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())