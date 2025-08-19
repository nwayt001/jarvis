"""
JARVIS Google Calendar Integration Module
Handles calendar events, reminders, and scheduling
"""
import os
import pickle
import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Google Calendar API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the token file.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

class JARVISCalendar:
    """Handle Google Calendar integration for JARVIS"""
    
    def __init__(self, credentials_file: str = 'credentials.json'):
        self.credentials_file = credentials_file
        self.token_file = 'token.pickle'
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar API"""
        creds = None
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    logger.error(f"Credentials file {self.credentials_file} not found!")
                    logger.info("Please follow these steps:")
                    logger.info("1. Go to https://console.cloud.google.com/")
                    logger.info("2. Create a new project or select existing")
                    logger.info("3. Enable Google Calendar API")
                    logger.info("4. Create credentials (OAuth 2.0 Client ID)")
                    logger.info("5. Download credentials.json and place in jarvis directory")
                    return
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            self.service = build('calendar', 'v3', credentials=creds)
            logger.info("Successfully connected to Google Calendar")
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
    
    def get_today_events(self) -> List[Dict]:
        """Get all events for today"""
        if not self.service:
            return []
        
        # Get start and end of today in UTC
        now = datetime.datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=today_start,
                timeMax=today_end,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return self._format_events(events)
        
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return []
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict]:
        """Get upcoming events within specified hours"""
        if not self.service:
            return []
        
        now = datetime.datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + datetime.timedelta(hours=hours)).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return self._format_events(events)
        
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return []
    
    def get_next_meeting(self) -> Optional[Dict]:
        """Get the next upcoming meeting"""
        if not self.service:
            return None
        
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                maxResults=1,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            if events:
                formatted = self._format_events(events)
                return formatted[0] if formatted else None
            return None
        
        except HttpError as error:
            logger.error(f'An error occurred: {error}')
            return None
    
    def _format_events(self, events: List) -> List[Dict]:
        """Format Google Calendar events into a cleaner structure"""
        formatted = []
        
        for event in events:
            # Get start time
            start = event.get('start')
            if start:
                start_time = start.get('dateTime', start.get('date'))
            else:
                continue
            
            # Get end time
            end = event.get('end')
            if end:
                end_time = end.get('dateTime', end.get('date'))
            else:
                end_time = None
            
            # Parse times
            try:
                if 'T' in start_time:  # DateTime format
                    start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None
                else:  # Date only (all-day event)
                    start_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d')
                    end_dt = datetime.datetime.strptime(end_time, '%Y-%m-%d') if end_time else None
            except:
                continue
            
            # Calculate time until event
            now = datetime.datetime.now(start_dt.tzinfo)
            time_until = start_dt - now
            
            formatted_event = {
                'summary': event.get('summary', 'No title'),
                'start': start_dt,
                'end': end_dt,
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'attendees': [att.get('email', '') for att in event.get('attendees', [])],
                'video_link': self._extract_video_link(event),
                'time_until': time_until,
                'is_all_day': 'date' in start and 'dateTime' not in start
            }
            
            formatted.append(formatted_event)
        
        return formatted
    
    def _extract_video_link(self, event: Dict) -> Optional[str]:
        """Extract video conference link from event"""
        # Check hangout link
        if 'hangoutLink' in event:
            return event['hangoutLink']
        
        # Check description for common video links
        description = event.get('description', '')
        location = event.get('location', '')
        
        # Common video conference patterns
        patterns = [
            r'(https?://[\w\-\.]*zoom\.us/j/[\w\-]+)',
            r'(https?://meet\.google\.com/[\w\-]+)',
            r'(https?://teams\.microsoft\.com/[\w\-/]+)',
            r'(https?://[\w\-\.]*webex\.com/[\w\-/]+)'
        ]
        
        import re
        for pattern in patterns:
            # Check description
            match = re.search(pattern, description)
            if match:
                return match.group(1)
            # Check location
            match = re.search(pattern, location)
            if match:
                return match.group(1)
        
        return None
    
    def format_event_for_speech(self, event: Dict) -> str:
        """Format an event for JARVIS to speak"""
        parts = []
        
        # Time
        if event['is_all_day']:
            parts.append(f"All-day event: {event['summary']}")
        else:
            time_str = event['start'].strftime('%I:%M %p').lstrip('0')
            parts.append(f"{event['summary']} at {time_str}")
        
        # Duration
        if event['end'] and not event['is_all_day']:
            duration = event['end'] - event['start']
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            if hours > 0:
                parts.append(f"({hours} hour{'s' if hours > 1 else ''}{f' {minutes} minutes' if minutes > 0 else ''})")
            elif minutes > 0:
                parts.append(f"({minutes} minutes)")
        
        # Location or video link
        if event['video_link']:
            parts.append("(Video conference)")
        elif event['location']:
            parts.append(f"at {event['location']}")
        
        # Time until
        time_until = event['time_until']
        if time_until.total_seconds() > 0:
            hours = int(time_until.total_seconds() // 3600)
            minutes = int((time_until.total_seconds() % 3600) // 60)
            
            if hours > 0 and minutes > 0:
                parts.append(f"in {hours} hour{'s' if hours > 1 else ''} and {minutes} minute{'s' if minutes > 1 else ''}")
            elif hours > 0:
                parts.append(f"in {hours} hour{'s' if hours > 1 else ''}")
            elif minutes > 0:
                parts.append(f"in {minutes} minute{'s' if minutes > 1 else ''}")
            else:
                parts.append("starting soon")
        
        return " ".join(parts)


# Standalone functions for easy integration
def get_calendar_summary() -> str:
    """Get a summary of today's calendar events"""
    try:
        cal = JARVISCalendar()
        events = cal.get_today_events()
        
        if not events:
            return "You have no scheduled events today, Sir."
        
        summary = [f"You have {len(events)} event{'s' if len(events) > 1 else ''} today:"]
        
        for event in events:
            summary.append(f"• {cal.format_event_for_speech(event)}")
        
        return "\n".join(summary)
    except Exception as e:
        logger.error(f"Failed to get calendar summary: {e}")
        return "I'm unable to access your calendar at the moment, Sir. Please ensure credentials are configured."


def check_upcoming_meeting(minutes_ahead: int = 15) -> Optional[str]:
    """Check if there's a meeting coming up soon"""
    try:
        cal = JARVISCalendar()
        next_meeting = cal.get_next_meeting()
        
        if next_meeting:
            time_until = next_meeting['time_until']
            if 0 < time_until.total_seconds() <= minutes_ahead * 60:
                return f"Sir, reminder: {cal.format_event_for_speech(next_meeting)}"
        
        return None
    except Exception as e:
        logger.error(f"Failed to check upcoming meeting: {e}")
        return None