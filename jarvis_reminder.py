"""
JARVIS Meeting Reminder System
Runs in background and alerts about upcoming meetings
"""
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Set
from jarvis_calendar import JARVISCalendar
from jarvis_tts_fixed import JARVISStreamingTTS

logger = logging.getLogger(__name__)

class JARVISReminder:
    """Background service for meeting reminders"""
    
    def __init__(self, tts_host: str, check_interval: int = 60):
        """
        Initialize reminder service
        
        Args:
            tts_host: URL of the TTS server
            check_interval: How often to check for meetings (seconds)
        """
        self.calendar = None
        self.tts = JARVISStreamingTTS(tts_host)
        self.check_interval = check_interval
        self.running = False
        self.thread = None
        self.reminded_events: Set[str] = set()  # Track what we've already reminded about
        
        # Reminder thresholds (in minutes before meeting)
        self.reminder_times = [15, 5, 1]  # Remind at 15, 5, and 1 minute before
        
        try:
            self.calendar = JARVISCalendar()
            logger.info("Calendar connected for reminders")
        except Exception as e:
            logger.error(f"Could not connect calendar for reminders: {e}")
    
    def start(self):
        """Start the reminder service"""
        if self.running:
            logger.warning("Reminder service already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._reminder_loop, daemon=True)
        self.thread.start()
        logger.info("JARVIS reminder service started")
    
    def stop(self):
        """Stop the reminder service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("JARVIS reminder service stopped")
    
    def _reminder_loop(self):
        """Main loop that checks for upcoming meetings"""
        while self.running:
            try:
                self._check_meetings()
            except Exception as e:
                logger.error(f"Error in reminder loop: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)
    
    def _check_meetings(self):
        """Check for meetings that need reminders"""
        if not self.calendar:
            return
        
        try:
            # Get events in the next 2 hours
            events = self.calendar.get_upcoming_events(hours=2)
            
            for event in events:
                self._process_event_reminder(event)
        
        except Exception as e:
            logger.error(f"Error checking meetings: {e}")
    
    def _process_event_reminder(self, event: dict):
        """Process reminders for a single event"""
        # Create unique event ID
        event_id = f"{event['summary']}_{event['start'].isoformat()}"
        
        # Get time until event
        time_until = event['time_until']
        minutes_until = time_until.total_seconds() / 60
        
        # Check each reminder threshold
        for reminder_minutes in self.reminder_times:
            # Create unique reminder ID
            reminder_id = f"{event_id}_{reminder_minutes}"
            
            # Check if we should send this reminder
            if (reminder_id not in self.reminded_events and 
                minutes_until <= reminder_minutes and 
                minutes_until > reminder_minutes - 1):  # Within 1-minute window
                
                self._send_reminder(event, reminder_minutes)
                self.reminded_events.add(reminder_id)
                
                # Clean up old reminders (older than 1 day)
                self._cleanup_old_reminders()
    
    def _send_reminder(self, event: dict, minutes: int):
        """Send a reminder for an event"""
        # Format the reminder message
        if minutes == 15:
            urgency = "Sir, you have"
        elif minutes == 5:
            urgency = "Sir, reminder: you have"
        else:  # 1 minute
            urgency = "Sir, urgent: your"
        
        # Build the message
        message_parts = [urgency]
        
        if event['is_all_day']:
            message_parts.append(f"an all-day event: {event['summary']}")
        else:
            time_str = event['start'].strftime('%I:%M %p').lstrip('0')
            if minutes == 1:
                message_parts.append(f"{event['summary']} is starting in one minute at {time_str}")
            else:
                message_parts.append(f"{event['summary']} in {minutes} minutes at {time_str}")
        
        # Add location or video link info
        if event['video_link']:
            message_parts.append("This is a video conference meeting")
        elif event['location']:
            message_parts.append(f"at {event['location']}")
        
        # Add attendee count if relevant
        if len(event['attendees']) > 1:
            message_parts.append(f"with {len(event['attendees'])} attendees")
        
        message = ". ".join(message_parts) + "."
        
        # Log and speak the reminder
        logger.info(f"Reminder: {message}")
        print(f"\n🔔 REMINDER: {message}\n")
        
        # Use TTS to speak the reminder
        if self.tts:
            self.tts.speak(message)
    
    def _cleanup_old_reminders(self):
        """Remove old reminder IDs to prevent memory growth"""
        # This is simplified - in production you'd want timestamp-based cleanup
        if len(self.reminded_events) > 100:
            # Keep only the 50 most recent
            self.reminded_events = set(list(self.reminded_events)[-50:])
    
    def get_status(self) -> str:
        """Get the current status of the reminder service"""
        if not self.running:
            return "Reminder service is not running"
        
        status = ["Reminder service is active"]
        status.append(f"Checking every {self.check_interval} seconds")
        status.append(f"Tracked {len(self.reminded_events)} reminders today")
        
        if self.calendar:
            try:
                next_meeting = self.calendar.get_next_meeting()
                if next_meeting:
                    status.append(f"Next meeting: {next_meeting['summary']} at {next_meeting['start'].strftime('%I:%M %p')}")
            except:
                pass
        
        return "\n".join(status)


# Integration function for JARVIS main script
def start_reminder_service(tts_host: str) -> JARVISReminder:
    """Start the reminder service and return the instance"""
    reminder = JARVISReminder(tts_host)
    reminder.start()
    return reminder