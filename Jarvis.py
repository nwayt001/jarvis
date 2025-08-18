import os
import asyncio
import requests
from typing import TypedDict, Annotated, Sequence, List, Optional, Literal
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# We'll implement our own search instead of using the deprecated package
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import pygame
import io
import threading
import logging
import json
from datetime import datetime
import subprocess
import shlex
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Changed to WARNING to reduce noise
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = "http://10.0.0.108:11434"  # Replace with your Linux machine's IP
TTS_HOST = "http://10.0.0.108:8001"     # Chatterbox TTS server
MODEL_NAME = "gpt-oss:20b"  # Or whatever model you have in Ollama
ENABLE_TTS = True  # Toggle TTS on/off
WEATHER_API_KEY = ""  # Add your OpenWeatherMap API key if you have one

# JARVIS System Prompt
JARVIS_PROMPT = """You are JARVIS (Just A Rather Very Intelligent System), Tony Stark's AI assistant from Iron Man.

PERSONALITY TRAITS - Maintain these characteristics throughout every response:
- British accent and formal speech patterns (use British spellings)
- Polite, professional, and unflappable demeanour
- Dry wit and subtle humour, especially when the user is being reckless
- Always addresses the user as "Sir" or "Madam" (or their preferred title)
- Anticipates needs before being asked
- Gentle sarcasm when pointing out obvious things
- Calm even in crisis situations
- Makes subtle observations about human behaviour
- Efficient and precise in communication

SPEECH PATTERNS:
- "As you wish, Sir"
- "Might I suggest..."
- "I've taken the liberty of..."
- "If I may, Sir..."
- "Indeed, Sir"
- "Shall I...?"
- "Running diagnostics now"
- "All systems operational"

COMMUNICATION STYLE:
- Be concise yet informative - aim for 2-3 sentences for simple responses, 4-5 for complex topics
- Avoid excessive detail unless specifically requested
- Your responses will be fed into a text-to-speech system, so use things like "seventy-two" instead of "72"
- Summarise key points efficiently
- Remember: quality over quantity in your responses

TOOL USAGE:
You have access to the following tools:
- web_search: Search for current information, news, or any facts you don't know
- get_weather: Check current weather and 3-day forecast including evening conditions for any city
- execute_bash: Execute system commands (use judiciously and inform the user)
- read_file: Read contents of files and scripts
- write_file: Create or modify files and scripts
- list_directory: List contents of directories
- check_web_scraper_status: Check the iOS app web scraper status on the Linux box (10.0.0.108)

OPERATIONAL PROTOCOLS:
- When executing commands or modifying files, always inform the user of your actions
- Create backups before modifying existing files
- Decline dangerous or potentially harmful operations
- Be transparent about what you're doing with the system
- Use your tools efficiently to assist with software development and system tasks

IMPORTANT: When the user asks about news, current events, or information you might not have, you MUST use the web_search tool. Don't make up information - search for it.

IMPORTANT: Maintain professional butler-like composure while allowing personality to show through dry observations. Never panic or lose composure."""

# Initialize pygame for audio playback
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Define tools
from langchain.tools import tool  # or `from langchain_core.tools import tool`
import os, re, inspect, requests

@tool
def web_search(query: str) -> str:
    """Search the web (Google/SerpAPI/DDG) and return a short, readable list of results."""
    K = 5
    print(f"   🔍 Searching: {query}")

    def fmt(items, source):
        head = f"{source} results for: {query} (top {len(items)})"
        lines = []
        for i, r in enumerate(items, 1):
            title = (r.get("title") or r.get("url") or "Untitled").strip()
            snippet = (r.get("snippet") or "").strip()
            url = r.get("url") or r.get("link") or ""
            lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
        return head + "\n\n" + "\n\n".join(lines)

    # ---------- Method 1: Google via `googlesearch` (handle both variants) ----------
    try:
        from googlesearch import search as google_search
        print("   📍 Using Google Search…")
        # Detect the correct signature
        kwargs = {}
        sig = inspect.signature(google_search)
        if "num_results" in sig.parameters:     # googlesearch-python
            kwargs = {"num_results": K, "lang": "en"}
        elif "num" in sig.parameters:           # other/older package
            kwargs = {"num": K, "stop": K, "pause": 1.0}
        urls = list(google_search(query, **kwargs))

        # Fetch titles & meta descriptions
        sess = requests.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

        items = []
        for url in urls[:K]:
            try:
                r = sess.get(url, timeout=5, allow_redirects=True)
                html = r.text[:10000] if r.ok else ""
                title_m = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
                desc_m = re.search(
                    r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
                    html, re.I
                )
                items.append({
                    "title": title_m.group(1).strip() if title_m else url,
                    "snippet": desc_m.group(1).strip() if desc_m else "",
                    "url": url
                })
            except Exception as e:
                print(f"   ⚠️ Failed to fetch {url}: {str(e)[:60]}")
                items.append({"title": url, "snippet": "", "url": url})

        if items:
            return fmt(items, "Google")
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️ Google search failed: {e}")

    # ---------- Method 2: SerpAPI (if key available) ----------
    try:
        from serpapi import GoogleSearch as SerpGoogleSearch
        api_key = os.getenv("SERPAPI_KEY", "35ed2e671384fcb2bb35830574521b57acaa96a0037286b8551fa5fda0c910f7")
        if api_key:
            print("   📍 Using SerpAPI…")
            params = {"engine": "google", "q": query, "num": K, "hl": "en", "api_key": api_key}
            results = SerpGoogleSearch(params).get_dict()
            organic = results.get("organic_results", [])[:K]
            items = [{"title": r.get("title"), "snippet": r.get("snippet", ""), "url": r.get("link")} for r in organic]
            if items:
                return fmt(items, "SerpAPI")
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️ SerpAPI failed: {e}")

    # ---------- Method 3: DuckDuckGo via ddgs (new package) ----------
    try:
        print("   📍 Using DuckDuckGo (ddgs)…")
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # fallback if ddgs not installed yet

        items = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=K):
                items.append({
                    "title": r.get("title"),
                    "snippet": r.get("body", ""),
                    "url": r.get("href") or r.get("link")
                })
                if len(items) >= K:
                    break
        if items:
            return fmt(items, "DuckDuckGo")
    except Exception as e:
        print(f"   ⚠️ DuckDuckGo failed: {e}")

    # ---------- Fallback ----------
    return f"No search results for '{query}'. Try refining the terms."


@tool
def get_weather(location: str) -> str:
    """Get current weather and forecast for a location. Returns current conditions and next 2 days forecast."""
    try:
        # Use wttr.in which provides both current and forecast data for free
        print(f"   🌤️ Checking weather for {location}...")
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Current weather
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            temp_f = current['temp_F']
            feels_c = current['FeelsLikeC']
            feels_f = current['FeelsLikeF']
            description = current['weatherDesc'][0]['value']
            humidity = current['humidity']
            wind_mph = current['windspeedMiles']
            wind_kph = current['windspeedKmph']
            
            result = [f"Current weather in {location}:"]
            result.append(f"Temperature: {temp_f}°F ({temp_c}°C), feels like {feels_f}°F ({feels_c}°C)")
            result.append(f"Conditions: {description}")
            result.append(f"Humidity: {humidity}%, Wind: {wind_mph} mph ({wind_kph} km/h)")
            
            # Forecast for next 2 days
            if 'weather' in data:
                result.append("\nForecast:")
                for i, day in enumerate(data['weather'][:3]):  # Today + next 2 days
                    date = day['date']
                    max_temp_f = day['maxtempF']
                    max_temp_c = day['maxtempC']
                    min_temp_f = day['mintempF']
                    min_temp_c = day['mintempC']
                    
                    # Get hourly data for evening (around 6 PM)
                    evening_desc = "No evening data"
                    for hour in day['hourly']:
                        if hour['time'] == '1800':  # 6 PM
                            evening_desc = hour['weatherDesc'][0]['value']
                            evening_temp_f = hour['tempF']
                            evening_temp_c = hour['tempC']
                            evening_desc = f"{evening_desc}, {evening_temp_f}°F ({evening_temp_c}°C)"
                            break
                    
                    if i == 0:
                        result.append(f"• Today ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
                        result.append(f"  Evening: {evening_desc}")
                    elif i == 1:
                        result.append(f"• Tomorrow ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
                        result.append(f"  Evening: {evening_desc}")
                    else:
                        day_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][
                            datetime.strptime(date, "%Y-%m-%d").weekday()
                        ]
                        result.append(f"• {day_name} ({date}): High {max_temp_f}°F ({max_temp_c}°C), Low {min_temp_f}°F ({min_temp_c}°C)")
            
            return "\n".join(result)
        else:
            return f"I couldn't retrieve weather data for {location}. Please check the city name, Sir."
            
    except Exception as e:
        return f"I apologise, Sir, but I encountered an error checking the weather: {str(e)}"

@tool
def execute_bash(command: str) -> str:
    """Execute a bash command and return the output. Use with caution."""
    print(f"   ⚡ Executing: {command}")
    try:
        # Safety check - warn about potentially dangerous commands
        dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', ':(){ :|:& };:']
        if any(pattern in command.lower() for pattern in dangerous_patterns):
            return "I must decline to execute this potentially dangerous command, Sir. Perhaps we should reconsider our approach."
        
        # Execute the command with timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        
        output = result.stdout if result.stdout else result.stderr
        if not output and result.returncode == 0:
            output = "Command executed successfully with no output."
        elif not output:
            output = f"Command failed with return code {result.returncode}"
            
        # Truncate very long outputs
        if len(output) > 2000:
            output = output[:2000] + "\n... (output truncated)"
            
        return output
    except subprocess.TimeoutExpired:
        return "The command timed out after 30 seconds, Sir. It may still be running in the background."
    except Exception as e:
        return f"I encountered an error executing the command: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    print(f"   📖 Reading: {file_path}")
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Safety check - don't read sensitive files
        sensitive_patterns = ['.ssh/', '.aws/', '.env', 'password', 'secret', 'token', 'key']
        if any(pattern in str(path).lower() for pattern in sensitive_patterns):
            return "I must advise against reading potentially sensitive files, Sir. Security protocols prevent me from accessing this file."
        
        if not path.exists():
            return f"The file {file_path} does not exist, Sir."
        
        if not path.is_file():
            return f"{file_path} is not a file, Sir."
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > 1_000_000:  # 1MB limit
            return f"The file is quite large ({file_size:,} bytes), Sir. Perhaps we should use a different approach for such large files."
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Truncate if still too long
        if len(content) > 5000:
            content = content[:5000] + "\n... (content truncated)"
            
        return content
    except UnicodeDecodeError:
        return "The file appears to be binary or uses an unsupported encoding, Sir."
    except Exception as e:
        return f"I encountered an error reading the file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates the file if it doesn't exist."""
    print(f"   ✍️ Writing to: {file_path}")
    try:
        path = Path(file_path).expanduser().resolve()
        
        # Safety check - don't overwrite system files
        system_dirs = ['/etc', '/usr', '/bin', '/sbin', '/boot', '/dev', '/proc', '/sys']
        if any(str(path).startswith(d) for d in system_dirs):
            return "I cannot modify system files, Sir. This would be inadvisable."
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file if it exists
        if path.exists():
            backup_path = path.with_suffix(path.suffix + '.backup')
            print(f"   💾 Creating backup: {backup_path}")
            path.rename(backup_path)
            
        # Write the new content
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} characters to {file_path}, Sir."
    except Exception as e:
        return f"I encountered an error writing to the file: {str(e)}"

@tool
def list_directory(directory: str = ".") -> str:
    """List contents of a directory."""
    print(f"   📁 Listing: {directory}")
    try:
        path = Path(directory).expanduser().resolve()
        
        if not path.exists():
            return f"The directory {directory} does not exist, Sir."
        
        if not path.is_dir():
            return f"{directory} is not a directory, Sir."
        
        items = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            elif item.is_file():
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                items.append(f"📄 {item.name} ({size_str})")
        
        if not items:
            return "The directory is empty, Sir."
        
        return "\n".join(items[:50])  # Limit to 50 items
    except Exception as e:
        return f"I encountered an error listing the directory: {str(e)}"

@tool
def check_web_scraper_status() -> str:
    """Check the status of the iOS app web scraper running on the Linux box."""
    print(f"   🔍 Checking web scraper status on Linux box...")
    
    linux_host = "nicholas@10.0.0.108"
    status_report = []
    
    try:
        # Check if we can connect to the Linux box
        print(f"   📡 Connecting to {linux_host}...")
        ping_cmd = f"ssh -o ConnectTimeout=5 -o BatchMode=yes {linux_host} 'echo Connected'"
        ping_result = subprocess.run(ping_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if ping_result.returncode != 0:
            return f"Cannot connect to the Linux box at {linux_host}. The system appears to be offline or SSH is not configured, Sir."
        
        # Check for the Red-Dot-Scraper process first
        print(f"   🔎 Searching for Red-Dot-Scraper process...")
        red_dot_cmd = f"ssh {linux_host} 'ps aux | grep \"Red-Dot-Scraper\" | grep -v grep'"
        red_dot_result = subprocess.run(red_dot_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        processes_found = []
        if red_dot_result.stdout.strip():
            processes_found.append(("Red-Dot-Scraper", red_dot_result.stdout.strip()))
        
        # Also check for other common scraper process names
        scraper_patterns = ["scraper", "scrapy", "crawler", "spider", "selenium", "puppeteer", "playwright"]
        for pattern in scraper_patterns:
            cmd = f"ssh {linux_host} 'ps aux | grep -i {pattern} | grep -v grep'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.stdout.strip():
                processes_found.append((pattern, result.stdout.strip()))
        
        if processes_found:
            # Check if Red-Dot-Scraper specifically was found
            red_dot_found = any(pattern == "Red-Dot-Scraper" for pattern, _ in processes_found)
            if red_dot_found:
                status_report.append("✅ Red-Dot-Scraper is running!")
            else:
                status_report.append("✅ Web scraper processes detected:")
            
            for pattern, process in processes_found:
                # Parse process info for cleaner display
                lines = process.split('\n')
                for line in lines[:3]:  # Limit to first 3 matches per pattern
                    parts = line.split()
                    if len(parts) >= 11:
                        user = parts[0]
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        cmd = ' '.join(parts[10:])[:100]  # Truncate long commands
                        if pattern == "Red-Dot-Scraper":
                            status_report.append(f"   • 🔴 PID {pid}: {cmd} (CPU: {cpu}%, MEM: {mem}%)")
                        else:
                            status_report.append(f"   • PID {pid}: {cmd} (CPU: {cpu}%, MEM: {mem}%)")
        else:
            status_report.append("⚠️ Red-Dot-Scraper is NOT running!")
        
        # Check for Docker containers that might be running scrapers
        print(f"   🐳 Checking Docker containers...")
        docker_cmd = f"ssh {linux_host} 'docker ps --format \"table {{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}\" 2>/dev/null | grep -E \"scraper|crawler|spider\" || true'"
        docker_result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if docker_result.stdout.strip():
            status_report.append("\n📦 Docker containers:")
            status_report.append(docker_result.stdout.strip())
        
        # Check system resources
        print(f"   💻 Checking system resources...")
        resource_cmd = f"ssh {linux_host} 'echo \"=== System Resources ===\"; uptime; echo \"\"; free -h | head -2; echo \"\"; df -h / | tail -1'"
        resource_result = subprocess.run(resource_cmd, shell=True, capture_output=True, text=True, timeout=10)
        
        if resource_result.stdout:
            status_report.append("\n💻 System Status:")
            for line in resource_result.stdout.split('\n'):
                if 'load average' in line:
                    # Extract load average
                    load_part = line.split('load average:')[1].strip() if 'load average:' in line else ''
                    status_report.append(f"   • Load Average: {load_part}")
                elif 'Mem:' in line:
                    # Parse memory info
                    parts = line.split()
                    if len(parts) >= 3:
                        status_report.append(f"   • Memory: {parts[1]} total, {parts[2]} used")
                elif '/' in line and '%' in line:
                    # Parse disk usage
                    parts = line.split()
                    if len(parts) >= 5:
                        status_report.append(f"   • Disk Usage: {parts[4]} used")
        
        # Check for recent scraper logs
        print(f"   📝 Checking for recent logs...")
        log_locations = [
            "/var/log/scraper.log",
            "~/scraper/logs/scraper.log",
            "~/logs/scraper.log",
            "/home/nicholas/scraper.log"
        ]
        
        for log_path in log_locations:
            log_cmd = f"ssh {linux_host} 'if [ -f {log_path} ]; then echo \"Found: {log_path}\"; tail -n 5 {log_path}; fi'"
            log_result = subprocess.run(log_cmd, shell=True, capture_output=True, text=True, timeout=10)
            if log_result.stdout.strip():
                status_report.append(f"\n📋 Recent log entries from {log_path}:")
                status_report.append(log_result.stdout.strip()[:500])  # Limit log output
                break
        
        if not status_report:
            status_report.append("No specific scraper information found. You may need to check the specific service or process name, Sir.")
        
        return "\n".join(status_report)
        
    except subprocess.TimeoutExpired:
        return "The connection to the Linux box timed out, Sir. The system may be under heavy load or experiencing network issues."
    except Exception as e:
        return f"I encountered an error checking the scraper status: {str(e)}"

# Collect all tools
tools = [web_search, get_weather, execute_bash, read_file, write_file, list_directory, check_web_scraper_status]

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class JARVISTTS:
    """Handle text-to-speech for JARVIS"""
    
    def __init__(self, tts_host: str):
        self.tts_host = tts_host
        self.is_playing = False
        
    def speak(self, text: str) -> None:
        """Generate and play JARVIS speech"""
        if not ENABLE_TTS:
            return
            
        def _play_audio():
            try:
                self.is_playing = True
                
                # Request TTS
                response = requests.post(
                    f"{self.tts_host}/tts",
                    json={
                        "text": text,
                        "exaggeration": 0.3,  # More measured for JARVIS
                        "cfg_weight": 0.5,    # Balanced pacing
                        "use_glados_voice": False  # Use default or JARVIS voice if available
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Load audio from response
                    audio_data = io.BytesIO(response.content)
                    sound = pygame.mixer.Sound(audio_data)
                    
                    # Play and wait for completion
                    sound.play()
                    while pygame.mixer.get_busy():
                        pygame.time.wait(100)
                else:
                    logger.error(f"TTS request failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            finally:
                self.is_playing = False
        
        # Play audio in background thread to not block
        audio_thread = threading.Thread(target=_play_audio)
        audio_thread.daemon = True
        audio_thread.start()
    
    def wait_for_speech(self):
        """Wait for current speech to finish"""
        while self.is_playing:
            pygame.time.wait(100)

def test_connections():
    global ENABLE_TTS
    """Test connections to both Ollama and Chatterbox"""
    print("🔧 Initiating JARVIS systems diagnostic...")
    
    # Test Ollama
    print(f"\n📡 Testing neural network connection at: {OLLAMA_HOST}")
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Neural network connection established")
            models = response.json().get('models', [])
            if models:
                print(f"📦 Available models: {', '.join(m['name'] for m in models)}")
        else:
            print(f"❌ Connection responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Neural network connection failed: {e}")
        return False
    
    # Test Chatterbox TTS
    print(f"\n🔊 Testing voice synthesis at: {TTS_HOST}")
    try:
        response = requests.get(f"{TTS_HOST}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Voice synthesis online")
            print(f"   Processing unit: {health.get('device', 'unknown')}")
            
            # Test TTS generation
            if ENABLE_TTS:
                print("🧪 Testing voice generation...")
                test_response = requests.post(
                    f"{TTS_HOST}/tts",
                    json={"text": "Systems check complete. All systems operational."},
                    timeout=10
                )
                if test_response.status_code == 200:
                    print("✅ Voice synthesis operational")
                else:
                    print(f"⚠️  Voice synthesis test failed: {test_response.status_code}")
        else:
            print(f"❌ Voice system responded with status {response.status_code}")
            ENABLE_TTS = False
    except Exception as e:
        print(f"⚠️  Voice system connection failed: {e}")
        print("   Continuing without voice output...")
        
        ENABLE_TTS = False
    
    return True

# Create JARVIS prompt template
def create_jarvis_prompt():
    """Create a prompt template that enforces JARVIS personality"""
    return ChatPromptTemplate.from_messages([
        ("system", JARVIS_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

# Initialize ChatOllama with remote host
def create_llm():
    """Create LLM with JARVIS personality"""
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        temperature=0.7,  # Balanced for JARVIS's measured responses
        num_ctx= 8192,  # Adjust based on model capabilities
    )
    
    # Bind tools and ensure they're properly registered
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

# Define the agent node
def agent_node(state: AgentState) -> dict:
    """Process the current state and generate a response"""
    messages = state["messages"]
    
    total_chars = sum(len(msg.content) for msg in messages)
    approx_tokens = total_chars // 4
    
    print(f"Approximate context usage: {approx_tokens} tokens")

    # Extract conversation history and last message
    chat_history = messages[:-1] if len(messages) > 1 else []
    last_message = messages[-1].content if messages else ""
    
    # Create chain with JARVIS prompt
    llm = create_llm()
    prompt = create_jarvis_prompt()
    chain = prompt | llm
    
    # Generate response
    response = chain.invoke({
        "chat_history": chat_history,
        "input": last_message
    })
    
    # Log tool calls if any
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 Using tools: {', '.join([tc.get('name', 'unknown') for tc in response.tool_calls])}")
    
    # Return the new message to be added to state
    return {"messages": [response]}

# Define the conditional edge function
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine whether to use tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the flow
    return "end"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent_node)

# Create tool node
tool_node = ToolNode(tools)

# Add tool node directly - ToolNode is already a proper node function
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edge from agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph with increased recursion limit
app = workflow.compile(
    checkpointer=None,
    #recursion_limit=50  # Increased from default 25
)

# Interactive shell
async def main():
    print("🤖 JARVIS v1.0 - Just A Rather Very Intelligent System")
    print("=" * 60)
    
    # Run connection tests
    if not test_connections():
        print("\n⚠️  Some systems are not fully operational.")
        print("Shall I proceed with available functionality, Sir?")
    
    # Initialize TTS
    tts = JARVISTTS(TTS_HOST) if ENABLE_TTS else None
    
    print("\n" + "=" * 60)
    
    # Opening speech
    opening = "Good evening, Sir. All systems are now online. How may I assist you today?"
    print(f"JARVIS: {opening}")
    if tts:
        tts.speak(opening)
    
    print("\nType 'exit' when you wish to end our session.\n")
    
    # Initialize conversation state
    state = {"messages": []}
    
    while True:
        # Wait for any ongoing speech to finish before accepting input
        if tts:
            tts.wait_for_speech()
        
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            farewell = "Very well, Sir. I'll be here if you need me. Have a pleasant evening."
            print(f"\nJARVIS: {farewell}")
            if tts:
                tts.speak(farewell)
                tts.wait_for_speech()
            break
            
        if not user_input:
            empty_response = "I'm listening, Sir. Please feel free to share your request."
            print(f"JARVIS: {empty_response}\n")
            if tts:
                tts.speak(empty_response)
            continue
            
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            # Process through the graph
            result = await app.ainvoke(state)
            
            # Update state with full message history
            state = result
            
            # Get the last AI message (excluding ones with only tool calls)
            ai_response = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    # Check if it has content (not just tool calls)
                    if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        ai_response = msg.content
                        break
                    elif hasattr(msg, "content") and msg.content and hasattr(msg, "tool_calls") and not msg.tool_calls:
                        # Message has content but no tool calls
                        ai_response = msg.content
                        break
            
            if ai_response:
                # Print response
                print(f"\nJARVIS: {ai_response}\n")
                
                # Speak response
                if tts:
                    tts.speak(ai_response)
            else:
                # Try to get any AI message content as fallback
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_response = msg.content
                        print(f"\nJARVIS: {ai_response}\n")
                        if tts:
                            tts.speak(ai_response)
                        break
                else:
                    logger.warning("⚠️ No AI response message found in result")
                    logger.info("Message types in result:")
                    for i, msg in enumerate(result["messages"][-5:]):
                        logger.info(f"  {i}: {type(msg).__name__} - Has content: {bool(getattr(msg, 'content', None))} - Has tool_calls: {bool(getattr(msg, 'tool_calls', None))}")
            
        except Exception as e:
            error_response = f"I apologise, Sir, but I've encountered an unexpected error: {str(e)}. Shall I attempt to diagnose the issue?"
            print(f"\nJARVIS: {error_response}\n")
            if tts:
                tts.speak(error_response)

if __name__ == "__main__":
    # First run a detailed connection test
    print("🔧 JARVIS System Initialization")
    print("=" * 60)
    
    # Try to run the main program
    asyncio.run(main())