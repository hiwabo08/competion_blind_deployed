


































































import base64
import heapq
import io
import json
import os
import re
import smtplib
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from dotenv import load_dotenv

# Load environment variables from .env (local development only)
load_dotenv()

try:

    from PIL import Image, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO as _UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
except Exception:
    TESSERACT_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    from deepface import DeepFace
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

import atexit

app = Flask(__name__)
app.secret_key = "visionassist2025"
CORS(app)



GROQ_API_KEY_1 = os.environ.get("GROQ_API_KEY_1", "")
GROQ_API_KEY_2 = os.environ.get("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.environ.get("GROQ_API_KEY_3", "")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

N8N_FORM_URL = os.environ.get("N8N_FORM_URL", "")
N8N_CHAT_WEBHOOK = os.environ.get("N8N_CHAT_WEBHOOK", "")
N8N_RAG_IMAGE_URL = os.environ.get("N8N_RAG_IMAGE_URL", "")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "your_twilio_sid")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "your_twilio_token")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+1234567890")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS", "your_email@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "your_app_password")
EMERGENCY_EMAIL = os.environ.get("EMERGENCY_EMAIL", "emergency_contact@gmail.com")

MEMORY_DIR = "memory_objects"
MEMORY_INDEX_FILE = "memory_index.json"
CONVERSATION_LOG = "conversation_log.txt"
CONVERSATION_MAX_LINES = 300
MEMORY_CONTEXT_LINES = 60

for _d in [MEMORY_DIR, "known_faces", "Currency"]:
    if not os.path.exists(_d):
        os.makedirs(_d)

_groq_key_lock = Lock()
_groq_key_states: Dict[str, Dict] = {}


def _init_key_state(key: str):
    if key not in _groq_key_states:
        _groq_key_states[key] = {"failures": 0, "last_used": 0.0, "cooldown_until": 0.0}


def _mark_key_success(key: str):
    with _groq_key_lock:
        _init_key_state(key)
        _groq_key_states[key]["failures"] = 0
        _groq_key_states[key]["last_used"] = time.time()


def _mark_key_failure(key: str, is_rate_limit: bool = False):
    with _groq_key_lock:
        _init_key_state(key)
        _groq_key_states[key]["failures"] += 1
        _groq_key_states[key]["last_used"] = time.time()
        if is_rate_limit:
            cooldown = min(60 * (2 ** (_groq_key_states[key]["failures"] - 1)), 300)
            _groq_key_states[key]["cooldown_until"] = time.time() + cooldown


def _is_key_available(key: str) -> bool:
    if not key or key.startswith("your_"):
        return False
    with _groq_key_lock:
        _init_key_state(key)
        if time.time() < _groq_key_states[key]["cooldown_until"]:
            return False
    return True


def _get_best_key(priority: str = "high") -> str:
    order = (
        [GROQ_API_KEY_3, GROQ_API_KEY_1, GROQ_API_KEY_2]
        if priority == "low"
        else [GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3]
    )
    for k in order:
        if _is_key_available(k):
            return k
    valid = [k for k in order if k]
    return valid[0] if valid else GROQ_API_KEY_1


def groq_request(payload: dict, priority: str = "high", max_retries: int = 3) -> Optional[dict]:
    all_keys = [GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3]
    
    # For high-priority requests (traffic light, page reader), aggressive retry with all keys
    if priority == "high":
        timeout_secs = 12  # Short timeout for high-priority
        # Try all 3 keys multiple times before giving up
        for attempt in range(4):  # 4 attempts to cycle through keys
            for key in all_keys:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                try:
                    resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=timeout_secs)
                    if resp.status_code == 200:
                        _mark_key_success(key)
                        return resp.json()
                    elif resp.status_code == 429:
                        _mark_key_failure(key, is_rate_limit=True)
                    elif resp.status_code in (401, 403):
                        _mark_key_failure(key)
                    else:
                        _mark_key_failure(key)
                except requests.exceptions.Timeout:
                    _mark_key_failure(key)
                except Exception as e:
                    print(f"Groq error: {e}")
                    _mark_key_failure(key)
            
            # If we're not on the last attempt, wait briefly before retrying
            if attempt < 3:
                time.sleep(0.3)
        
        # All attempts exhausted
        return None
    
    # For low-priority requests, use original logic with key rotation and longer timeout
    tried_keys = set()
    for attempt in range(max_retries):
        key = _get_best_key(priority)
        if key in tried_keys:
            for k in all_keys:
                if k not in tried_keys and _is_key_available(k):
                    key = k
                    break
        tried_keys.add(key)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(GROQ_URL, json=payload, headers=headers, timeout=45)
            if resp.status_code == 200:
                _mark_key_success(key)
                return resp.json()
            elif resp.status_code == 429:
                _mark_key_failure(key, is_rate_limit=True)
                time.sleep(2 * (attempt + 1))
            elif resp.status_code in (401, 403):
                _mark_key_failure(key)
            else:
                _mark_key_failure(key)
        except requests.exceptions.Timeout:
            _mark_key_failure(key)
            time.sleep(1)
        except Exception as e:
            print(f"Groq exception: {e}")
            _mark_key_failure(key)
    return None


_global_stop_flag = False
_global_stop_flag_lock = Lock()
_global_stop_sequence = 0


def set_global_stop(value: bool):
    global _global_stop_flag, _global_stop_sequence
    with _global_stop_flag_lock:
        _global_stop_flag = value
        if value:
            _global_stop_sequence += 1


def is_global_stopped() -> bool:
    with _global_stop_flag_lock:
        return _global_stop_flag


def get_global_stop_sequence() -> int:
    with _global_stop_flag_lock:
        return _global_stop_sequence


_mode_lock = Lock()
_active_mode = "idle"
_active_mode_label = ""
_voice_stopped = False
_voice_stop_lock = Lock()


def get_active_mode() -> str:
    with _mode_lock:
        return _active_mode


def set_active_mode(mode: str, label: str = "") -> dict:
    global _active_mode, _active_mode_label
    with _mode_lock:
        if _active_mode == mode:
            return {"success": True, "was_same": True}
        if _active_mode not in ("idle", "camera", "assistant"):
            return {
                "success": False,
                "conflict": _active_mode,
                "label": _active_mode_label,
                "message": f"I am currently running {_active_mode_label}. Please say stop all first, then try again.",
            }
        _active_mode = mode
        _active_mode_label = label or mode
        return {"success": True}


def release_mode(mode: str):
    global _active_mode, _active_mode_label
    with _mode_lock:
        if _active_mode == mode:
            _active_mode = "idle"
            _active_mode_label = ""


def force_stop_all_modes():
    global _active_mode, _active_mode_label
    with _mode_lock:
        _active_mode = "idle"
        _active_mode_label = ""
    set_global_stop(True)
    stop_voice_output()


def stop_voice_output():
    global _voice_stopped
    with _voice_stop_lock:
        _voice_stopped = True


def resume_voice_output():
    global _voice_stopped
    with _voice_stop_lock:
        _voice_stopped = False


def is_voice_stopped() -> bool:
    with _voice_stop_lock:
        return _voice_stopped


_conv_log_lock = Lock()


def log_conversation(speaker: str, text: str):
    if not text:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {speaker}: {text}\n"
    with _conv_log_lock:
        with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        try:
            with open(CONVERSATION_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > CONVERSATION_MAX_LINES:
                with open(CONVERSATION_LOG, "w", encoding="utf-8") as f:
                    f.writelines(lines[-CONVERSATION_MAX_LINES:])
        except Exception:
            pass


def get_conversation_context() -> str:
    try:
        with _conv_log_lock:
            if not os.path.exists(CONVERSATION_LOG):
                return ""
            with open(CONVERSATION_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
        recent = lines[-MEMORY_CONTEXT_LINES:]
        return "".join(recent).strip()
    except Exception:
        return ""


_memory_index_lock = Lock()


def load_memory_index() -> List[dict]:
    try:
        if os.path.exists(MEMORY_INDEX_FILE):
            with open(MEMORY_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_memory_index(index: List[dict]):
    with _memory_index_lock:
        with open(MEMORY_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


def find_memory_by_name(query: str) -> Optional[dict]:
    if not query:
        return None
    index = load_memory_index()
    query_l = query.lower().strip()
    for entry in index:
        if entry["name"].lower() == query_l:
            return entry
    for entry in index:
        if query_l in entry["name"].lower() or entry["name"].lower() in query_l:
            return entry
    query_words = set(query_l.split())
    best_score = 0
    best_entry = None
    for entry in index:
        entry_words = set(entry["name"].lower().split())
        score = len(query_words & entry_words)
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_score > 0:
        return best_entry
    for entry in index:
        if query_l in entry.get("description", "").lower():
            return entry
    return None


def list_all_memories() -> List[dict]:
    return load_memory_index()


def delete_memory(memory_id: str) -> bool:
    index = load_memory_index()
    new_index = []
    deleted = False
    for entry in index:
        if entry["id"] == memory_id:
            deleted = True
            for photo in entry.get("photos", []):
                photo_path = os.path.join(MEMORY_DIR, photo)
                if os.path.exists(photo_path):
                    try:
                        os.remove(photo_path)
                    except Exception:
                        pass
        else:
            new_index.append(entry)
    if deleted:
        save_memory_index(new_index)
    return deleted


VOICE_ERRORS = {
    "no_camera": "The camera is not available. Please make sure your camera is connected and not used by another app. Then say My Eye start camera.",
    "no_face": "I could not find a face in the camera. Please make sure you are looking directly at the camera with good lighting. Then try again.",
    "no_speech": "I did not hear you clearly. Please say My Eye, wait a moment, then speak your command.",
    "api_busy": "The AI is busy right now. Please wait a few seconds and try again.",
    "network": "There seems to be a network problem. Please check your internet connection and try again.",
    "no_memory": "I do not have that item in my memory yet. To save it, say My Eye remember this, then describe the object and where you keep it.",
    "no_faces_saved": "I do not have any saved faces yet. To save someone's face, say My Eye capture face.",
    "mode_busy": "I am currently busy with another task. Please say My Eye stop all, then try your new command.",
    "no_query": "I did not understand what you are looking for. Please say My Eye find, then clearly say the object name.",
    "camera_needed": "I need the camera to do this. Please say My Eye start camera, wait for it to open, then try again.",
    "timeout": "I am taking too long to respond. Please try again.",
}


def get_voice_error(error_type: str, extra: str = "") -> str:
    base = VOICE_ERRORS.get(error_type, "Something went wrong. Please say My Eye and try again.")
    if extra:
        return f"{base} {extra}"
    return base


def _pattern_fallback(command_text: str) -> dict:
    t = command_text.lower()
    action = "general_question"
    extracted_name = ""
    extracted_query = ""
    extracted_destination = ""
    is_live = "live" in t or "lagatar" in t
    needs_camera = False
    countdown_needed = False
    voice_response = "I am working on your request right now."

    if any(w in t for w in ["stop all", "sab band", "band karo sab", "stop everything"]):
        action = "stop_all"
        voice_response = "Stopping everything right now."
    elif any(w in t for w in ["stop voice", "chup", "stop speaking"]):
        action = "stop_voice"
        voice_response = "I will stop speaking now."
    elif any(w in t for w in ["resume voice", "bolo shuru", "speak again"]):
        action = "resume_voice"
        voice_response = "I am speaking again."
    elif any(w in t for w in ["start assistant", "assistant mode", "assistant shuru"]):
        action = "assistant_mode"
        voice_response = "Hello dear friend! I am your VisionAssist assistant. What would you like me to do?"
    elif any(w in t for w in ["food", "khana", "khaana", "bhojan"]):
        action = "food_live" if is_live else "food_detection"
        needs_camera = True
        voice_response = "Starting the food detector now. Please point the camera towards your food."
    elif any(w in t for w in ["stair", "seedhi", "stairs", "step"]):
        action = "stairs_live" if is_live else "stairs_detection"
        needs_camera = True
        voice_response = "I will check the stairs for you. Please point the camera towards the stairs."
    elif any(w in t for w in ["traffic", "signal"]):
        action = "traffic_live" if is_live else "traffic_detection"
        needs_camera = True
        voice_response = "Checking the traffic light now."
    elif any(w in t for w in ["page", "read", "book", "padhna", "padho"]):
        action = "page_reader"
        needs_camera = True
        voice_response = "Page reader starting. Please hold your book in front of the camera."
    elif any(w in t for w in ["rag", "scan document"]):
        action = "rag_scanner"
        needs_camera = True
        voice_response = "Document scanner ready. Hold any document in front of the camera."
    elif any(w in t for w in ["capture face", "photo nikal", "face save", "save face"]):
        action = "face_capture"
        needs_camera = True
        countdown_needed = True
        m = re.search(r"(?:save as|naam se|as|named?)\s+([a-zA-Z\u0900-\u097F]+)", t)
        if m:
            extracted_name = m.group(1).strip().title()
        voice_response = f"Face capture ready. Please make sure the person is looking at the camera with good lighting."
    elif any(w in t for w in ["recognize", "pehchano", "who is", "kaun hai"]):
        action = "face_recognize"
        needs_camera = True
        voice_response = "Looking at the person in the camera now."
    elif any(w in t for w in ["money", "note", "paisa", "currency", "rupee"]):
        action = "money_detect"
        needs_camera = True
        voice_response = "Please hold the currency note clearly in front of the camera."
    elif any(w in t for w in ["find", "kahan hai", "dhundo", "locate"]):
        action = "object_find"
        needs_camera = True
        m = re.search(r"(?:find my?|find|kahan hai|locate)\s+(.+)", t)
        if m:
            extracted_query = m.group(1).strip()
            extracted_query = re.sub(r"\b(mera|meri|my|the)\b", "", extracted_query).strip()
        voice_response = f"I will help you find your {extracted_query or 'item'}."
    elif any(w in t for w in ["remember", "yaad rakh", "save this"]):
        action = "memory_remember"
        needs_camera = True
        countdown_needed = True
        voice_response = "Sure! Please describe the object and where you keep it."
    elif any(w in t for w in ["do you remember", "where is my", "mera kahan", "memory find"]):
        action = "memory_find"
        m = re.search(r"(?:where is my?|mera|meri|do you remember my?)\s+(.+)", t)
        if m:
            extracted_query = m.group(1).strip()
        voice_response = f"Let me check my memory for {extracted_query or 'that item'}."
    elif any(w in t for w in ["list memory", "what do you remember", "kya yaad hai"]):
        action = "memory_list"
        voice_response = "Let me tell you everything I have in my memory."
    elif any(w in t for w in ["navigate", "jaana hai", "directions to", "go to"]):
        action = "navigate_outdoor"
        m = re.search(r"(?:navigate to|go to|jaana hai)\s+(.+)", t)
        if m:
            extracted_destination = m.group(1).strip()
        voice_response = f"Finding the route to {extracted_destination or 'your destination'} now."
    elif any(w in t for w in ["sos", "emergency", "bachao", "help emergency"]):
        action = "sos"
        voice_response = "EMERGENCY! I am activating SOS alert right now!"
    elif any(w in t for w in ["location", "where am i", "kahan hoon"]):
        action = "get_location"
        voice_response = "Getting your GPS location right now."
    elif any(w in t for w in ["start camera", "camera on", "open camera"]):
        action = "camera_start"
        voice_response = "Opening the camera for you now."
    elif any(w in t for w in ["stop camera", "camera off", "close camera"]):
        action = "camera_stop"
        voice_response = "Stopping the camera now."
    elif any(w in t for w in ["what is in front", "describe", "kya hai", "yahan kya", "scene"]):
        action = "describe_scene"
        needs_camera = True
        voice_response = "Let me look at what is in front of you right now."
    else:
        action = "general_question"
        voice_response = "Let me find the answer to your question."

    return {
        "action": action,
        "extracted_name": extracted_name,
        "extracted_query": extracted_query,
        "extracted_destination": extracted_destination,
        "is_live": is_live,
        "confidence": 0.75,
        "language": "hinglish",
        "voice_response": voice_response,
        "needs_camera": needs_camera,
        "countdown_needed": countdown_needed,
        "explanation": "Pattern fallback",
    }


def ai_interpret_command(command_text: str) -> dict:
    AI_COMMAND_SYSTEM = """You are VisionAssist AI command interpreter for a blind person.
Analyze the user's voice command (ANY language) and return the correct action.

Available actions: food_detection, food_live, stairs_detection, stairs_live,
traffic_detection, traffic_live, page_reader, rag_scanner, face_capture,
face_recognize, money_detect, object_find, memory_remember, memory_find,
memory_list, assistant_mode, navigate_outdoor, navigate_indoor, sos,
stop_all, stop_voice, resume_voice, get_location, camera_start, camera_stop,
describe_scene, general_question

Return ONLY valid JSON:
{
  "action": "food_detection",
  "extracted_name": "",
  "extracted_query": "",
  "extracted_destination": "",
  "is_live": false,
  "confidence": 0.95,
  "language": "hindi",
  "voice_response": "Starting the food detector now. Please point camera at your food.",
  "needs_camera": true,
  "countdown_needed": false,
  "explanation": "User asked about food"
}"""
    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": AI_COMMAND_SYSTEM},
                {"role": "user", "content": f'Voice command: "{command_text}"'},
            ],
            "max_tokens": 350,
            "temperature": 0.1,
        }
        result = groq_request(payload, priority="high")
        if result:
            raw = result["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        print(f"AI interpreter error: {e}")
    return _pattern_fallback(command_text)


_assistant_active = False
_assistant_lock = Lock()


def set_assistant_active(val: bool):
    global _assistant_active
    with _assistant_lock:
        _assistant_active = val


def get_assistant_active() -> bool:
    with _assistant_lock:
        return _assistant_active


def preprocess_image(b64: str, max_w: int = 800) -> str:
    if not PIL_AVAILABLE:
        return b64
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        ow, oh = img.size
        if ow > max_w:
            img = img.resize((max_w, int(oh * max_w / ow)), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.1)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return b64


def groq_vision_call(system_prompt: str, user_prompt: str, frames: list, max_tokens: int = 300, priority: str = "high") -> str:
    if not frames:
        raise ValueError("No frames provided")
    content = [{"type": "text", "text": user_prompt}]
    for f in frames[:6]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}})
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    result = groq_request(payload, priority=priority)
    if result is None:
        raise RuntimeError(get_voice_error("api_busy"))
    text = result["choices"][0]["message"]["content"].strip()
    if not text:
        raise ValueError("Empty response from AI. Please try again.")
    return text


def save_memory_object(description: str, frames: List[str]) -> dict:
    if not frames:
        return {"success": False, "error": get_voice_error("camera_needed")}
    if not description:
        return {"success": False, "error": "Please describe what you want me to remember."}

    parse_prompt = (
        f'Parse this object description for a memory system:\n"{description}"\n\n'
        f"Return ONLY valid JSON:\n"
        f'{{"name": "short object name", "location": "where it is", "color": "color if mentioned or empty string"}}\n'
        f"Return only the JSON, nothing else."
    )
    parsed_name = description
    parsed_location = "unknown location"
    parsed_color = ""
    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": parse_prompt}],
            "max_tokens": 100,
            "temperature": 0.1,
        }
        result = groq_request(payload, priority="high")
        if result:
            raw = result["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                parsed = json.loads(json_match.group())
                parsed_name = parsed.get("name", description)
                parsed_location = parsed.get("location", "unknown location")
                parsed_color = parsed.get("color", "")
    except Exception as e:
        print(f"Memory parse error: {e}")
        words = description.lower().split()
        if "in" in words:
            idx = words.index("in")
            parsed_name = " ".join(words[:idx])
            parsed_location = " ".join(words[idx + 1:])
        elif "on" in words:
            idx = words.index("on")
            parsed_name = " ".join(words[:idx])
            parsed_location = " ".join(words[idx + 1:])

    memory_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    photo_filenames = []

    for i, frame_b64 in enumerate(frames[:2]):
        safe_name = re.sub(r"[^a-z0-9_]", "_", parsed_name.lower())[:30]
        filename = f"{memory_id}_{safe_name}_{i+1}.jpg"
        filepath = os.path.join(MEMORY_DIR, filename)
        try:
            img_bytes = base64.b64decode(frame_b64)
            if CV2_AVAILABLE:
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    cv2.imwrite(filepath, frame)
                    photo_filenames.append(filename)
            elif PIL_AVAILABLE:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.save(filepath, "JPEG", quality=90)
                photo_filenames.append(filename)
        except Exception as e:
            print(f"Photo save error: {e}")

    if not photo_filenames:
        return {"success": False, "error": "I could not save the photos. Please make sure the camera is working."}

    entry = {
        "id": memory_id,
        "name": parsed_name,
        "location": parsed_location,
        "description": description,
        "color": parsed_color,
        "photos": photo_filenames,
        "timestamp": timestamp,
        "times_found": 0,
    }
    index = load_memory_index()
    index = [e for e in index if e["name"].lower() != parsed_name.lower()]
    index.append(entry)
    save_memory_index(index)
    log_conversation("SYSTEM", f"Memory saved: {parsed_name} at {parsed_location}")

    return {
        "success": True,
        "id": memory_id,
        "name": parsed_name,
        "location": parsed_location,
        "color": parsed_color,
        "photos": len(photo_filenames),
        "voice_response": (
            f"I have saved it! I will remember that your {parsed_color + ' ' if parsed_color else ''}{parsed_name} "
            f"is at {parsed_location}. I took {len(photo_filenames)} photo{'s' if len(photo_filenames) > 1 else ''} "
            f"so I can recognize it visually. Just say My Eye where is my {parsed_name} and I will help you find it."
        ),
    }


def _try_visual_confirm_memory(memory_entry: dict, frames: List[str]) -> str:
    try:
        if not memory_entry.get("photos"):
            return ""
        saved_b64 = []
        for photo_name in memory_entry["photos"][:2]:
            photo_path = os.path.join(MEMORY_DIR, photo_name)
            if os.path.exists(photo_path):
                with open(photo_path, "rb") as pf:
                    saved_b64.append(base64.b64encode(pf.read()).decode())
        if not saved_b64:
            return ""
        name = memory_entry["name"]
        location = memory_entry["location"]
        color = memory_entry.get("color", "")
        color_str = f"the {color} " if color else ""
        verify_system = (
            "You help a blind person find an object using memory photos and current camera. "
            "Reference photos show the object. Current photos are the live camera. "
            "If you see the object: describe EXACTLY where it is (left, right, center, near, far, on what surface). "
            "If not visible: say clearly it is not in view. 2 sentences max. Natural spoken English only."
        )
        content = [{"type": "text", "text": f"Looking for {color_str}{name} (usually at {location}). Reference photos first, then current camera:"}]
        for b64 in saved_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        content.append({"type": "text", "text": "Current camera:"})
        for b64 in frames[:2]:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": verify_system}, {"role": "user", "content": content}],
            "max_tokens": 150,
            "temperature": 0.1,
        }
        result_json = groq_request(payload, priority="high")
        if result_json:
            scene_desc = result_json["choices"][0]["message"]["content"].strip()
            not_found = ["not visible", "not see", "cannot see", "can't see", "not in view", "not found"]
            if any(w in scene_desc.lower() for w in not_found):
                return ""
            return f"I can see it! {scene_desc}"
    except Exception as e:
        print(f"Visual confirm error: {e}")
    return ""


def find_memory_object(query: str, current_frames: List[str]) -> dict:
    memory_entry = find_memory_by_name(query)
    if not memory_entry:
        if current_frames:
            result = find_object_in_frames(current_frames, query)
            if result["success"]:
                return {"found_in_memory": False, "confirmed_in_scene": True, "message": result["result"]}
        return {
            "found_in_memory": False,
            "confirmed_in_scene": False,
            "message": get_voice_error("no_memory", f"If you want me to remember it, say My Eye remember my {query}."),
        }

    name = memory_entry["name"]
    location = memory_entry["location"]
    color = memory_entry.get("color", "")
    color_str = f"the {color} " if color else ""

    base_message = (
        f"Yes, I remember, dear friend! You keep {color_str}{name} at {location}. "
        f"Please go to {location}. When you are there, say My Eye I am at the location and I will confirm it on camera."
    )

    confirmed_in_scene = False
    if current_frames and memory_entry.get("photos"):
        confirmed_text = _try_visual_confirm_memory(memory_entry, current_frames)
        if confirmed_text:
            confirmed_in_scene = True
            base_message = confirmed_text + f" It is the {color_str}{name} you usually keep at {location}."

    try:
        index = load_memory_index()
        for e in index:
            if e["id"] == memory_entry["id"]:
                e["times_found"] = e.get("times_found", 0) + 1
        save_memory_index(index)
    except Exception:
        pass

    log_conversation("SYSTEM", f"Memory recall: {name} -> {location}")
    return {
        "found_in_memory": True,
        "name": name,
        "location": location,
        "color": color,
        "confirmed_in_scene": confirmed_in_scene,
        "message": base_message,
    }


def _finder_gemini(frames_b64: list, object_query: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        OBJECT_FINDER_SYSTEM = (
            "You are VisionAssist AI helping a blind person find a lost object. "
            "Analyze images from a room scan and locate the specific object. "
            "Say direction (left/right/behind/ahead), distance (very close/nearby/across room), "
            "nearby landmarks, and height from ground. "
            "If NOT visible: say you could not find it. 3 sentences max. No markdown."
        )
        parts = [
            {
                "text": (
                    f"{OBJECT_FINDER_SYSTEM}\n\n"
                    f"These {len(frames_b64)} images are from a 360-degree room scan. "
                    f"The user is searching for: {object_query}\n\n"
                    f"Analyze ALL frames together and locate the object."
                )
            }
        ]
        for i, b64 in enumerate(frames_b64):
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
            parts.append({"text": f"[Frame {i+1} of {len(frames_b64)} at {int(i * 360 / max(len(frames_b64), 1))}° rotation]"})
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"maxOutputTokens": 300, "temperature": 0.3},
        }
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45,
        )
        if resp.status_code == 200:
            candidates = resp.json().get("candidates", [])
            if candidates:
                parts_out = candidates[0].get("content", {}).get("parts", [])
                if parts_out:
                    result = parts_out[0].get("text", "").strip()
                    if result:
                        return result
        return None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def _finder_groq(frames_b64: list, object_query: str) -> Optional[str]:
    OBJECT_FINDER_SYSTEM = (
        "You are VisionAssist AI helping a blind person find a lost object. "
        "Analyze images from a room scan and locate the specific object. "
        "Say direction, distance, nearby landmarks, and height from ground. "
        "If NOT visible: say you could not find it. 3 sentences max. No markdown."
    )
    n = len(frames_b64)
    sample = frames_b64 if n <= 3 else [frames_b64[0], frames_b64[n // 2], frames_b64[n - 1]]
    not_found = ["cannot", "not visible", "not find", "could not find", "don't see"]
    best_result = None
    for i, frame_b64 in enumerate(sample):
        angle = int(i * 360 / max(len(sample), 1))
        try:
            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": OBJECT_FINDER_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Frame at {angle}° rotation. Looking for: {object_query}. Is it visible?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}", "detail": "high"}},
                        ],
                    },
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            }
            result_json = groq_request(payload, priority="low")
            if result_json:
                result = result_json["choices"][0]["message"]["content"].strip()
                if result and not any(p in result.lower() for p in not_found):
                    return result
                if not best_result and result:
                    best_result = result
            time.sleep(1)
        except Exception as e:
            print(f"Groq finder error: {e}")
    return best_result


def find_object_in_frames(frames_b64: list, object_query: str) -> dict:
    if not frames_b64:
        return {"success": False, "result": get_voice_error("camera_needed"), "model_used": "none", "frames_analyzed": 0}
    if not object_query:
        return {"success": False, "result": get_voice_error("no_query"), "model_used": "none", "frames_analyzed": 0}
    result = _finder_gemini(frames_b64, object_query)
    if result:
        return {"success": True, "result": result, "model_used": "Gemini 1.5 Flash", "frames_analyzed": len(frames_b64)}
    result = _finder_groq(frames_b64, object_query)
    if result:
        return {"success": True, "result": result, "model_used": "Groq Llama Vision", "frames_analyzed": min(3, len(frames_b64))}
    return {
        "success": False,
        "result": (
            f"Dear friend, I could not find your {object_query} in the camera scan. "
            f"It may be hidden or out of camera view. "
            f"Please try looking in the last place you used it and feel along surfaces slowly."
        ),
        "model_used": "fallback",
        "frames_analyzed": len(frames_b64),
    }


def ask_groq_general(question: str) -> str:
    conv_ctx = get_conversation_context()
    system_msg = (
        "You are VisionAssist AI, a helpful voice assistant for a blind person. "
        "Answer briefly in natural spoken language. No markdown. 2-3 sentences max. "
        + (f"Context:\n{conv_ctx[-400:]}" if conv_ctx else "")
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": question}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    result = groq_request(payload, priority="high")
    if result:
        return result["choices"][0]["message"]["content"].strip()
    return get_voice_error("api_busy")


def ask_rag_assistant(question: str, context: str = "") -> str:
    log_conversation("USER", question)
    try:
        conv_context = get_conversation_context()
        if conv_context:
            context = f"Conversation history:\n{conv_context}\n\n{context}"
        payload = {"message": question, "context": context, "timestamp": time.time()}
        response = requests.post(N8N_CHAT_WEBHOOK, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", str(data)) if isinstance(data, dict) else str(data)
            log_conversation("SYSTEM", answer[:200])
            return answer
        return ask_groq_general(question)
    except Exception as e:
        print(f"RAG error: {e}")
        return ask_groq_general(question)


arduino_connected = False
arduino = None
front_distance = "0"
steps = "0"
sonar_active = True
serial_lock = Lock()
arduino_port = "COM10"
detection_mode = "normal"
navigation_mode = "outdoor"


def init_arduino():
    global arduino, arduino_connected
    if not SERIAL_AVAILABLE:
        return False
    try:
        arduino = serial.Serial("COM10", 9600, timeout=1)
        time.sleep(2)
        arduino_connected = True
        print("Arduino connected on COM10")
        threading.Thread(target=read_serial, daemon=True).start()
        return True
    except Exception as e:
        print(f"Arduino connection error: {e}")
        arduino_connected = False
        return False


def read_serial():
    global front_distance, steps, arduino_connected
    while sonar_active:
        if arduino and arduino_connected:
            try:
                with serial_lock:
                    if arduino.in_waiting > 0:
                        data = arduino.readline().decode().strip()
                        if data.startswith("DISTANCE:"):
                            try:
                                parts = data.split(",")
                                front_distance = parts[0].split(":")[1]
                                steps = parts[1].split(":")[1]
                            except Exception:
                                pass
            except Exception:
                arduino_connected = False
        time.sleep(0.1)


try:
    init_arduino()
except Exception:
    print("Arduino not connected")

yolo_model = None
if ULTRALYTICS_AVAILABLE and CV2_AVAILABLE:
    try:
        yolo_model = _UltralyticsYOLO("yolov8n.pt")
        print("YOLO model loaded")
    except Exception as _e:
        print(f"YOLO error: {_e}")

_myeye_yolo_model = None
_myeye_yolo_lock = threading.Lock()


def _get_myeye_yolo():
    global _myeye_yolo_model
    if not ULTRALYTICS_AVAILABLE or not CV2_AVAILABLE:
        return None
    if _myeye_yolo_model is not None:
        return _myeye_yolo_model
    with _myeye_yolo_lock:
        if _myeye_yolo_model is None:
            try:
                _myeye_yolo_model = _UltralyticsYOLO("yolov8n.pt")
                dummy = np.zeros((64, 64, 3), np.uint8)
                _myeye_yolo_model(dummy, verbose=False)
            except Exception as _ex:
                print(f"MyEye YOLO failed: {_ex}")
                _myeye_yolo_model = None
    return _myeye_yolo_model


sift = None
bf = None
if CV2_AVAILABLE:
    try:
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()
    except Exception as _e:
        print(f"SIFT initialization failed: {_e}")

latest_objects = []
camera_active = False
camera_lock = Lock()
sos_active = False
sos_timer = None
emergency_contacts = []
rag_detection_counter = 0
read_page_detection_counter = 0
last_rag_capture_time = 0
last_read_capture_time = 0

DOCUMENT_CLASSES = ["book", "page", "paper", "document", "magazine", "newspaper", "text", "letter", "note", "card", "flyer", "poster"]
YOLO_DOCUMENT_IDS = [73, 74, 75, 76, 77, 78]


@dataclass
class Node:
    id: str
    name: str
    x: float
    y: float
    description: str = ""


@dataclass
class Edge:
    from_node: str
    to_node: str
    distance: float
    direction: str
    description: str = ""


class IndoorNavigationSystem:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []

    def add_edge(self, edge: Edge):
        self.edges.setdefault(edge.from_node, []).append(edge)
        rev = Edge(
            from_node=edge.to_node,
            to_node=edge.from_node,
            distance=edge.distance,
            direction=self._rev_dir(edge.direction),
            description=edge.description,
        )
        self.edges.setdefault(edge.to_node, []).append(rev)

    def _rev_dir(self, d: str) -> str:
        return {"left": "right", "right": "left", "straight": "straight", "back": "forward"}.get(d, "straight")

    def dijkstra_shortest_path(self, start_id: str, end_id: str) -> Tuple[List[str], float, List[Dict]]:
        if start_id not in self.nodes or end_id not in self.nodes:
            return [], float("inf"), []
        distances = {n: float("inf") for n in self.nodes}
        distances[start_id] = 0
        previous = {n: None for n in self.nodes}
        pq = [(0, start_id)]
        visited = set()
        while pq:
            cur_d, cur = heapq.heappop(pq)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == end_id:
                break
            for edge in self.edges.get(cur, []):
                nb = edge.to_node
                nd = cur_d + edge.distance
                if nd < distances[nb]:
                    distances[nb] = nd
                    previous[nb] = (cur, edge)
                    heapq.heappush(pq, (nd, nb))
        path, directions = [], []
        cur = end_id
        while cur != start_id:
            if previous[cur] is None:
                return [], float("inf"), []
            prev_node, edge = previous[cur]
            path.insert(0, cur)
            directions.insert(0, {
                "from": prev_node,
                "to": cur,
                "from_name": self.nodes[prev_node].name,
                "to_name": self.nodes[cur].name,
                "distance": edge.distance,
                "direction": edge.direction,
                "description": edge.description,
            })
            cur = prev_node
        path.insert(0, start_id)
        return path, distances[end_id], directions

    def get_navigation_instructions(self, start_id: str, end_id: str) -> Dict:
        path, total, directions = self.dijkstra_shortest_path(start_id, end_id)
        if not path:
            return {"success": False, "error": "No path found"}
        instructions = []
        cumulative = 0
        for i, step in enumerate(directions):
            cumulative += step["distance"]
            instructions.append({
                "step": i + 1,
                "instruction": f"Walk {step['direction']} for {step['distance']:.0f} steps to reach {step['to_name']}",
                "direction": step["direction"],
                "steps": step["distance"],
                "cumulative_steps": cumulative,
            })
        return {"success": True, "total_steps": int(total), "instructions": instructions}

    def get_current_location(self, image_data: str, indoor_map: Dict) -> Optional[str]:
        try:
            prompt = (
                f"Based on this indoor map: {json.dumps(indoor_map, indent=2)}\n\n"
                f"Looking at the camera image, which room is the user currently in?\n"
                f"Available rooms: {[r['name'] for r in indoor_map.get('rooms', [])]}\n\n"
                f"Return ONLY the room ID."
            )
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ]}],
                "temperature": 0.2,
                "max_tokens": 50,
            }
            result = groq_request(payload, priority="high")
            if result:
                loc = result["choices"][0]["message"]["content"].strip().lower()
                loc = loc.replace('"', "").replace("'", "").strip()
                for room in indoor_map.get("rooms", []):
                    if loc == room["id"].lower() or loc in room["name"].lower():
                        return room["id"]
            if indoor_map.get("rooms"):
                return indoor_map["rooms"][0]["id"]
            return None
        except Exception as e:
            print(f"Location detection error: {e}")
            return indoor_map["rooms"][0]["id"] if indoor_map.get("rooms") else None


nav_system = IndoorNavigationSystem()
indoor_map_data = None


def load_emergency_contacts():
    global emergency_contacts
    try:
        if os.path.exists("emergency_contacts.json"):
            with open("emergency_contacts.json", "r") as f:
                emergency_contacts = json.load(f)
        else:
            emergency_contacts = [{"name": "Emergency Services", "number": "911", "email": "", "method": "call"}]
    except Exception as e:
        print(f"Error loading contacts: {e}")
        emergency_contacts = []


load_emergency_contacts()


def detect_objects(frame) -> list:
    if yolo_model is None:
        return []
    try:
        conf = 0.15 if detection_mode == "rapid" else 0.25
        results = yolo_model(frame, conf=conf, verbose=False)
        detected = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected.append(yolo_model.names[cls_id])
        return list(dict.fromkeys(detected))
    except Exception:
        return []


def describe_objects_simple(objects: list) -> str:
    if not objects:
        return "I don't see any objects right now."
    unique = list(dict.fromkeys(objects))
    if len(unique) == 1:
        return f"I can see a {unique[0]} in front of you."
    elif len(unique) == 2:
        return f"I can see a {unique[0]} and a {unique[1]} in front of you."
    else:
        return f"I can see {', '.join(unique[:-1])}, and a {unique[-1]} in front of you."


def check_text_in_image(image_pil) -> Tuple[bool, str]:
    if not TESSERACT_AVAILABLE:
        return True, ""
    try:
        text = pytesseract.image_to_string(image_pil)
        text_found = text.strip()
        return len(text_found) > 10, text_found
    except Exception:
        return True, ""


def send_sms_alert(location: str, maps_link: str) -> bool:
    try:
        if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_ACCOUNT_SID != "your_twilio_sid":
            client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = f"SOS EMERGENCY! Your contact needs help!\nLocation: {location}\nMaps: {maps_link}"
            for contact in emergency_contacts:
                if contact.get("number") and contact["method"] in ["sms", "both"]:
                    client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=contact["number"])
        return True
    except Exception:
        return False


def send_email_alert(location: str, maps_link: str) -> bool:
    try:
        if EMAIL_ADDRESS and EMAIL_ADDRESS != "your_email@gmail.com":
            msg = MIMEMultipart()
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = EMERGENCY_EMAIL
            msg["Subject"] = "SOS EMERGENCY ALERT"
            body = f"<h2>SOS EMERGENCY ALERT</h2><p>Location: {location}</p><p>Maps: <a href='{maps_link}'>{maps_link}</a></p>"
            msg.attach(MIMEText(body, "html"))
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            return True
    except Exception:
        return False


def _hsv_tl_color(frame_b64: str) -> str:
    if not CV2_AVAILABLE:
        return "UNKNOWN"
    model = _get_myeye_yolo()
    if model is None:
        return "UNKNOWN"
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return "UNKNOWN"
        results = model(frame, verbose=False, conf=0.30, iou=0.45)
        h, w = frame.shape[:2]
        best_conf = 0
        best_color = "UNKNOWN"
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id != 9 or conf <= 0.30:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (3, 3), 0), cv2.COLOR_BGR2HSV)
                r1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
                r2 = cv2.inRange(hsv, np.array([160, 120, 80]), np.array([180, 255, 255]))
                yc = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([38, 255, 255]))
                gc = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([100, 255, 255]))
                scores = {
                    "RED": int(cv2.countNonZero(cv2.add(r1, r2))),
                    "YELLOW": int(cv2.countNonZero(yc)),
                    "GREEN": int(cv2.countNonZero(gc)),
                }
                best = max(scores, key=scores.get)
                if scores[best] > 100 and conf > best_conf:
                    best_conf = conf
                    best_color = best
        return best_color
    except Exception as e:
        print(f"HSV error: {e}")
        return "UNKNOWN"


def create_structured_map_from_response(text: str, frames: list) -> dict:
    rooms, paths = [], []
    keywords = ["kitchen", "living", "bedroom", "bathroom", "dining", "hall", "entrance", "office"]
    found_rooms = list(dict.fromkeys([kw for kw in keywords if kw in text.lower()]))
    if not found_rooms:
        found_rooms = ["entrance", "living_room", "kitchen"][: max(2, len(frames) // 3)]
    for i, name in enumerate(found_rooms):
        rooms.append({
            "id": name,
            "name": name.replace("_", " ").title(),
            "x": 100 + i * 200,
            "y": 200 + (i % 2) * 100,
            "description": f"The {name.replace('_', ' ')} area",
        })
    for i in range(len(rooms) - 1):
        paths.append({
            "from": rooms[i]["id"],
            "to": rooms[i + 1]["id"],
            "distance": 12 + i * 3,
            "direction": "straight" if i % 2 == 0 else "right",
            "description": f"Walk from {rooms[i]['name']} to {rooms[i+1]['name']}",
        })
    return {"rooms": rooms, "paths": paths}


FOOD_SYSTEM = (
    "You are a food recognition assistant for visually impaired people. "
    "Analyze the images and identify all visible food. "
    "Include food names, estimated quantity, number of plates, glasses or cups. "
    "End with meal type: breakfast, lunch, dinner, or snack. "
    "3 sentences max. No markdown. Natural spoken English only."
)
FOOD_USER = "Identify all food visible in these frames. List each food item with quantity, count plates and glasses, state meal type. 3 sentences max."

STAIR_SYSTEM = (
    "You assist a blind person navigating stairs. Response is spoken aloud immediately. "
    "Natural spoken English only. No bullet points, no markdown. 3 to 5 sentences maximum. "
    "If stairs visible: step count estimate, direction UP or DOWN, handrail position, surface condition, any hazards. "
    "If no stairs: say No stairs visible, then describe what you see briefly."
)
STAIR_USER = "Analyze these frames. Describe stairs for a blind person who needs to navigate right now. Start with safety information."

TRAFFIC_SYSTEM = (
    "You are a traffic light detector for a blind pedestrian in India. "
    "RED light = pedestrians CAN cross (cars stopped). "
    "GREEN light = pedestrians MUST STOP (cars moving). "
    "Reply in MAXIMUM 6 words only. Examples: "
    "'Red light, safe to cross now.' "
    "'Green light, stop, cars are moving.' "
    "'Yellow light, wait, do not cross.' "
    "'No traffic light visible, be careful.' "
    "ONLY 6 words. No explanation. No markdown."
)
TRAFFIC_USER = "What color is the traffic light? Reply in 6 words only."

PAGE_READER_SYSTEM = (
    "You are helping a blind person read a book or document page. "
    "Read ALL text exactly as written, word for word, in order. "
    "If there is no text, describe the image in vivid detail. "
    "Be clear and natural for text-to-speech. No markdown formatting."
)
PAGE_READER_USER = (
    "These frames are from a short 2-3 second video of a book or document page. "
    "Analyze all frames together and read the complete page content aloud as if speaking to a blind person."
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/global-stop-status", methods=["GET"])
def global_stop_status():
    return jsonify({"stopped": is_global_stopped(), "sequence": get_global_stop_sequence()})


@app.route("/api/global-stop-reset", methods=["POST"])
def global_stop_reset():
    set_global_stop(False)
    return jsonify({"success": True, "stopped": False})


@app.route("/api/mode/global-stop", methods=["POST"])
def mode_global_stop():
    force_stop_all_modes()
    set_assistant_active(False)
    log_conversation("USER", "stop all")
    msg = "All stopped. Camera and all features have been turned off. To start again, say My Eye start camera."
    log_conversation("SYSTEM", msg)
    return jsonify({
        "success": True,
        "hard_stop": True,
        "message": msg,
        "voice_response": msg,
        "active_mode": "idle",
        "stop_sequence": get_global_stop_sequence(),
    })


@app.route("/api/mode/status", methods=["GET"])
def mode_status():
    return jsonify({
        "active_mode": get_active_mode(),
        "active_label": _active_mode_label,
        "voice_stopped": is_voice_stopped(),
        "global_stopped": is_global_stopped(),
        "assistant_active": get_assistant_active(),
    })


@app.route("/api/mode/set", methods=["POST"])
def mode_set():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "idle")
    label = data.get("label", mode)
    result = set_active_mode(mode, label)
    return jsonify(result)


@app.route("/api/mode/release", methods=["POST"])
def mode_release():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "")
    if mode:
        release_mode(mode)
    return jsonify({"success": True, "active_mode": get_active_mode()})


@app.route("/api/voice/stop", methods=["POST"])
def voice_stop():
    stop_voice_output()
    return jsonify({"success": True, "voice_stopped": True})


@app.route("/api/voice/resume", methods=["POST"])
def voice_resume():
    resume_voice_output()
    return jsonify({"success": True, "voice_stopped": False, "message": "Voice output resumed."})


@app.route("/api/voice/status", methods=["GET"])
def voice_status():
    return jsonify({"voice_stopped": is_voice_stopped()})


@app.route("/api/process-command", methods=["POST"])
def process_command():
    global navigation_mode
    try:
        data = request.get_json()
        command_text = data.get("command", "").strip()
        if not command_text:
            msg = get_voice_error("no_speech")
            return jsonify({"error": "empty_command", "message": msg}), 400

        log_conversation("USER", command_text)
        cleaned = command_text.lower().strip()
        for wake in ["my eye", "my i", "my ai", "hey vision", "vision assist"]:
            if cleaned.startswith(wake):
                cleaned = cleaned[len(wake):].strip(",. ")
                break

        if not cleaned:
            msg = "Yes dear friend, I am listening! What would you like me to do?"
            return jsonify({"status": "success", "action": "wake_only", "message": msg, "voice_response": msg})

        if get_assistant_active():
            return jsonify({
                "status": "success",
                "category": "assistant_query",
                "action": "assistant_query",
                "query": cleaned,
                "message": "Processing through assistant...",
                "voice_response": "Let me handle that for you.",
            })

        interpreted = ai_interpret_command(cleaned)
        action = interpreted.get("action", "general_question")
        is_live = interpreted.get("is_live", False)
        ext_name = interpreted.get("extracted_name", "")
        ext_query = interpreted.get("extracted_query", "")
        ext_dest = interpreted.get("extracted_destination", "")
        voice_resp = interpreted.get("voice_response", "Working on it right now.")
        needs_cam = interpreted.get("needs_camera", False)
        countdown = interpreted.get("countdown_needed", False)

        response = {
            "status": "success",
            "category": action,
            "original_command": command_text,
            "interpreted": interpreted,
            "voice_response": voice_resp,
            "needs_camera": needs_cam,
            "countdown_needed": countdown,
        }

        if action == "stop_all":
            force_stop_all_modes()
            set_assistant_active(False)
            response.update({
                "action": "global_stop",
                "hard_stop": True,
                "kill_camera": True,
                "kill_intervals": True,
                "message": voice_resp,
                "stop_sequence": get_global_stop_sequence(),
            })

        elif action == "stop_voice":
            stop_voice_output()
            response.update({"action": "stop_voice", "message": voice_resp})

        elif action == "resume_voice":
            resume_voice_output()
            response.update({"action": "resume_voice", "message": voice_resp})

        elif action == "camera_start":
            set_global_stop(False)
            response.update({"action": "start_camera", "message": voice_resp})

        elif action == "camera_stop":
            response.update({"action": "stop_camera", "message": voice_resp, "kill_camera": True})

        elif action == "assistant_mode":
            set_assistant_active(True)
            release_mode(get_active_mode())
            response.update({"action": "start_assistant", "message": voice_resp, "assistant_active": True})

        elif action == "memory_remember":
            response.update({
                "action": "start_memory_remember",
                "message": voice_resp,
                "prompt_for_description": True,
                "auto_capture_camera": True,
                "countdown_needed": True,
            })

        elif action == "memory_find":
            if ext_query:
                mem_result = find_memory_object(ext_query, [])
                msg = mem_result["message"]
                response.update({
                    "action": "memory_find_result",
                    "query": ext_query,
                    "found": mem_result.get("found_in_memory", False),
                    "message": msg,
                    "voice_response": msg,
                })
            else:
                msg = "What would you like me to find? Please say the object name."
                response.update({"action": "memory_find_prompt", "message": msg, "voice_response": msg})

        elif action == "memory_list":
            memories = list_all_memories()
            if memories:
                names_str = ", ".join([f"{m['name']} at {m['location']}" for m in memories])
                msg = f"Dear friend, I remember {len(memories)} item{'s' if len(memories) > 1 else ''}. {names_str}."
            else:
                msg = "I don't have any memories saved yet. To save something, say My Eye remember this."
            response.update({"action": "memory_list", "message": msg, "voice_response": msg})

        elif action in ("food_detection", "food_live"):
            lbl = "Food Detector Live" if (is_live or action == "food_live") else "Food Detector"
            md = "food_live" if (is_live or action == "food_live") else "food"
            mode_result = set_active_mode(md, lbl)
            if mode_result["success"]:
                response.update({"action": f"start_{md}", "message": voice_resp})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action in ("stairs_detection", "stairs_live"):
            lbl = "Stair Detector Live" if (is_live or action == "stairs_live") else "Stair Detector"
            md = "stairs_live" if (is_live or action == "stairs_live") else "stairs"
            mode_result = set_active_mode(md, lbl)
            if mode_result["success"]:
                response.update({"action": f"start_{md}", "message": voice_resp})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action in ("traffic_detection", "traffic_live"):
            lbl = "Traffic Light Live" if (is_live or action == "traffic_live") else "Traffic Light Detector"
            md = "traffic_live" if (is_live or action == "traffic_live") else "traffic"
            mode_result = set_active_mode(md, lbl)
            if mode_result["success"]:
                response.update({"action": f"start_{md}", "message": voice_resp})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action == "page_reader":
            mode_result = set_active_mode("page_reader", "Page Reader")
            if mode_result["success"]:
                response.update({"action": "start_page_reader", "message": voice_resp})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action == "rag_scanner":
            mode_result = set_active_mode("rag_scanner", "Document Scanner")
            if mode_result["success"]:
                response.update({"action": "start_rag_scanner", "message": voice_resp})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action == "face_capture":
            if ext_name:
                response.update({
                    "action": "start_face_capture_with_name",
                    "name": ext_name,
                    "extracted_name": ext_name,
                    "capture_with_name": True,
                    "countdown_needed": True,
                    "message": voice_resp,
                })
            else:
                mode_result = set_active_mode("face_capture", "Face Capture")
                if mode_result["success"]:
                    response.update({
                        "action": "start_face_capture",
                        "message": voice_resp,
                        "prompt_for_name": True,
                        "countdown_needed": True,
                    })
                else:
                    msg = mode_result.get("message", get_voice_error("mode_busy"))
                    response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action == "face_recognize":
            response.update({"action": "recognize_face", "message": voice_resp})

        elif action == "money_detect":
            response.update({"action": "detect_money", "message": voice_resp})

        elif action == "describe_scene":
            response.update({"action": "describe_scene", "message": voice_resp})

        elif action == "object_find":
            mode_result = set_active_mode("object_finder", "Object Finder")
            if mode_result["success"]:
                query = ext_query or ""
                if query:
                    mem_entry = find_memory_by_name(query)
                    if mem_entry:
                        mem_msg = f"I remember! You keep your {mem_entry['name']} at {mem_entry['location']}. Go there first."
                        response.update({
                            "action": "memory_find_then_scan",
                            "query": query,
                            "memory_found": True,
                            "memory_location": mem_entry["location"],
                            "message": mem_msg,
                            "voice_response": mem_msg,
                        })
                    else:
                        response.update({"action": "start_object_finder", "query": query, "message": voice_resp})
                else:
                    response.update({"action": "start_object_finder_prompt", "query": "", "message": voice_resp, "prompt_for_object": True})
            else:
                msg = mode_result.get("message", get_voice_error("mode_busy"))
                response.update({"action": "mode_conflict", "message": msg, "voice_response": msg})

        elif action == "navigate_outdoor":
            dest = ext_dest or ""
            if dest:
                response.update({"action": "navigate", "destination": dest, "message": voice_resp})
            else:
                msg = "Where would you like to go? Please say the destination name clearly."
                response.update({"action": "navigate_prompt", "message": msg, "voice_response": msg})

        elif action == "navigate_indoor":
            dest = ext_dest or ext_query or ""
            response.update({"action": "indoor_navigate", "destination": dest, "message": voice_resp})

        elif action == "sos":
            response.update({"action": "trigger_sos", "message": voice_resp})

        elif action == "get_location":
            response.update({"action": "get_location", "message": voice_resp})

        else:
            answer = ask_rag_assistant(cleaned)
            response.update({"action": "answer", "answer": answer, "message": answer, "voice_response": answer})

        final_msg = response.get("voice_response") or response.get("message") or ""
        if final_msg:
            log_conversation("SYSTEM", final_msg[:300])
        return jsonify(response)

    except Exception as e:
        print(f"Command processing error: {traceback.format_exc()}")
        err_msg = "Dear friend, I had a problem processing that request. " + get_voice_error("network")
        return jsonify({"error": str(e), "message": err_msg, "voice_response": err_msg}), 500


@app.route("/api/assistant/start", methods=["POST"])
def assistant_start():
    set_assistant_active(True)
    msg = (
        "Hello dear friend! I am your VisionAssist assistant and I am fully awake and ready. "
        "I can help you with everything: finding your things, remembering where you keep objects, "
        "recognizing people, reading documents, checking traffic lights, or answering any question. "
        "Just say My Eye followed by what you need. What would you like me to do?"
    )
    log_conversation("SYSTEM", msg)
    return jsonify({"success": True, "message": msg, "voice_response": msg, "assistant_active": True})


@app.route("/api/assistant/stop", methods=["POST"])
def assistant_stop():
    set_assistant_active(False)
    msg = "Assistant mode has ended. I am still here whenever you need me. Just say My Eye start assistant to begin again."
    log_conversation("SYSTEM", msg)
    return jsonify({"success": True, "message": msg, "voice_response": msg, "assistant_active": False})


@app.route("/api/assistant/query", methods=["POST"])
def assistant_query_route():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()
        frames = data.get("frames", [])

        if not query:
            msg = "I didn't hear your request clearly. Please say My Eye and then tell me what you need."
            return jsonify({"success": False, "message": msg, "voice_response": msg, "action_taken": "no_query"})

        log_conversation("USER", f"[ASSISTANT] {query}")
        processed_frames = []
        if frames:
            processed_frames = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]

        t = query.lower()

        if any(w in t for w in ["where is my", "do you remember", "kahan hai", "find my", "dhundo"]):
            subject = query
            for prefix in ["where is my", "do you remember my", "find my", "kahan hai mera", "kahan hai"]:
                if prefix in t:
                    subject = t.split(prefix)[-1].strip()
                    break
            subject = re.sub(r"\b(mera|meri|my|the)\b", "", subject).strip()
            mem_result = find_memory_object(subject, processed_frames)
            msg = mem_result["message"]
            log_conversation("SYSTEM", msg)
            return jsonify({
                "success": True,
                "action_taken": "memory_find",
                "message": msg,
                "voice_response": msg,
                "needs_camera": False,
                "found": mem_result.get("found_in_memory", False),
                "extra": {},
            })

        if any(w in t for w in ["who is", "kaun hai", "pehchano", "recognize"]):
            if not processed_frames:
                msg = "Please make sure the camera is on and pointing at the person. Then ask me again."
                return jsonify({"success": True, "action_taken": "face_recognize", "message": msg, "voice_response": msg, "needs_camera": True, "extra": {}})
            if DEEPFACE_AVAILABLE and CV2_AVAILABLE:
                try:
                    img_bytes = base64.b64decode(processed_frames[0])
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    known_images = [f for f in os.listdir("known_faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if not known_images:
                        msg = get_voice_error("no_faces_saved")
                    else:
                        dfs = DeepFace.find(img_path=frame, db_path="known_faces", model_name="Facenet", enforce_detection=False, silent=True)
                        if dfs and len(dfs) > 0 and not dfs[0].empty:
                            match_path = dfs[0].iloc[0]["identity"]
                            distance = float(dfs[0].iloc[0].get("distance", 1.0))
                            filename = os.path.basename(match_path)
                            base_name = os.path.splitext(filename)[0].replace("_", " ").title()
                            name = "".join([c for c in base_name if not c.isdigit()]).strip()
                            if distance < 0.6:
                                msg = f"I recognize this person! This is {name}."
                            else:
                                msg = "I see a person but I do not recognize them from my saved faces."
                        else:
                            msg = get_voice_error("no_face")
                except Exception as e:
                    msg = "Face recognition had a problem. Please try again."
            else:
                msg = "Face recognition needs DeepFace installed."
            log_conversation("SYSTEM", msg)
            return jsonify({"success": True, "action_taken": "face_recognize", "message": msg, "voice_response": msg, "needs_camera": False, "extra": {}})

        if any(w in t for w in ["what is in front", "what do you see", "describe", "kya hai", "yahan kya", "scene"]):
            if not processed_frames:
                msg = "I need the camera to describe the scene. Please make sure the camera is on."
                return jsonify({"success": True, "action_taken": "scene_describe", "message": msg, "voice_response": msg, "needs_camera": True, "extra": {}})
            try:
                result = groq_vision_call(
                    "You are the eyes of a blind person. Describe scenes clearly and helpfully. No markdown. Natural speech.",
                    f"What do you see? User asked: {query}. Describe clearly for a blind person.",
                    processed_frames,
                    max_tokens=300,
                    priority="high",
                )
                log_conversation("SYSTEM", result)
                return jsonify({"success": True, "action_taken": "scene_describe", "message": result, "voice_response": result, "needs_camera": False, "extra": {}})
            except Exception:
                msg = get_voice_error("api_busy")
                return jsonify({"success": True, "action_taken": "scene_describe", "message": msg, "voice_response": msg, "needs_camera": False, "extra": {}})

        answer = ask_groq_general(query)
        log_conversation("SYSTEM", answer)
        return jsonify({
            "success": True,
            "action_taken": "general_answer",
            "message": answer,
            "voice_response": answer,
            "needs_camera": False,
            "extra": {},
        })

    except Exception as e:
        print(f"Assistant query error: {traceback.format_exc()}")
        err_msg = "Dear friend, I had a technical problem. " + get_voice_error("network")
        return jsonify({"success": False, "message": err_msg, "voice_response": err_msg}), 500


@app.route("/api/assistant/status", methods=["GET"])
def assistant_status():
    return jsonify({"assistant_active": get_assistant_active()})


@app.route("/api/memory/save", methods=["POST"])
def memory_save():
    try:
        data = request.get_json(silent=True) or {}
        description = data.get("description", "").strip()
        frames = data.get("frames", [])

        if not description:
            msg = "I need a description to save this. Please say what the object is and where you keep it."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        processed = [preprocess_image(f, 800) for f in frames[:2] if f and len(f) > 100]
        if not processed:
            msg = "The photos did not process correctly. Please make sure the camera is working and try again."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        log_conversation("USER", f"Remember this: {description}")
        result = save_memory_object(description, processed)

        if result["success"]:
            msg = result.get("voice_response", f"I have saved {result['name']} at {result['location']} in my memory.")
            log_conversation("SYSTEM", msg)
            return jsonify({"success": True, "message": msg, "voice_response": msg, "memory_id": result["id"], "name": result["name"], "location": result["location"]})
        else:
            msg = result.get("error", "Memory save failed. Please try again.")
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 500

    except Exception as e:
        print(f"Memory save error: {traceback.format_exc()}")
        msg = "Something went wrong while saving. " + get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/find", methods=["POST"])
def memory_find():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()
        frames = data.get("frames", [])

        if not query:
            msg = "Please tell me what you are looking for."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        log_conversation("USER", f"Looking for: {query}")
        processed_frames = []
        if frames:
            processed_frames = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]

        result = find_memory_object(query, processed_frames)
        msg = result["message"]
        log_conversation("SYSTEM", msg)

        return jsonify({
            "success": True,
            "found_in_memory": result.get("found_in_memory", False),
            "confirmed_in_scene": result.get("confirmed_in_scene", False),
            "name": result.get("name", query),
            "location": result.get("location", ""),
            "message": msg,
            "voice_response": msg,
        })

    except Exception as e:
        print(f"Memory find error: {traceback.format_exc()}")
        msg = "I had a problem searching. " + get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/verify-location", methods=["POST"])
def memory_verify_location():
    try:
        data = request.get_json(silent=True) or {}
        query = data.get("query", "").strip()
        frames = data.get("frames", [])

        if not query or not frames:
            msg = "I need the camera to verify the location."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        log_conversation("USER", f"I am at the location, looking for: {query}")
        memory_entry = find_memory_by_name(query)
        processed = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]

        if not processed:
            msg = get_voice_error("no_camera")
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 400

        if memory_entry and memory_entry.get("photos"):
            confirmed_text = _try_visual_confirm_memory(memory_entry, processed)
            if confirmed_text:
                log_conversation("SYSTEM", confirmed_text)
                return jsonify({"success": True, "found": True, "message": confirmed_text, "voice_response": confirmed_text})

        result = find_object_in_frames(processed, query)
        msg = result.get("result", get_voice_error("no_query"))
        log_conversation("SYSTEM", msg)
        return jsonify({"success": result["success"], "found": result["success"], "message": msg, "voice_response": msg})

    except Exception as e:
        print(f"Memory verify error: {traceback.format_exc()}")
        msg = get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/list", methods=["GET"])
def memory_list():
    try:
        index = list_all_memories()
        readable = []
        for entry in index:
            readable.append({
                "id": entry["id"],
                "name": entry["name"],
                "location": entry["location"],
                "color": entry.get("color", ""),
                "description": entry.get("description", ""),
                "timestamp": entry.get("timestamp", ""),
                "times_found": entry.get("times_found", 0),
                "photos": len(entry.get("photos", [])),
            })
        if readable:
            names_parts = [f"{e['color'] + ' ' if e['color'] else ''}{e['name']} at {e['location']}" for e in readable]
            names_str = ", ".join(names_parts)
            tts_message = f"I remember {len(readable)} item{'s' if len(readable) > 1 else ''}. Here they are: {names_str}. Just say My Eye where is my and the object name to find any of them."
        else:
            tts_message = "I don't have any memories saved yet. To save something, say My Eye remember this and describe the object and where you keep it."
        return jsonify({"success": True, "memories": readable, "count": len(readable), "tts_message": tts_message, "voice_response": tts_message})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/delete", methods=["POST"])
def memory_delete():
    try:
        data = request.get_json(silent=True) or {}
        mem_id = data.get("id", "").strip()
        name = data.get("name", "").strip()
        if mem_id:
            deleted = delete_memory(mem_id)
            if deleted:
                msg = "Memory deleted successfully."
                return jsonify({"success": True, "message": msg, "voice_response": msg})
            msg = "I could not find that memory to delete."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 404
        if name:
            entry = find_memory_by_name(name)
            if entry:
                deleted = delete_memory(entry["id"])
                if deleted:
                    msg = f"I have forgotten {name}. It has been removed from my memory."
                    log_conversation("SYSTEM", msg)
                    return jsonify({"success": True, "message": msg, "voice_response": msg})
            msg = f"I don't have {name} in my memory."
            return jsonify({"success": False, "message": msg, "voice_response": msg}), 404
        msg = "Please provide the name of what you want me to forget."
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 400
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/clear-all", methods=["POST"])
def memory_clear_all():
    try:
        index = load_memory_index()
        for entry in index:
            for photo in entry.get("photos", []):
                photo_path = os.path.join(MEMORY_DIR, photo)
                if os.path.exists(photo_path):
                    try:
                        os.remove(photo_path)
                    except Exception:
                        pass
        save_memory_index([])
        log_conversation("SYSTEM", "All memories cleared")
        msg = "I have cleared all my memories. Starting fresh."
        return jsonify({"success": True, "message": msg, "voice_response": msg})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "message": msg, "voice_response": msg}), 500


@app.route("/api/memory/start-remember", methods=["POST"])
def memory_start_remember():
    msg = (
        "Of course! Please describe the object and where you keep it. "
        "For example, say: my red keys are on the kitchen table. "
        "Make sure the camera is pointing at the object. "
        "I will count down from 3 and then automatically take 2 photos."
    )
    log_conversation("SYSTEM", msg)
    return jsonify({
        "success": True,
        "message": msg,
        "voice_response": msg,
        "prompt_for_description": True,
        "auto_capture_camera": True,
        "countdown_needed": True,
        "active_mode": "memory_remember",
    })


@app.route("/api/conversation/log", methods=["POST"])
def conversation_log_route():
    try:
        data = request.get_json(silent=True) or {}
        speaker = data.get("speaker", "USER").upper()
        text = data.get("text", "").strip()
        if text:
            log_conversation(speaker, text)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/conversation/context", methods=["GET"])
def conversation_context_route():
    try:
        ctx = get_conversation_context()
        return jsonify({"success": True, "context": ctx, "lines": len(ctx.splitlines())})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/conversation/clear", methods=["POST"])
def conversation_clear():
    try:
        with _conv_log_lock:
            with open(CONVERSATION_LOG, "w", encoding="utf-8") as f:
                f.write("")
        msg = "Conversation history cleared."
        return jsonify({"success": True, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/face/start-capture", methods=["POST"])
def face_start_capture():
    result = set_active_mode("face_capture", "Face Capture")
    if not result["success"]:
        msg = result.get("message", get_voice_error("mode_busy"))
        return jsonify({"success": False, "message": msg, "voice_response": msg, "conflict": result.get("conflict")})
    msg = "Face capture is ready. Please make sure the person is looking directly at the camera with good lighting."
    return jsonify({"success": True, "message": msg, "voice_response": msg, "prompt_for_name": True, "active_mode": "face_capture", "countdown_needed": True})


@app.route("/api/face/set-name-listening", methods=["POST"])
def face_set_name_listening():
    return jsonify({"success": True, "listening_for_name": True})


@app.route("/api/capture-face", methods=["POST"])
def capture_face_api():
    if not CV2_AVAILABLE:
        msg = "Camera processing is not available. Please install OpenCV: pip install opencv-python"
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        name = data.get("name", "").strip()
        for noise in ["say the name", "the name is", "name is", "my name is", "save as", "naam se"]:
            if name.lower().startswith(noise):
                name = name[len(noise):].strip()
        if not name:
            msg = "I did not get the person's name. Please say their name clearly."
            return jsonify({"status": "need_name", "message": msg, "voice_response": msg}), 400
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            msg = get_voice_error("no_camera")
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 400
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(faces) == 0:
            msg = get_voice_error("no_face")
            return jsonify({"status": "no_face", "message": msg, "voice_response": msg}), 200
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
        safe_name = name.lower().replace(" ", "_")
        save_path = os.path.join("known_faces", f"{safe_name}.jpg")
        cv2.imwrite(save_path, frame)
        for pkl_file in os.listdir("known_faces"):
            if pkl_file.endswith(".pkl"):
                try:
                    os.remove(os.path.join("known_faces", pkl_file))
                except Exception:
                    pass
        release_mode("face_capture")
        msg = f"Face captured and saved successfully! I now know {name}. Next time say My Eye who is this and I will recognize them."
        log_conversation("SYSTEM", msg)
        return jsonify({"status": "success", "message": msg, "voice_response": msg, "name": name})
    except Exception as e:
        print(f"Capture face error: {e}")
        release_mode("face_capture")
        msg = "There was a problem saving the face. " + get_voice_error("network")
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500


@app.route("/api/recognize-face", methods=["POST"])
def recognize_face():
    if not DEEPFACE_AVAILABLE:
        msg = "Face recognition requires DeepFace. Please install it: pip install deepface"
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500
    if not CV2_AVAILABLE:
        msg = "Camera processing not available. Please install OpenCV: pip install opencv-python"
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            msg = get_voice_error("no_camera")
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 400
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
        known_images = [f for f in os.listdir("known_faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not known_images:
            msg = get_voice_error("no_faces_saved")
            return jsonify({"status": "success", "audioDescription": msg, "voice_response": msg})
        dfs = DeepFace.find(img_path=frame, db_path="known_faces", model_name="Facenet", enforce_detection=False, silent=True)
        recognized_faces = []
        if dfs and len(dfs) > 0 and not dfs[0].empty:
            match_path = dfs[0].iloc[0]["identity"]
            distance = float(dfs[0].iloc[0].get("distance", 1.0))
            filename = os.path.basename(match_path)
            base_name = os.path.splitext(filename)[0].replace("_", " ").title()
            recognized_name = "".join([c for c in base_name if not c.isdigit()]).strip()
            if distance < 0.6:
                description = f"I recognize this person! This is {recognized_name}."
                recognized_faces.append({"name": recognized_name, "distance": round(distance, 2)})
            elif distance < 0.75:
                description = f"This might be {recognized_name} but I am not very sure. The lighting or angle may not be ideal."
            else:
                description = "I see a person here, but I do not recognize them from my saved faces."
        else:
            description = get_voice_error("no_face")
        log_conversation("SYSTEM", description)
        return jsonify({"status": "success", "audioDescription": description, "voice_response": description, "recognizedFaces": recognized_faces})
    except Exception as e:
        print(f"Face recognition error: {traceback.format_exc()}")
        msg = "Face recognition had a problem. Please make sure the face is clearly visible and well lit."
        return jsonify({"status": "error", "error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/detect-money", methods=["POST"])
def detect_money_api():
    if sift is None or bf is None:
        msg = "Currency detection requires OpenCV with SIFT. Please install OpenCV."
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if not os.path.exists("Currency"):
            msg = "Currency reference images not found. Please add currency images to the Currency folder."
            return jsonify({"status": "success", "audioDescription": msg, "voice_response": msg})
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)
        best_match_name = None
        max_good_matches = 0
        if des_frame is not None:
            for filename in os.listdir("Currency"):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    ref_path = os.path.join("Currency", filename)
                    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                    if ref_img is None:
                        continue
                    kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
                    if des_ref is None:
                        continue
                    matches = bf.knnMatch(des_ref, des_frame, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.70 * n.distance]
                    if len(good_matches) > max_good_matches and len(good_matches) > 35:
                        max_good_matches = len(good_matches)
                        name = os.path.splitext(filename)[0].split("_")[0]
                        best_match_name = f"{name} Rupees" if name.isdigit() else name
        if max_good_matches > 35 and best_match_name:
            description = f"I can see a {best_match_name} note in your hand."
        else:
            description = "I cannot clearly identify the currency note. Please hold the note flat with the numbers visible and try again."
        log_conversation("SYSTEM", description)
        return jsonify({"status": "success", "audioDescription": description, "voice_response": description, "currency": best_match_name})
    except Exception as e:
        msg = "Currency detection had a problem. " + get_voice_error("network")
        return jsonify({"status": "error", "error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "audioDescription": get_voice_error("no_camera")})
        detected = detect_objects(frame)
        description = describe_objects_simple(detected)
        return jsonify({"status": "success", "detectedObjects": detected, "audioDescription": description, "voice_response": description})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"status": "error", "error": str(e), "audioDescription": msg, "voice_response": msg}), 500


@app.route("/api/sonar-data", methods=["GET"])
def get_sonar_data():
    return jsonify({"connected": arduino_connected, "port": arduino_port, "distance": front_distance, "steps": steps})


@app.route("/api/detection-mode", methods=["GET", "POST"])
def handle_detection_mode():
    global detection_mode
    if request.method == "GET":
        return jsonify({"mode": detection_mode})
    data = request.get_json()
    detection_mode = data.get("mode", "normal")
    return jsonify({"status": "success", "mode": detection_mode})


@app.route("/api/arduino/status", methods=["GET"])
def arduino_status():
    return jsonify({"connected": arduino_connected, "port": arduino_port, "distance": front_distance, "steps": steps})


@app.route("/api/arduino/reconnect", methods=["POST"])
def reconnect_arduino():
    global arduino, arduino_connected
    if arduino and hasattr(arduino, "is_open") and arduino.is_open:
        arduino.close()
    success = init_arduino()
    return jsonify({"success": success, "connected": arduino_connected, "port": arduino_port})


@app.route("/api/geocode", methods=["POST"])
def geocode():
    try:
        data = request.get_json()
        address = data.get("address", "")
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "VisionAssist/1.0"},
            timeout=10,
        )
        jdata = resp.json()
        if jdata:
            return jsonify({"lat": float(jdata[0]["lat"]), "lon": float(jdata[0]["lon"]), "display_name": jdata[0]["display_name"]})
        msg = f"I could not find the location {address}. Please try saying the address more specifically."
        return jsonify({"error": "Location not found", "message": msg, "voice_response": msg}), 404
    except Exception as e:
        msg = "Location search failed. " + get_voice_error("network")
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/reverse-geocode", methods=["POST"])
def reverse_geocode():
    try:
        data = request.get_json()
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": data.get("lat"), "lon": data.get("lon"), "format": "json"},
            headers={"User-Agent": "VisionAssist/1.0"},
            timeout=10,
        )
        jdata = resp.json()
        return jsonify({"address": jdata.get("display_name", "Unknown location")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sos/trigger", methods=["POST"])
def trigger_sos():
    global sos_active, sos_timer
    try:
        data = request.get_json()
        location = data.get("location", "Unknown")
        lat = data.get("lat")
        lon = data.get("lon")
        sos_active = True
        maps_link = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "Location not available"
        sms_sent = send_sms_alert(location, maps_link)
        email_sent = send_email_alert(location, maps_link)
        log_conversation("SYSTEM", f"SOS triggered at {location}")
        sos_timer = threading.Timer(300.0, lambda: globals().update(sos_active=False))
        sos_timer.start()
        msg = "SOS activated! I am sending emergency alerts to your contacts right now. Please stay calm and stay where you are."
        return jsonify({"status": "success", "sos_active": True, "message": msg, "voice_response": msg, "alerts": {"sms": sms_sent, "email": email_sent}})
    except Exception as e:
        msg = "There was a problem activating SOS. Please call emergency services directly."
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/sos/cancel", methods=["POST"])
def cancel_sos():
    global sos_active, sos_timer
    sos_active = False
    if sos_timer:
        sos_timer.cancel()
        sos_timer = None
    msg = "SOS has been cancelled. I hope everything is okay."
    return jsonify({"status": "success", "sos_active": False, "message": msg, "voice_response": msg})


@app.route("/api/sos/contacts", methods=["GET", "POST"])
def manage_contacts():
    global emergency_contacts
    if request.method == "GET":
        return jsonify({"contacts": emergency_contacts})
    try:
        data = request.get_json()
        emergency_contacts = data.get("contacts", emergency_contacts)
        with open("emergency_contacts.json", "w") as f:
            json.dump(emergency_contacts, f)
        return jsonify({"status": "success", "contacts": emergency_contacts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect-document-frame", methods=["POST"])
def detect_document_frame():
    global rag_detection_counter, read_page_detection_counter
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        mode = data.get("mode", "rag")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid frame"}), 400
        if yolo_model is None:
            if mode == "rag":
                rag_detection_counter += 1
                return jsonify({"status": "detected", "objects": [{"label": "document", "confidence": 0.5}], "counter": rag_detection_counter})
            else:
                read_page_detection_counter += 1
                return jsonify({"status": "detected", "objects": [{"label": "page", "confidence": 0.5}], "counter": read_page_detection_counter})
        results = yolo_model(frame, conf=0.15, verbose=False)
        detected_objects = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                confidence = float(box.conf[0])
                is_doc = label.lower() in DOCUMENT_CLASSES or cls_id in YOLO_DOCUMENT_IDS
                if is_doc and confidence > 0.2:
                    detected_objects.append({"label": label, "confidence": confidence})
        if detected_objects:
            if mode == "rag":
                rag_detection_counter += 1
            else:
                read_page_detection_counter += 1
            counter = rag_detection_counter if mode == "rag" else read_page_detection_counter
            return jsonify({"status": "detected", "objects": detected_objects, "counter": counter})
        else:
            if mode == "rag":
                rag_detection_counter = 0
            else:
                read_page_detection_counter = 0
            return jsonify({"status": "not_detected", "counter": 0})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/auto-capture-rag", methods=["POST"])
def auto_capture_rag():
    global rag_detection_counter, last_rag_capture_time
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        if PIL_AVAILABLE:
            image_pil = Image.open(io.BytesIO(image_bytes))
            has_text, text_found = check_text_in_image(image_pil)
            if not has_text:
                msg = "I detected a document but could not read any text. Please make sure the text is clearly visible and try again."
                return jsonify({"status": "no_text", "message": msg, "voice_response": msg})
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"status": "error", "message": "Invalid frame"})
        _, img_encoded = cv2.imencode(".jpg", frame)
        img_bytes_out = img_encoded.tobytes()
        try:
            files = {"file": ("rag_document.jpg", img_bytes_out, "image/jpeg")}
            requests.post(N8N_RAG_IMAGE_URL, files=files, timeout=30)
        except Exception as upload_err:
            print(f"n8n upload error: {upload_err}")
        last_rag_capture_time = time.time()
        rag_detection_counter = 0
        msg = "Document captured and saved to the knowledge base! You can now ask me questions about this document."
        return jsonify({"status": "success", "message": msg, "voice_response": msg})
    except Exception as e:
        msg = "Document capture had a problem. " + get_voice_error("network")
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500


@app.route("/api/page-reader/analyze", methods=["POST"])
def page_reader_analyze():
    global read_page_detection_counter, last_read_capture_time
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames", [])
        if not frames:
            image_data = data.get("image_data", "")
            if image_data:
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                frames = [image_data]
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 400
        processed = [preprocess_image(f, 900) for f in frames[:3] if f and len(f) > 100]
        if not processed:
            msg = "The camera frames could not be processed. Please make sure the camera is working."
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 400
        try:
            description = groq_vision_call(PAGE_READER_SYSTEM, PAGE_READER_USER, processed, max_tokens=1500, priority="high")
        except RuntimeError as e:
            msg = get_voice_error("api_busy")
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 503
        last_read_capture_time = time.time()
        read_page_detection_counter = 0
        log_conversation("SYSTEM", f"Page read: {description[:100]}...")
        return jsonify({"status": "success", "description": description, "message": description, "voice_response": description, "frames_used": len(processed)})
    except Exception as e:
        msg = "Page reading had a problem. " + get_voice_error("network")
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500


@app.route("/api/auto-capture-read-page", methods=["POST"])
def auto_capture_read_page():
    global read_page_detection_counter, last_read_capture_time
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        processed = preprocess_image(image_data, 900)
        try:
            description = groq_vision_call(PAGE_READER_SYSTEM, PAGE_READER_USER, [processed], max_tokens=1200, priority="high")
        except RuntimeError as e:
            msg = get_voice_error("api_busy")
            return jsonify({"status": "error", "message": msg, "voice_response": msg}), 503
        last_read_capture_time = time.time()
        read_page_detection_counter = 0
        return jsonify({"status": "success", "description": description, "message": description, "voice_response": description})
    except Exception as e:
        msg = "Page reading had a problem. " + get_voice_error("network")
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500


@app.route("/api/rag-upload", methods=["POST"])
def rag_upload():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file received"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400
        try:
            resp = requests.post(N8N_FORM_URL, files={"file": (file.filename, file.stream, file.mimetype)}, timeout=30)
            if resp.status_code == 200:
                msg = f"Document uploaded to knowledge base successfully!"
                return jsonify({"status": "success", "message": msg, "voice_response": msg})
            msg = "File received but upload had an issue. Please try again."
            return jsonify({"status": "success", "message": msg, "voice_response": msg})
        except Exception:
            msg = "File processed."
            return jsonify({"status": "success", "message": msg, "voice_response": msg})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"status": "error", "message": msg, "voice_response": msg}), 500


@app.route("/api/food", methods=["POST"])
def api_food():
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames", [])
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        processed = [preprocess_image(f, 800) for f in frames if f and len(f) > 100]
        if not processed:
            msg = "Camera frames could not be processed. Please try again."
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        result = groq_vision_call(FOOD_SYSTEM, FOOD_USER, processed, max_tokens=250, priority="low")
        log_conversation("SYSTEM", f"Food: {result}")
        return jsonify({"success": True, "result": result, "voice_response": result})
    except RuntimeError as e:
        msg = get_voice_error("api_busy")
        return jsonify({"success": False, "result": msg, "voice_response": msg})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "result": msg, "voice_response": msg})


@app.route("/api/stairs", methods=["POST"])
def api_stairs():
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames", [])
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        processed = [preprocess_image(f, 900) for f in frames if f and len(f) > 100]
        if not processed:
            msg = "Camera frames could not be processed. Please try again."
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        result = groq_vision_call(STAIR_SYSTEM, STAIR_USER, processed, max_tokens=300, priority="low")
        log_conversation("SYSTEM", f"Stairs: {result}")
        return jsonify({"success": True, "result": result, "voice_response": result})
    except RuntimeError as e:
        msg = get_voice_error("api_busy")
        return jsonify({"success": False, "result": msg, "voice_response": msg})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "result": msg, "voice_response": msg})


def _fast_hsv_color(frame_b64: str):
    """Returns (color, confidence_score) using HSV only, no YOLO, very fast."""
    if not CV2_AVAILABLE:
        return "UNKNOWN", 0
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return "UNKNOWN", 0
        h, w = frame.shape[:2]
        # Focus only on top-half of image where traffic lights usually are
        roi = frame[0:h//2, w//4:3*w//4]
        hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255]))
        r2 = cv2.inRange(hsv, np.array([165, 150, 100]), np.array([180, 255, 255]))
        red_mask = cv2.add(r1, r2)
        green_mask = cv2.inRange(hsv, np.array([40, 100, 80]), np.array([90, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 120, 100]), np.array([35, 255, 255]))
        scores = {
            "RED": int(cv2.countNonZero(red_mask)),
            "GREEN": int(cv2.countNonZero(green_mask)),
            "YELLOW": int(cv2.countNonZero(yellow_mask)),
        }
        best = max(scores, key=scores.get)
        best_score = scores[best]
        if best_score < 150:
            return "UNKNOWN", best_score
        return best, best_score
    except Exception as e:
        print(f"Fast HSV error: {e}")
        return "UNKNOWN", 0


def _groq_traffic_fast(frame_b64: str) -> str:
    """Single frame, minimal tokens, fastest possible Groq call."""
    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": TRAFFIC_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRAFFIC_USER},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 20,
            "temperature": 0.0,
            "stream": False,
        }
        result = groq_request(payload, priority="high", max_retries=1)
        if result:
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Groq traffic fast error: {e}")
    return ""


@app.route("/api/traffic", methods=["POST"])
def api_traffic():
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames", [])
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400

        raw_frame = frames[0]
        if not raw_frame or len(raw_frame) < 100:
            msg = "Camera frame empty. Please try again."
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400

        # Shrink image to 320x240 for speed
        small_frame = raw_frame
        if PIL_AVAILABLE:
            try:
                img_bytes = base64.b64decode(raw_frame)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((320, 240), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                small_frame = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                small_frame = raw_frame

        # Step 1: instant HSV (no API)
        hsv_color, hsv_score = _fast_hsv_color(small_frame)
        traffic_result = ""
        method_used = ""

        if hsv_color == "RED" and hsv_score >= 150:
            traffic_result = "Red light. Safe to cross now."
            method_used = "HSV"
        elif hsv_color == "GREEN" and hsv_score >= 150:
            traffic_result = "Green light. Stop, cars are moving."
            method_used = "HSV"
        elif hsv_color == "YELLOW" and hsv_score >= 150:
            traffic_result = "Yellow light. Wait, do not cross."
            method_used = "HSV"
        else:
            # Step 2: HSV not confident — single-frame Groq with low detail
            groq_result = _groq_traffic_fast(small_frame)
            if groq_result:
                gl = groq_result.lower()
                if "red" in gl:
                    traffic_result = "Red light. Safe to cross now."
                elif "green" in gl:
                    traffic_result = "Green light. Stop, cars are moving."
                elif "yellow" in gl:
                    traffic_result = "Yellow light. Wait, do not cross."
                else:
                    traffic_result = groq_result
                method_used = "Groq"
            else:
                traffic_result = "Cannot see traffic light. Be very careful."
                method_used = "fallback"

        log_conversation("SYSTEM", f"Traffic [{method_used}] score={hsv_score}: {traffic_result}")
        return jsonify({
            "success": True,
            "result": traffic_result,
            "voice_response": traffic_result,
            "method": method_used,
            "hsv_score": hsv_score,
        })
    except Exception as e:
        print(f"Traffic error: {traceback.format_exc()}")
        msg = "Traffic detection error. Please try again."
        return jsonify({"success": False, "result": msg, "voice_response": msg})


@app.route("/api/whatshere", methods=["POST"])
def api_whatshere():
    try:
        data = request.get_json(silent=True) or {}
        frames = data.get("frames", [])
        question = (data.get("question") or "").strip()
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        if not question:
            question = "What do you see in front of me? Please describe the scene for a blind person."
        processed = [preprocess_image(f, 900) for f in frames if f and len(f) > 100]
        if not processed:
            msg = "Camera frames could not be processed."
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        conv_ctx = get_conversation_context()
        user_prompt = (
            f'The blind user is asking: "{question}"\n'
            f"{'Recent context: ' + conv_ctx[-500:] if conv_ctx else ''}\n"
            "Look carefully at all the provided frames and answer their question. "
            "Be specific about positions (left, right, center, near, far). "
            "Address them warmly. 2 to 4 sentences max."
        )
        result = groq_vision_call(
            "You are the eyes of a blind person. Describe scenes clearly, warmly, and helpfully. No markdown. Natural speech.",
            user_prompt,
            processed,
            max_tokens=300,
            priority="high",
        )
        log_conversation("USER", question)
        log_conversation("SYSTEM", result)
        return jsonify({"success": True, "result": result, "voice_response": result})
    except RuntimeError as e:
        msg = get_voice_error("api_busy")
        return jsonify({"success": False, "result": msg, "voice_response": msg})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"success": False, "result": msg, "voice_response": msg})


@app.route("/api/whatsfront", methods=["POST"])
def api_whatsfront():
    return api_whatshere()


@app.route("/api/find-object", methods=["POST"])
def find_object_api():
    try:
        data = request.get_json(force=True)
        if not data:
            msg = "No data received. Please try again."
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        query = data.get("query", "").strip()
        frames = data.get("frames", [])
        if not query:
            msg = get_voice_error("no_query")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        if not frames:
            msg = get_voice_error("camera_needed")
            return jsonify({"success": False, "result": msg, "voice_response": msg}), 400
        log_conversation("USER", f"Find object: {query}")
        result = find_object_in_frames(frames, query)
        if result.get("result"):
            log_conversation("SYSTEM", result["result"][:200])
        result["voice_response"] = result.get("result", "")
        return jsonify(result)
    except Exception as e:
        print(traceback.format_exc())
        msg = "Object finding had a problem. " + get_voice_error("network")
        return jsonify({"success": False, "result": msg, "voice_response": msg}), 500


@app.route("/api/indoor/upload-video", methods=["POST"])
def indoor_upload_video():
    global indoor_map_data, nav_system
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, file.filename)
        file.save(video_path)
        if not CV2_AVAILABLE:
            os.remove(video_path)
            os.rmdir(temp_dir)
            return jsonify({"error": "OpenCV not installed"}), 500
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30
        frame_interval = max(1, int(fps * 2))
        frames, frame_count = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                resized = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames.append(base64.b64encode(buffer).decode("utf-8"))
                if len(frames) >= 10:
                    break
            frame_count += 1
        cap.release()
        os.remove(video_path)
        os.rmdir(temp_dir)
        map_prompt = (
            "You are an expert indoor mapping AI. Analyze these video frames from a walkthrough of a building.\n"
            "Return ONLY a valid JSON object:\n"
            '{"rooms": [{"id": "unique_id", "name": "Room Name", "x": 100, "y": 150, "description": "Brief description"}],'
            '"paths": [{"from": "room_id_1", "to": "room_id_2", "distance": 15, "direction": "straight", "description": "Walk straight"}]}'
        )
        content = [{"type": "text", "text": map_prompt}]
        for frm in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frm}"}})
        payload = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0.3, "max_tokens": 2000}
        result = groq_request(payload, priority="high")
        if result:
            response_text = result["choices"][0]["message"]["content"]
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                try:
                    indoor_map_data = json.loads(json_match.group())
                except Exception:
                    indoor_map_data = create_structured_map_from_response(response_text, frames)
            else:
                indoor_map_data = create_structured_map_from_response(response_text, frames)
        else:
            indoor_map_data = create_structured_map_from_response("", frames)
        if not indoor_map_data.get("rooms"):
            indoor_map_data["rooms"] = [{"id": "room_1", "name": "Main Area", "x": 400, "y": 300, "description": "Main area"}]
        if not indoor_map_data.get("paths"):
            indoor_map_data["paths"] = []
        nav_system = IndoorNavigationSystem()
        for room in indoor_map_data.get("rooms", []):
            nav_system.add_node(Node(id=room["id"], name=room["name"], x=room.get("x", 0), y=room.get("y", 0), description=room.get("description", "")))
        for path in indoor_map_data.get("paths", []):
            nav_system.add_edge(Edge(from_node=path["from"], to_node=path["to"], distance=path["distance"], direction=path["direction"], description=path.get("description", "")))
        with open("indoor_map.json", "w") as f:
            json.dump(indoor_map_data, f, indent=2)
        msg = f"Indoor map created successfully with {len(indoor_map_data.get('rooms', []))} rooms."
        return jsonify({"status": "success", "message": msg, "voice_response": msg, "rooms": [{"id": r["id"], "name": r["name"]} for r in indoor_map_data.get("rooms", [])]})
    except Exception as e:
        print(f"Video upload error: {traceback.format_exc()}")
        msg = "Video upload had a problem. " + get_voice_error("network")
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/indoor/navigate", methods=["POST"])
def indoor_navigate():
    try:
        data = request.get_json()
        destination = data.get("destination", "").lower().strip()
        current_image = data.get("image_data", None)
        if not indoor_map_data:
            msg = "No indoor map is available yet. Please upload a walkthrough video of your home first."
            return jsonify({"error": msg, "voice_response": msg}), 400
        dest_room_id = None
        for room in indoor_map_data.get("rooms", []):
            if destination in room["name"].lower() or destination in room["id"].lower():
                dest_room_id = room["id"]
                break
        if not dest_room_id:
            room_names = ", ".join([r["name"] for r in indoor_map_data.get("rooms", [])])
            msg = f"I could not find {destination} in your indoor map. Available rooms are: {room_names}."
            return jsonify({"error": msg, "voice_response": msg}), 404
        start_room_id = None
        if current_image:
            if "," in current_image:
                current_image = current_image.split(",")[1]
            start_room_id = nav_system.get_current_location(current_image, indoor_map_data)
        if not start_room_id and indoor_map_data.get("rooms"):
            start_room_id = indoor_map_data["rooms"][0]["id"]
        if start_room_id not in nav_system.nodes:
            start_room_id = indoor_map_data["rooms"][0]["id"]
        instructions = nav_system.get_navigation_instructions(start_room_id, dest_room_id)
        if not instructions["success"]:
            msg = f"I could not find a path to {destination}."
            return jsonify({"error": msg, "voice_response": msg}), 404
        spoken = f"To reach the {destination}: "
        for i, inst in enumerate(instructions["instructions"]):
            spoken += ("First, " if i == 0 else "Then, ") + inst["instruction"].lower() + ". "
        spoken += f"Total approximately {instructions['total_steps']} steps."
        return jsonify({
            "status": "success",
            "current_location": nav_system.nodes[start_room_id].name,
            "destination": destination,
            "total_steps": instructions["total_steps"],
            "instructions": instructions["instructions"],
            "voice_response": spoken,
        })
    except Exception as e:
        msg = "Indoor navigation had a problem. " + get_voice_error("network")
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/indoor/get-current-location", methods=["POST"])
def indoor_get_current_location():
    try:
        data = request.get_json()
        image_data = data.get("image_data", "")
        if "," in image_data:
            image_data = image_data.split(",")[1]
        if not indoor_map_data:
            msg = "No indoor map available. Please upload a home walkthrough video first."
            return jsonify({"error": msg, "voice_response": msg}), 400
        location_id = nav_system.get_current_location(image_data, indoor_map_data)
        if location_id and location_id in nav_system.nodes:
            name = nav_system.nodes[location_id].name
            msg = f"Based on the camera view, you appear to be in the {name}."
            return jsonify({"status": "success", "location": {"id": location_id, "name": name}, "voice_response": msg})
        elif indoor_map_data.get("rooms"):
            first_room = indoor_map_data["rooms"][0]
            msg = f"I will assume you are in the {first_room['name']} area."
            return jsonify({"status": "success", "location": {"id": first_room["id"], "name": first_room["name"]}, "voice_response": msg})
        msg = "I could not determine your current location."
        return jsonify({"error": msg, "voice_response": msg}), 404
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/indoor/map", methods=["GET"])
def indoor_get_map():
    if indoor_map_data:
        return jsonify({"status": "success", "map": indoor_map_data, "rooms": [{"id": r["id"], "name": r["name"]} for r in indoor_map_data.get("rooms", [])]})
    return jsonify({"status": "error", "message": "No indoor map available"}), 404


@app.route("/api/indoor/navigation-modes", methods=["POST"])
def navigation_modes():
    global navigation_mode
    data = request.get_json()
    navigation_mode = data.get("mode", "outdoor")
    return jsonify({"status": "success", "mode": navigation_mode})


@app.route("/api/voice-destination", methods=["POST"])
def voice_destination():
    try:
        data = request.get_json()
        text = data.get("text", "")
        corrected = ask_groq_general(f"Correct this location name, fix spelling errors, return ONLY the corrected name: {text}")
        if corrected and len(corrected) < 100:
            return jsonify({"status": "success", "corrected": corrected.strip(), "original": text})
        return jsonify({"status": "success", "corrected": text, "original": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/assistant-query", methods=["POST"])
def assistant_query_legacy():
    try:
        data = request.get_json()
        question = data.get("question", "")
        context = data.get("context", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        if any(w in question.lower() for w in ["sos", "emergency", "bachao"]):
            msg = "Emergency detected. Activating SOS now."
            return jsonify({"status": "success", "response": msg, "voice_response": msg, "trigger_sos": True})
        response = ask_rag_assistant(question, context)
        return jsonify({"status": "success", "response": response, "voice_response": response, "trigger_sos": False})
    except Exception as e:
        msg = get_voice_error("network")
        return jsonify({"error": str(e), "message": msg, "voice_response": msg}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    key_status = {}
    for i, k in enumerate([GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3], 1):
        avail = _is_key_available(k)
        state = _groq_key_states.get(k, {})
        key_status[f"groq_key_{i}"] = {"available": avail, "failures": state.get("failures", 0)}
    memories = list_all_memories()
    conv_lines = len(get_conversation_context().splitlines())
    known_faces = []
    if os.path.exists("known_faces"):
        known_faces = [f for f in os.listdir("known_faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return jsonify({
        "status": "ok",
        "version": "v14",
        "pil": PIL_AVAILABLE,
        "opencv": CV2_AVAILABLE,
        "yolo": ULTRALYTICS_AVAILABLE,
        "deepface": DEEPFACE_AVAILABLE,
        "tesseract": TESSERACT_AVAILABLE,
        "serial": SERIAL_AVAILABLE,
        "arduino_connected": arduino_connected,
        "groq_keys": key_status,
        "gemini_key_set": bool(GEMINI_API_KEY),
        "groq_model": GROQ_MODEL,
        "active_mode": get_active_mode(),
        "voice_stopped": is_voice_stopped(),
        "global_stopped": is_global_stopped(),
        "assistant_active": get_assistant_active(),
        "memory": {"total_memories": len(memories), "known_faces": len(known_faces), "conversation_lines": conv_lines},
    })


@app.route("/favicon.ico")
def favicon():
    return "", 204


def cleanup():
    global sonar_active, arduino, sos_active, sos_timer
    sonar_active = False
    sos_active = False
    if sos_timer:
        sos_timer.cancel()
    time.sleep(0.3)
    if arduino and hasattr(arduino, "is_open") and arduino.is_open:
        arduino.close()


atexit.register(cleanup)


# ==================== MCP DIRECT ENDPOINTS (No camera check) ====================
# These endpoints are for the Prompt Opinion hackathon.
# They bypass the camera_active check and work directly with base64 images.

@app.route('/api/mcp/traffic', methods=['POST'])
def mcp_traffic():
    """MCP endpoint for traffic light detection - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 800) for f in frames[:1] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(TRAFFIC_SYSTEM, TRAFFIC_USER, processed, max_tokens=80, priority="high")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})





@app.route('/api/mcp/food', methods=['POST'])
def mcp_food():
    """MCP endpoint for food identification - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 800) for f in frames[:4] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(FOOD_SYSTEM, FOOD_USER, processed, max_tokens=250, priority="low")
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/read', methods=['POST'])
def mcp_read():
    """MCP endpoint for document/page reading - no camera check required"""
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        # Also support 'frames' array for compatibility
        if not image_data and data.get('frames'):
            image_data = data.get('frames', [])[0] if data.get('frames') else ''
        if not image_data:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = preprocess_image(image_data, 900)
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        result = groq_vision_call(PAGE_READER_SYSTEM, PAGE_READER_USER, [processed], max_tokens=1500, priority="high")
        return jsonify({"success": True, "result": result, "description": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/scene', methods=['POST'])
def mcp_scene():
    """MCP endpoint for scene description - no camera check required"""
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        question = data.get('question', 'What is in front of me? Describe the scene.')
        if not frames:
            return jsonify({"success": False, "result": "No image provided. Please send a base64 encoded image."})
        processed = [preprocess_image(f, 900) for f in frames[:3] if f and len(f) > 100]
        if not processed:
            return jsonify({"success": False, "result": "Could not process the image. Please ensure it's a valid JPEG."})
        user_prompt = (
            f'The user asks: "{question}". '
            'Describe clearly what you see in this image for a visually impaired person. '
            'Be specific about locations (left, right, center, near, far).'
        )
        result = groq_vision_call(
            "You are describing a scene for a visually impaired person. Be clear, helpful, and specific about object locations. No markdown. Natural speech.",
            user_prompt,
            processed,
            max_tokens=300,
            priority="high"
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "result": f"Error: {str(e)}"})


@app.route('/api/mcp/health', methods=['GET'])
def mcp_health():
    """Health check endpoint for MCP server"""
    return jsonify({"status": "healthy", "version": "1.0", "tools": ["traffic", "food", "read", "scene"]})




# ==================== PROPER MCP SERVER ENDPOINT ====================
# Prompt Opinion requires JSON-RPC 2.0 over HTTP (Streamable HTTP transport)
# Single endpoint at /mcp that handles initialize, tools/list, tools/call

import json as _json

_MCP_TOOLS = [
    {
        "name": "traffic_light_detector",
        "description": "Detects traffic light color (red/green/yellow) from a base64 image. RED = safe to cross, GREEN = stop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array with one base64 encoded JPEG image of the traffic light"
                }
            },
            "required": ["frames"]
        }
    },
    {
        "name": "food_identifier",
        "description": "Identifies food items visible in an image. Returns food names, quantities, and meal type.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of base64 encoded JPEG images showing food"
                }
            },
            "required": ["frames"]
        }
    },
    {
        "name": "document_reader",
        "description": "Reads and extracts all text from a document, book page, prescription, or bill in an image.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_data": {
                    "type": "string",
                    "description": "Base64 encoded JPEG image of the document"
                }
            },
            "required": ["image_data"]
        }
    },
    {
        "name": "scene_describer",
        "description": "Describes what is in front of a visually impaired person. Specifies object locations (left, right, near, far).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of base64 encoded JPEG images of the scene"
                },
                "question": {
                    "type": "string",
                    "description": "Optional question about the scene"
                }
            },
            "required": ["frames"]
        }
    }
]


def _direct_groq_vision(system: str, user: str, frame_b64: str, max_tokens: int = 200) -> str:
    """Fully self-contained Groq vision call — reads keys from os.environ,
    uses raw requests, zero dependency on module-level state or failure counter."""
    import requests as _req
    import base64 as _b64
    from PIL import Image as _Image
    import io as _io

    # Strip data URI prefix if present before decode
    raw_b64 = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64

    # Resize to tiny to stay well within Groq limits
    try:
        raw = _b64.b64decode(raw_b64 + "=" * (4 - len(raw_b64) % 4))
        img = _Image.open(_io.BytesIO(raw)).convert("RGB")
        img.thumbnail((100, 100))
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=40)
        raw_b64 = _b64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[_direct_groq_vision] resize: {e}")

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{raw_b64}",
                    "detail": "low"
                }}
            ]}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    # Read keys fresh from environment — never touches module-level state
    keys = [
        os.environ.get("GROQ_API_KEY_1", ""),
        os.environ.get("GROQ_API_KEY_2", ""),
        os.environ.get("GROQ_API_KEY_3", ""),
    ]

    for key in keys:
        if not key:
            continue
        try:
            r = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                timeout=25
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"[_direct_groq_vision] status={r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"[_direct_groq_vision] request error: {e}")

    return "Could not process image."


def _mcp_call_tool(name: str, arguments: dict) -> str:
    """Route MCP tool calls through _direct_groq_vision to bypass the failure counter."""
    try:
        if name == "traffic_light_detector":
            frames = arguments.get("frames", [])
            if not frames:
                return "No image provided."
            return _direct_groq_vision(TRAFFIC_SYSTEM, TRAFFIC_USER, frames[0], max_tokens=80)

        elif name == "food_identifier":
            frames = arguments.get("frames", [])
            if not frames:
                return "No image provided."
            return _direct_groq_vision(FOOD_SYSTEM, FOOD_USER, frames[0], max_tokens=250)

        elif name == "document_reader":
            image_data = arguments.get("image_data", "")
            if not image_data and arguments.get("frames"):
                image_data = arguments["frames"][0]
            if not image_data:
                return "No image provided."
            return _direct_groq_vision(PAGE_READER_SYSTEM, PAGE_READER_USER, image_data, max_tokens=1500)

        elif name == "scene_describer":
            frames = arguments.get("frames", [])
            question = arguments.get("question", "What is in front of me?")
            if not frames:
                return "No image provided."
            return _direct_groq_vision(
                "You are describing a scene for a visually impaired person. Be clear and specific. No markdown. Natural speech.",
                f'User asks: "{question}". Describe clearly with object locations (left, right, near, far).',
                frames[0],
                max_tokens=300,
            )

        else:
            return f"Unknown tool: {name}"

    except Exception as e:
        return f"Error: {str(e)}"



@app.route('/mcp', methods=['GET', 'POST', 'DELETE', 'OPTIONS'])
def mcp_endpoint():
    """
    Proper MCP JSON-RPC 2.0 endpoint (Streamable HTTP transport).
    Prompt Opinion connects here.
    """
    # CORS headers for Prompt Opinion
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, Mcp-Session-Id, MCP-Protocol-Version',
        'Content-Type': 'application/json'
    }

    if request.method == 'OPTIONS':
        return '', 204, headers

    if request.method == 'GET':
        # SSE stream for server-initiated messages — return 405 (we don't need it)
        return jsonify({"error": "SSE not supported, use POST"}), 405, headers

    if request.method == 'DELETE':
        return '', 200, headers

    # POST — handle JSON-RPC
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({
            "jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        }), 400, headers

    rpc_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    # --- initialize ---
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "VisionAssist AI Smart Glasses",
                    "version": "1.0.0"
                }
            }
        }
        return jsonify(response), 200, headers

    # --- initialized (notification, no response needed) ---
    if method == "notifications/initialized":
        return '', 202, headers

    # --- tools/list ---
    if method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {"tools": _MCP_TOOLS}
        }
        return jsonify(response), 200, headers

    # --- tools/call ---
    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result_text = _mcp_call_tool(tool_name, arguments)
        response = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
        }
        return jsonify(response), 200, headers

    # --- ping ---
    if method == "ping":
        return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": {}}), 200, headers

    # --- unknown method ---
    return jsonify({
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"}
    }), 200, headers


@app.route('/mcp/health', methods=['GET'])
def mcp_proper_health():
    return jsonify({"status": "healthy", "version": "1.0", "tools": [t["name"] for t in _MCP_TOOLS]})

@app.route('/api/test-groq', methods=['GET'])
def test_groq_simple():
    """Test Groq with a tiny programmatically generated image"""
    try:
        from PIL import Image, ImageDraw
        import io, base64
        
        # Make tiny red circle image
        img = Image.new('RGB', (64, 64), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse([5, 5, 59, 59], fill=(255, 0, 0))
        
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Direct Groq call
        import requests as req
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this circle? One word answer."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low"
                    }}
                ]
            }],
            "max_tokens": 10,
            "temperature": 0.0
        }
        
        keys = [
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", "")
        ]
        
        results = {}
        for i, key in enumerate(keys, 1):
            if not key:
                results[f"key_{i}"] = "NOT SET"
                continue
            try:
                r = req.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    timeout=15
                )
                results[f"key_{i}"] = {
                    "status_code": r.status_code,
                    "response": r.json() if r.status_code == 200 else r.text[:200]
                }
            except Exception as e:
                results[f"key_{i}"] = f"Exception: {str(e)}"
        
        return jsonify({"success": True, "groq_results": results})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    

@app.route('/api/test-mcp-traffic', methods=['GET'])
def test_mcp_traffic():
    from PIL import Image, ImageDraw
    import io, base64
    img = Image.new('RGB', (64, 64), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([5, 5, 59, 59], fill=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    processed = preprocess_image(b64, 320)
    result = groq_vision_call(TRAFFIC_SYSTEM, TRAFFIC_USER, [processed], max_tokens=80, priority="high")
    return jsonify({"success": True, "result": result})




@app.route('/api/debug-mcp', methods=['POST'])
def debug_mcp():
    import requests as _req
    import base64 as _b64
    from PIL import Image as _Image
    import io as _io
    
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        frame = frames[0]
        
        # Step 1 - resize
        raw = _b64.b64decode(frame)
        img = _Image.open(_io.BytesIO(raw)).convert("RGB")
        img.thumbnail((150, 150))
        buf = _io.BytesIO()
        img.save(buf, format='JPEG', quality=55)
        small = _b64.b64encode(buf.getvalue()).decode()
        
        # Step 2 - direct groq call NO groq_vision_call
        keys = [
            os.environ.get("GROQ_API_KEY_1", ""),
            os.environ.get("GROQ_API_KEY_2", ""),
            os.environ.get("GROQ_API_KEY_3", "")
        ]
        
        for key in keys:
            if not key:
                continue
            r = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "What color is the traffic light?"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{small}",
                            "detail": "low"
                        }}
                    ]}],
                    "max_tokens": 20,
                    "temperature": 0.0
                },
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                timeout=25
            )
            return jsonify({
                "status_code": r.status_code,
                "response": r.json() if r.status_code == 200 else r.text,
                "key_used": key[:8] + "..."
            })
            
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/debug-direct', methods=['GET'])
def debug_direct():
    import requests as _req
    import base64 as _b64
    from PIL import Image as _Image
    import io as _io

    # Generate red image internally - no external image needed
    img = _Image.new('RGB', (64, 64), color=(0,0,0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([5,5,59,59], fill=(255,0,0))
    buf = _io.BytesIO()
    img.save(buf, format='JPEG', quality=60)
    small = _b64.b64encode(buf.getvalue()).decode()

    keys = [
        os.environ.get("GROQ_API_KEY_1",""),
        os.environ.get("GROQ_API_KEY_2",""),
        os.environ.get("GROQ_API_KEY_3","")
    ]

    results = {}
    for i, key in enumerate(keys, 1):
        if not key:
            results[f"key_{i}"] = "NOT SET"
            continue
        try:
            r = _req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "What color is this circle? One word."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{small}",
                            "detail": "low"
                        }}
                    ]}],
                    "max_tokens": 10,
                    "temperature": 0.0
                },
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                timeout=25
            )
            results[f"key_{i}"] = {
                "status": r.status_code,
                "body": r.json() if r.status_code == 200 else r.text[:300]
            }
        except Exception as e:
            results[f"key_{i}"] = str(e)

    return jsonify(results)


if __name__ == "__main__":
    if ULTRALYTICS_AVAILABLE and CV2_AVAILABLE:
        threading.Thread(target=_get_myeye_yolo, daemon=True).start()

    print("=" * 70)
    print("VisionAssist AI v14 - Starting...")
    print("=" * 70)

    memories = list_all_memories()
    known_faces = []
    if os.path.exists("known_faces"):
        known_faces = [f for f in os.listdir("known_faces") if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"Memory: {len(memories)} saved objects")
    print(f"Known faces: {len(known_faces)}")
    print(f"Arduino: {'Connected on ' + arduino_port if arduino_connected else 'Not connected'}")
    print(f"PIL: {'OK' if PIL_AVAILABLE else 'MISSING - pip install pillow'}")
    print(f"OpenCV: {'OK' if CV2_AVAILABLE else 'MISSING - pip install opencv-python'}")
    print(f"YOLO: {'OK' if ULTRALYTICS_AVAILABLE else 'Optional - pip install ultralytics'}")
    print(f"DeepFace: {'OK' if DEEPFACE_AVAILABLE else 'Optional - pip install deepface'}")
    print()
    print("Server running at: http://localhost:5000")
    print("=" * 70)

    if not os.path.exists("templates"):
        os.makedirs("templates")

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)

