"""
Utility Functions for Hand Tracking Controller
===============================================
Helper functions for gesture detection, configuration, and feedback.
"""

import json
import sys
import numpy as np
from pathlib import Path


# ============== GEOMETRY HELPERS ==============

def get_angle(a, b, c):
    """
    Calculate angle between three points (a, b, c) with b as the vertex.
    
    Args:
        a, b, c: Tuples of (x, y) coordinates
    
    Returns:
        Angle in degrees (0-180)
    """
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle


def get_distance(landmark_list):
    """
    Calculate scaled distance between two landmarks.
    
    Args:
        landmark_list: List of at least 2 (x, y) tuples
    
    Returns:
        Scaled distance (0-1000) or None if invalid input
    """
    if len(landmark_list) < 2:
        return None
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])


def get_euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1, p2: Tuples of (x, y) coordinates
    
    Returns:
        Distance as float
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# ============== CONFIGURATION ==============

DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 640, "height": 480},
    "detection": {
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7,
        "max_num_hands": 1
    },
    "cursor": {"smoothing_factor": 0.5, "frame_reduction": 100},
    "gestures": {
        "pinch_threshold": 40,
        "double_pinch_time": 0.5,
        "drag_hold_time": 0.2,
        "scroll_sensitivity": 5.0,
        "fist_threshold": 4
    },
    "cooldowns": {
        "click": 0.3,
        "double_click": 0.5,
        "screenshot": 1.0,
        "scroll": 0.05
    },
    "feedback": {
        "enable_sound": True,
        "enable_visual": True,
        "sound_frequency": 800,
        "sound_duration": 100
    },
    "screenshots": {
        "folder": "screenshots",
        "format": "png",
        "prefix": "screenshot"
    }
}


def load_config(config_path="config.json"):
    """
    Load configuration from JSON file, falling back to defaults.
    
    Args:
        config_path: Path to config.json file
    
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            # Deep merge user config with defaults
            for key in config:
                if key in user_config:
                    if isinstance(config[key], dict):
                        config[key].update(user_config[key])
                    else:
                        config[key] = user_config[key]
        print(f"Loaded config from {config_path}")
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using defaults.")
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using defaults.")
    
    return config


def save_config(config, config_path="config.json"):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config.json
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Error saving config: {e}")


# ============== AUDIO FEEDBACK ==============

def play_sound(frequency=800, duration=100):
    """
    Play a beep sound for feedback (cross-platform).
    
    Args:
        frequency: Sound frequency in Hz (Windows only)
        duration: Duration in milliseconds (Windows only)
    """
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.Beep(frequency, duration)
        elif sys.platform == 'darwin':
            # macOS
            import os
            os.system('afplay /System/Library/Sounds/Tink.aiff &')
        else:
            # Linux - try multiple methods
            try:
                import os
                os.system('paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null &')
            except:
                print('\a', end='', flush=True)
    except Exception:
        pass  # Silently fail if sound doesn't work


def play_click_sound():
    """Play a short click sound."""
    play_sound(800, 50)


def play_activate_sound():
    """Play activation sound."""
    play_sound(1200, 100)


def play_deactivate_sound():
    """Play deactivation sound."""
    play_sound(400, 100)


def play_screenshot_sound():
    """Play screenshot sound."""
    play_sound(1000, 200)
