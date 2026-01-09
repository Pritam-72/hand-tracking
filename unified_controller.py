"""
Unified Hand-Tracking Mouse Controller
=======================================
Control your mouse with hand gestures using your webcam.
Combines the best features from both original controllers with improvements.

Gestures:
---------
- Double Pinch (👌👌): Activate cursor control
- Hand Movement: Move cursor (when active)
- Single Pinch (👌): Left click
- Pinch + Hold: Click and drag
- Pinch + Vertical Move: Scroll up/down
- Open Palm (🖐️): Right click
- Closed Fist (✊): Deactivate cursor
- Three Fingers + Thumb: Screenshot

Requirements:
    pip install -r requirements.txt

Usage:
    python unified_controller.py
    Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import json
import os
import sys
import time
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from pynput.mouse import Button, Controller as MouseController

# Disable PyAutoGUI fail-safe for edge-of-screen movements
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


# ============== CONFIGURATION ==============
def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    default_config = {
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
    
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            # Merge user config with defaults
            for key in default_config:
                if key in user_config:
                    if isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
    except FileNotFoundError:
        print(f"Config file not found. Using defaults.")
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}. Using defaults.")
    
    return default_config


# ============== AUDIO FEEDBACK ==============
def play_sound(frequency=800, duration=100):
    """Play a beep sound for feedback (Windows compatible)."""
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.Beep(frequency, duration)
        else:
            # For Linux/Mac, use system beep or print
            print('\a', end='', flush=True)
    except Exception:
        pass  # Silently fail if sound doesn't work


# ============== STATE MACHINE ==============
class CursorState(IntEnum):
    INACTIVE = 0    # Not tracking cursor
    ACTIVE = 1      # Cursor follows hand
    CLICKING = 2    # Click in progress
    DRAGGING = 3    # Mouse button held, dragging
    SCROLLING = 4   # Scroll mode


# ============== LANDMARK INDICES ==============
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18
WRIST = 0


# ============== HELPER FUNCTIONS ==============
def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_angle(a, b, c):
    """Calculate angle between three points (a, b, c) with b as the vertex."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle


# ============== MAIN CONTROLLER CLASS ==============
class UnifiedHandController:
    def __init__(self, config_path="config.json"):
        """Initialize the unified hand controller."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize mouse controller
        self.mouse = MouseController()
        
        # Get screen dimensions (cross-platform)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config["detection"]["max_num_hands"],
            min_detection_confidence=self.config["detection"]["min_detection_confidence"],
            min_tracking_confidence=self.config["detection"]["min_tracking_confidence"]
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config["camera"]["index"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"]["height"])
        
        # State variables
        self.state = CursorState.INACTIVE
        self.smooth_x, self.smooth_y = 0, 0
        self.last_pinch_time = 0
        self.pinch_count = 0
        self.pinch_start_time = 0
        self.was_pinching = False
        self.scroll_start_y = 0
        
        # OPTIMIZATION: Position history for double-smoothing
        self.position_history = []
        self.history_size = 5  # Average over last 5 positions
        
        # OPTIMIZATION: Gesture debouncing
        self.gesture_history = {'pinch': [], 'fist': [], 'palm': [], 'two_finger': []}
        self.debounce_frames = 3  # Require 3 consistent frames
        
        # Scroll tracking
        self.scroll_last_y = 0
        self.scroll_active = False
        
        # OPTIMIZATION: FPS limiting
        self.target_fps = 30
        self.last_frame_time = 0
        
        # Cooldown trackers
        self.last_click_time = 0
        self.last_screenshot_time = 0
        self.last_scroll_time = 0
        self.last_right_click_time = 0
        
        # Visual feedback
        self.action_text = ""
        self.action_time = 0
        self.action_color = (255, 255, 255)
        
        # Create screenshots folder
        self.screenshots_folder = Path(self.config["screenshots"]["folder"])
        self.screenshots_folder.mkdir(exist_ok=True)
        
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
    
    # ---------- GESTURE DETECTION ----------
    def is_pinching(self, landmarks, frame_width, frame_height):
        """Check if thumb and index finger are pinching."""
        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]
        
        thumb_pos = (thumb.x * frame_width, thumb.y * frame_height)
        index_pos = (index.x * frame_width, index.y * frame_height)
        
        distance = get_distance(thumb_pos, index_pos)
        return distance < self.config["gestures"]["pinch_threshold"]
    
    def is_fist(self, landmarks):
        """Check if hand is closed (fist/closed palm)."""
        fingertips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pip_joints = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        
        closed_count = 0
        for tip, pip in zip(fingertips, pip_joints):
            if landmarks[tip].y > landmarks[pip].y:
                closed_count += 1
        
        # Check thumb
        if landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x:
            closed_count += 1
        
        return closed_count >= self.config["gestures"]["fist_threshold"]
    
    def is_open_palm(self, landmarks):
        """Check if hand is open (all fingers extended)."""
        fingertips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        pip_joints = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
        
        extended_count = 0
        for tip, pip in zip(fingertips, pip_joints):
            if landmarks[tip].y < landmarks[pip].y:
                extended_count += 1
        
        return extended_count >= 4
    
    def is_three_fingers_thumb(self, landmarks):
        """Check for screenshot gesture: thumb + three fingers extended, pinky closed."""
        # Index, middle, ring extended
        fingers_extended = (
            landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y and
            landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y and
            landmarks[RING_TIP].y < landmarks[RING_PIP].y
        )
        # Pinky closed
        pinky_closed = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
        
        return fingers_extended and pinky_closed
    
    def is_two_finger_scroll(self, landmarks):
        """Check for scroll gesture: index + middle extended, ring + pinky closed."""
        # Index and middle extended
        index_extended = landmarks[INDEX_TIP].y < landmarks[INDEX_PIP].y
        middle_extended = landmarks[MIDDLE_TIP].y < landmarks[MIDDLE_PIP].y
        
        # Ring and pinky closed
        ring_closed = landmarks[RING_TIP].y > landmarks[RING_PIP].y
        pinky_closed = landmarks[PINKY_TIP].y > landmarks[PINKY_PIP].y
        
        # Thumb should be relaxed (not pinching)
        thumb_relaxed = True  # We'll allow thumb in any position
        
        return index_extended and middle_extended and ring_closed and pinky_closed
    
    def get_index_position(self, landmarks, frame_width, frame_height):
        """Get index fingertip position in frame coordinates."""
        index = landmarks[INDEX_TIP]
        return (index.x * frame_width, index.y * frame_height)
    
    def map_to_screen(self, x, y, frame_width, frame_height):
        """Map camera coordinates to screen coordinates with boundary margin."""
        reduction = self.config["cursor"]["frame_reduction"]
        
        x = np.clip(x, reduction, frame_width - reduction)
        y = np.clip(y, reduction, frame_height - reduction)
        
        screen_x = np.interp(x, (reduction, frame_width - reduction), (0, self.screen_width))
        screen_y = np.interp(y, (reduction, frame_height - reduction), (0, self.screen_height))
        
        return screen_x, screen_y
    
    def update_smooth_position(self, x, y):
        """Apply double-smoothing: exponential + moving average."""
        factor = self.config["cursor"]["smoothing_factor"]
        
        # First pass: exponential smoothing
        self.smooth_x = self.smooth_x * factor + x * (1 - factor)
        self.smooth_y = self.smooth_y * factor + y * (1 - factor)
        
        # Second pass: moving average over history
        self.position_history.append((self.smooth_x, self.smooth_y))
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # Average all positions in history
        avg_x = sum(p[0] for p in self.position_history) / len(self.position_history)
        avg_y = sum(p[1] for p in self.position_history) / len(self.position_history)
        
        return avg_x, avg_y
    
    def debounce_gesture(self, gesture_name, current_value):
        """Debounce gesture detection to prevent false positives."""
        history = self.gesture_history[gesture_name]
        history.append(current_value)
        
        if len(history) > self.debounce_frames:
            history.pop(0)
        
        # Only return True if gesture detected in majority of recent frames
        if len(history) >= self.debounce_frames:
            return sum(history) >= (self.debounce_frames - 1)
        return False
    
    # ---------- ACTION HANDLERS ----------
    def perform_click(self):
        """Perform a left click with cooldown."""
        current_time = time.time()
        if current_time - self.last_click_time >= self.config["cooldowns"]["click"]:
            self.mouse.click(Button.left)
            self.last_click_time = current_time
            self.show_action("LEFT CLICK", (0, 255, 255))
            if self.config["feedback"]["enable_sound"]:
                play_sound(800, 50)
            return True
        return False
    
    def perform_right_click(self):
        """Perform a right click with cooldown."""
        current_time = time.time()
        if current_time - self.last_right_click_time >= self.config["cooldowns"]["click"]:
            self.mouse.click(Button.right)
            self.last_right_click_time = current_time
            self.show_action("RIGHT CLICK", (255, 0, 255))
            if self.config["feedback"]["enable_sound"]:
                play_sound(600, 50)
            return True
        return False
    
    def take_screenshot(self):
        """Take a screenshot with cooldown."""
        current_time = time.time()
        if current_time - self.last_screenshot_time >= self.config["cooldowns"]["screenshot"]:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            prefix = self.config["screenshots"]["prefix"]
            fmt = self.config["screenshots"]["format"]
            filename = self.screenshots_folder / f"{prefix}_{timestamp}.{fmt}"
            
            try:
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filename))
                self.last_screenshot_time = current_time
                self.show_action(f"SCREENSHOT SAVED", (255, 255, 255))
                if self.config["feedback"]["enable_sound"]:
                    play_sound(1000, 200)
                print(f"Screenshot saved: {filename}")
                return True
            except Exception as e:
                self.show_action("SCREENSHOT FAILED", (0, 0, 255))
                print(f"Screenshot error: {e}")
        return False
    
    def perform_scroll(self, current_y):
        """Perform scroll based on vertical movement."""
        current_time = time.time()
        if current_time - self.last_scroll_time >= self.config["cooldowns"]["scroll"]:
            delta_y = self.scroll_start_y - current_y
            scroll_amount = int(delta_y * self.config["gestures"]["scroll_sensitivity"] / 10)
            
            if abs(scroll_amount) > 0:
                pyautogui.scroll(scroll_amount)
                self.scroll_start_y = current_y
                self.last_scroll_time = current_time
    
    def show_action(self, text, color):
        """Set action text for visual feedback."""
        self.action_text = text
        self.action_time = time.time()
        self.action_color = color
    
    # ---------- MAIN GESTURE HANDLER ----------
    def handle_gestures(self, landmarks, frame_width, frame_height):
        """Process hand gestures and update state."""
        current_time = time.time()
        
        # Raw gesture detection
        raw_pinching = self.is_pinching(landmarks, frame_width, frame_height)
        raw_fist = self.is_fist(landmarks)
        raw_open_palm = self.is_open_palm(landmarks)
        raw_two_finger = self.is_two_finger_scroll(landmarks)
        screenshot_gesture = self.is_three_fingers_thumb(landmarks)
        
        # OPTIMIZATION: Debounce gestures to prevent false positives
        pinching = self.debounce_gesture('pinch', raw_pinching)
        fist = self.debounce_gesture('fist', raw_fist)
        open_palm = self.debounce_gesture('palm', raw_open_palm)
        two_finger_scroll = self.debounce_gesture('two_finger', raw_two_finger)
        
        # Get index finger position
        index_x, index_y = self.get_index_position(landmarks, frame_width, frame_height)
        screen_x, screen_y = self.map_to_screen(index_x, index_y, frame_width, frame_height)
        
        # Check for screenshot gesture (works in any state)
        if screenshot_gesture and self.state != CursorState.INACTIVE:
            self.take_screenshot()
        
        # State machine logic
        if self.state == CursorState.INACTIVE:
            # Check for double pinch to activate
            if pinching and not self.was_pinching:
                if current_time - self.last_pinch_time < self.config["gestures"]["double_pinch_time"]:
                    self.pinch_count += 1
                else:
                    self.pinch_count = 1
                self.last_pinch_time = current_time
                
                if self.pinch_count >= 2:
                    self.state = CursorState.ACTIVE
                    self.pinch_count = 0
                    self.smooth_x, self.smooth_y = screen_x, screen_y
                    self.show_action("CURSOR ACTIVATED", (0, 255, 0))
                    if self.config["feedback"]["enable_sound"]:
                        play_sound(1200, 100)
                    print(">>> CURSOR ACTIVATED <<<")
        
        elif self.state == CursorState.ACTIVE:
            # Check for fist to deactivate
            if fist:
                self.state = CursorState.INACTIVE
                self.pinch_count = 0
                self.show_action("CURSOR DEACTIVATED", (0, 0, 255))
                if self.config["feedback"]["enable_sound"]:
                    play_sound(400, 100)
                print(">>> CURSOR DEACTIVATED <<<")
            
            # Check for open palm (right click)
            elif open_palm:
                self.perform_right_click()
            
            # Check for two-finger scroll gesture
            elif two_finger_scroll:
                if not self.scroll_active:
                    self.scroll_active = True
                    self.scroll_last_y = index_y
                    self.state = CursorState.SCROLLING
                    self.show_action("SCROLL MODE", (255, 165, 0))
                else:
                    # Perform scroll based on vertical movement
                    delta_y = self.scroll_last_y - index_y
                    scroll_amount = int(delta_y * self.config["gestures"]["scroll_sensitivity"])
                    if abs(scroll_amount) > 0:
                        pyautogui.scroll(scroll_amount)
                        self.scroll_last_y = index_y
            
            # Check for pinch (click, drag, or scroll)
            elif pinching:
                if not self.was_pinching:
                    self.pinch_start_time = current_time
                    self.scroll_start_y = index_y
                
                # Check if held long enough for drag
                hold_time = current_time - self.pinch_start_time
                if hold_time > self.config["gestures"]["drag_hold_time"]:
                    if self.state != CursorState.DRAGGING:
                        self.state = CursorState.DRAGGING
                        self.mouse.press(Button.left)
                        self.show_action("DRAGGING", (0, 165, 255))
                        print(">>> DRAG STARTED <<<")
                else:
                    self.state = CursorState.CLICKING
            else:
                # Pinch released
                if self.was_pinching:
                    if self.state == CursorState.CLICKING:
                        self.perform_click()
                    elif self.state == CursorState.DRAGGING:
                        self.mouse.release(Button.left)
                        self.show_action("DRAG ENDED", (0, 255, 0))
                        print(">>> DRAG ENDED <<<")
                    self.state = CursorState.ACTIVE
                
                # Reset scroll state when two-finger gesture stops
                if self.scroll_active and not two_finger_scroll:
                    self.scroll_active = False
                    self.state = CursorState.ACTIVE
            
            # Move cursor
            if self.state in [CursorState.ACTIVE, CursorState.CLICKING, CursorState.DRAGGING]:
                smooth_x, smooth_y = self.update_smooth_position(screen_x, screen_y)
                self.mouse.position = (int(smooth_x), int(smooth_y))
        
        elif self.state == CursorState.DRAGGING:
            # Continue dragging
            if fist:
                self.mouse.release(Button.left)
                self.state = CursorState.INACTIVE
                self.show_action("DRAG CANCELLED", (0, 0, 255))
                print(">>> DRAG CANCELLED <<<")
            elif not pinching:
                self.mouse.release(Button.left)
                self.state = CursorState.ACTIVE
                self.show_action("DRAG ENDED", (0, 255, 0))
                print(">>> DRAG ENDED <<<")
            else:
                smooth_x, smooth_y = self.update_smooth_position(screen_x, screen_y)
                self.mouse.position = (int(smooth_x), int(smooth_y))
                # Check for scroll during drag
                self.perform_scroll(index_y)
        
        self.was_pinching = pinching
        return screen_x, screen_y
    
    # ---------- UI HELPERS ----------
    def get_state_text(self):
        """Get display text for current state."""
        states = {
            CursorState.INACTIVE: "INACTIVE - Double pinch to activate",
            CursorState.ACTIVE: "ACTIVE - Pinch=click, Fist=off, Palm=right-click",
            CursorState.CLICKING: "CLICKING...",
            CursorState.DRAGGING: "DRAGGING...",
            CursorState.SCROLLING: "SCROLLING..."
        }
        return states.get(self.state, "UNKNOWN")
    
    def get_state_color(self):
        """Get display color for current state."""
        colors = {
            CursorState.INACTIVE: (128, 128, 128),
            CursorState.ACTIVE: (0, 255, 0),
            CursorState.CLICKING: (0, 255, 255),
            CursorState.DRAGGING: (0, 165, 255),
            CursorState.SCROLLING: (255, 165, 0)
        }
        return colors.get(self.state, (255, 255, 255))
    
    def draw_ui(self, frame, frame_width, frame_height):
        """Draw UI elements on frame."""
        # Draw state indicator
        state_text = self.get_state_text()
        state_color = self.get_state_color()
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Draw activation circle
        circle_color = (0, 255, 0) if self.state != CursorState.INACTIVE else (0, 0, 255)
        cv2.circle(frame, (frame_width - 30, 30), 15, circle_color, -1)
        
        # Draw action feedback (fades after 1 second)
        if self.action_text and time.time() - self.action_time < 1.0:
            alpha = 1.0 - (time.time() - self.action_time)
            color = tuple(int(c * alpha) for c in self.action_color)
            cv2.putText(frame, self.action_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw gesture hints
        hints = [
            "Gestures: Double-Pinch=ON | Fist=OFF",
            "Pinch=Click | Hold=Drag | Palm=Right-Click",
            "3 Fingers+Thumb = Screenshot"
        ]
        y_offset = frame_height - 70
        for hint in hints:
            cv2.putText(frame, hint, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 20
    
    # ---------- MAIN LOOP ----------
    def run(self):
        """Main loop."""
        print("=" * 55)
        print("   UNIFIED HAND MOUSE CONTROLLER")
        print("=" * 55)
        print(f"\nScreen: {self.screen_width}x{self.screen_height}")
        print(f"Screenshots folder: {self.screenshots_folder.absolute()}")
        print("\nGESTURES:")
        print("  1. Double pinch            -> ACTIVATE cursor")
        print("  2. Move hand               -> Move cursor")
        print("  3. Single pinch            -> LEFT CLICK")
        print("  4. Pinch + hold            -> DRAG")
        print("  5. Open palm               -> RIGHT CLICK")
        print("  6. 3 fingers + thumb       -> SCREENSHOT")
        print("  7. Closed fist             -> DEACTIVATE")
        print("\nPress 'q' to quit")
        print("-" * 55)
        
        try:
            while True:
                # OPTIMIZATION: FPS limiting to reduce CPU usage
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < 1.0 / self.target_fps:
                    time.sleep((1.0 / self.target_fps) - elapsed)
                self.last_frame_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Process hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Handle gestures
                        self.handle_gestures(
                            hand_landmarks.landmark,
                            frame_width, frame_height
                        )
                
                # Draw UI
                self.draw_ui(frame, frame_width, frame_height)
                
                # Show frame
                cv2.imshow("Unified Hand Controller", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.state == CursorState.DRAGGING:
            self.mouse.release(Button.left)
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nUnified Hand Controller stopped.")


# ============== ENTRY POINT ==============
if __name__ == "__main__":
    controller = UnifiedHandController()
    controller.run()
