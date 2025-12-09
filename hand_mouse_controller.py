"""
Hand-Tracking Mouse Controller
==============================
Control your mouse with hand gestures using your webcam.

Gestures:
- Double Pinch (👌👌): Activate cursor control
- Single Pinch (👌): Left click
- Pinch + Hold + Move: Click and drag
- Hand Movement: Move cursor (when active)
- Closed Palm (✊): Deactivate cursor

Requirements:
    pip install opencv-python mediapipe pynput numpy

Usage:
    python hand_mouse_controller.py
    Press 'q' to quit
"""

import os
# Fix Qt platform issue on Wayland systems
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import mediapipe as mp
import numpy as np
import time
from enum import IntEnum
from pynput.mouse import Button, Controller as MouseController

# ============== CONFIGURATION ==============
SMOOTHING_FACTOR = 0.5       # Cursor smoothing (0.0-1.0, higher = smoother)
CLICK_THRESHOLD = 40         # Pinch detection distance (pixels)
DOUBLE_PINCH_TIME = 0.5      # Max time between pinches for double-pinch (seconds)
DRAG_HOLD_TIME = 0.2         # Time to hold pinch before drag mode (seconds)
FRAME_REDUCTION = 100        # Screen boundary margin (pixels)
CAMERA_INDEX = 0             # Camera device index
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ============== MOUSE CONTROLLER ==============
mouse = MouseController()

# Get screen size using Xlib
try:
    from Xlib import display
    d = display.Display()
    screen = d.screen()
    SCREEN_WIDTH = screen.width_in_pixels
    SCREEN_HEIGHT = screen.height_in_pixels
except:
    # Fallback
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

# ============== STATE MACHINE ==============
class CursorState(IntEnum):
    INACTIVE = 0   # Not tracking cursor
    ACTIVE = 1     # Cursor follows hand
    CLICKING = 2   # Click in progress
    DRAGGING = 3   # Mouse button held, dragging

# ============== LANDMARK INDICES ==============
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_IP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

# ============== HELPER FUNCTIONS ==============
def get_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_pinching(landmarks, frame_width, frame_height):
    """Check if thumb and index finger are pinching."""
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    
    thumb_pos = (thumb.x * frame_width, thumb.y * frame_height)
    index_pos = (index.x * frame_width, index.y * frame_height)
    
    distance = get_distance(thumb_pos, index_pos)
    return distance < CLICK_THRESHOLD

def is_fist(landmarks):
    """Check if hand is closed (fist/closed palm)."""
    fingertips = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pip_joints = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    
    closed_count = 0
    for tip, pip in zip(fingertips, pip_joints):
        if landmarks[tip].y > landmarks[pip].y:
            closed_count += 1
    
    if landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x:
        closed_count += 1
    
    return closed_count >= 4

def get_index_position(landmarks, frame_width, frame_height):
    """Get index fingertip position in frame coordinates."""
    index = landmarks[INDEX_TIP]
    return (index.x * frame_width, index.y * frame_height)

def map_to_screen(x, y, frame_width, frame_height, screen_width, screen_height):
    """Map camera coordinates to screen coordinates with boundary margin."""
    x = np.clip(x, FRAME_REDUCTION, frame_width - FRAME_REDUCTION)
    y = np.clip(y, FRAME_REDUCTION, frame_height - FRAME_REDUCTION)
    
    screen_x = np.interp(x, 
                         (FRAME_REDUCTION, frame_width - FRAME_REDUCTION),
                         (0, screen_width))
    screen_y = np.interp(y,
                         (FRAME_REDUCTION, frame_height - FRAME_REDUCTION),
                         (0, screen_height))
    
    return screen_x, screen_y

# ============== MAIN CLASS ==============
class HandMouseApp:
    def __init__(self):
        # Initialize Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Screen dimensions
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        
        # State variables
        self.state = CursorState.INACTIVE
        self.smooth_x, self.smooth_y = 0, 0
        self.last_pinch_time = 0
        self.pinch_count = 0
        self.pinch_start_time = 0
        self.was_pinching = False
        
    def update_smooth_position(self, x, y):
        """Apply exponential smoothing to cursor position."""
        self.smooth_x = self.smooth_x * SMOOTHING_FACTOR + x * (1 - SMOOTHING_FACTOR)
        self.smooth_y = self.smooth_y * SMOOTHING_FACTOR + y * (1 - SMOOTHING_FACTOR)
        return self.smooth_x, self.smooth_y
    
    def handle_gestures(self, landmarks, frame_width, frame_height):
        """Process hand gestures and update state."""
        current_time = time.time()
        pinching = is_pinching(landmarks, frame_width, frame_height)
        fist = is_fist(landmarks)
        
        # Get index finger position
        index_x, index_y = get_index_position(landmarks, frame_width, frame_height)
        screen_x, screen_y = map_to_screen(
            index_x, index_y,
            frame_width, frame_height,
            self.screen_width, self.screen_height
        )
        
        # State machine logic
        if self.state == CursorState.INACTIVE:
            # Check for double pinch to activate
            if pinching and not self.was_pinching:
                if current_time - self.last_pinch_time < DOUBLE_PINCH_TIME:
                    self.pinch_count += 1
                else:
                    self.pinch_count = 1
                self.last_pinch_time = current_time
                
                if self.pinch_count >= 2:
                    self.state = CursorState.ACTIVE
                    self.pinch_count = 0
                    self.smooth_x, self.smooth_y = screen_x, screen_y
                    print(">>> CURSOR ACTIVATED <<<")
                    
        elif self.state == CursorState.ACTIVE:
            # Check for closed palm to deactivate
            if fist:
                self.state = CursorState.INACTIVE
                self.pinch_count = 0
                print(">>> CURSOR DEACTIVATED <<<")
                
            # Check for pinch (click or drag)
            elif pinching:
                if not self.was_pinching:
                    self.pinch_start_time = current_time
                    
                # Check if held long enough for drag
                if current_time - self.pinch_start_time > DRAG_HOLD_TIME:
                    if self.state != CursorState.DRAGGING:
                        self.state = CursorState.DRAGGING
                        mouse.press(Button.left)
                        print(">>> DRAG STARTED <<<")
                else:
                    self.state = CursorState.CLICKING
            else:
                # Pinch released
                if self.was_pinching:
                    if self.state == CursorState.CLICKING:
                        mouse.click(Button.left)
                        print(">>> CLICK <<<")
                    elif self.state == CursorState.DRAGGING:
                        mouse.release(Button.left)
                        print(">>> DRAG ENDED <<<")
                    self.state = CursorState.ACTIVE
            
            # Move cursor (when active or dragging)
            if self.state in [CursorState.ACTIVE, CursorState.CLICKING, CursorState.DRAGGING]:
                smooth_x, smooth_y = self.update_smooth_position(screen_x, screen_y)
                mouse.position = (int(smooth_x), int(smooth_y))
                
        elif self.state == CursorState.DRAGGING:
            # Continue dragging
            if fist:
                mouse.release(Button.left)
                self.state = CursorState.INACTIVE
                print(">>> DRAG CANCELLED <<<")
            elif not pinching:
                mouse.release(Button.left)
                self.state = CursorState.ACTIVE
                print(">>> DRAG ENDED <<<")
            else:
                smooth_x, smooth_y = self.update_smooth_position(screen_x, screen_y)
                mouse.position = (int(smooth_x), int(smooth_y))
        
        self.was_pinching = pinching
        return screen_x, screen_y
    
    def get_state_text(self):
        """Get display text for current state."""
        states = {
            CursorState.INACTIVE: "INACTIVE (Double pinch to activate)",
            CursorState.ACTIVE: "ACTIVE (Pinch=click, Fist=deactivate)",
            CursorState.CLICKING: "CLICKING...",
            CursorState.DRAGGING: "DRAGGING..."
        }
        return states.get(self.state, "UNKNOWN")
    
    def get_state_color(self):
        """Get display color for current state."""
        colors = {
            CursorState.INACTIVE: (128, 128, 128),
            CursorState.ACTIVE: (0, 255, 0),
            CursorState.CLICKING: (0, 255, 255),
            CursorState.DRAGGING: (0, 165, 255)
        }
        return colors.get(self.state, (255, 255, 255))
    
    def run(self):
        """Main loop."""
        print("=" * 50)
        print("Hand Mouse Controller Started!")
        print(f"Screen: {self.screen_width}x{self.screen_height}")
        print("Press 'q' to quit")
        print("=" * 50)
        print("\nInstructions:")
        print("  1. Double pinch (thumb+index x2) to ACTIVATE")
        print("  2. Move hand to move cursor")
        print("  3. Pinch once to CLICK")
        print("  4. Pinch and hold to DRAG")
        print("  5. Close fist to DEACTIVATE")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                
                # Convert to RGB for Mediapipe
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
                
                # Draw state indicator
                state_text = self.get_state_text()
                state_color = self.get_state_color()
                cv2.putText(frame, state_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Draw activation circle
                circle_color = (0, 255, 0) if self.state != CursorState.INACTIVE else (0, 0, 255)
                cv2.circle(frame, (frame_width - 30, 30), 15, circle_color, -1)
                
                # Show frame
                cv2.imshow("Hand Mouse Controller", frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.state == CursorState.DRAGGING:
            mouse.release(Button.left)
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nHand Mouse Controller stopped.")

# ============== ENTRY POINT ==============
if __name__ == "__main__":
    controller = HandMouseApp()
    controller.run()
