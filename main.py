"""
Live Mouse Control Using Hand Gestures
======================================
Replicated from: https://github.com/Deepakdj007/Computer-Vision/tree/main/live_mouse_control_using_hand_gestures

Gestures:
- Thumb+Index close together with index finger extended: Move mouse
- Index finger bent (pointing): Left click  
- Middle finger bent (pointing): Right click
- Both index and middle fingers bent: Double click
- Both fingers bent with thumb+index close: Screenshot

Requirements:
    pip install opencv-python mediapipe pyautogui numpy pynput
           
Usage:
    python main.py
    Press 'q' to quit
"""

import os
# Fix Qt platform issue on Wayland systems
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import mediapipe as mp
import pyautogui
import random
import util
from pynput.mouse import Button, Controller

mouse = Controller()

screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def find_finger_tip(processed):
    """Find the index finger tip from processed hand landmarks."""
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None


def move_mouse(index_finger_tip):
    """Move mouse cursor based on index finger tip position."""
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    """Check if gesture indicates a left click (index finger bent, middle extended)."""
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    """Check if gesture indicates a right click (middle finger bent, index extended)."""
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    """Check if gesture indicates a double click (both fingers bent)."""
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    """Check if gesture indicates screenshot (both fingers bent + thumb close to index)."""
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


def detect_gesture(frame, landmark_list, processed):
    """Detect and execute hand gestures."""
    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        # Move mouse when thumb and index are close together and index is extended
        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50 and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list, thumb_index_dist):
            try:
                im1 = pyautogui.screenshot()
                label = random.randint(1, 1000)
                im1.save(f'my_screenshot_{label}.png')
                cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            except Exception as e:
                cv2.putText(frame, "Screenshot Failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Screenshot error: {e}")


def main():
    """Main function to run the hand gesture mouse controller."""
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("=" * 50)
    print("Live Mouse Control Using Hand Gestures")
    print("=" * 50)
    print("\nGestures:")
    print("  - Thumb+Index close + Index extended: Move cursor")
    print("  - Index finger bent: Left click")
    print("  - Middle finger bent: Right click")
    print("  - Both fingers bent: Double click")
    print("  - Both bent + Thumb close to index: Screenshot")
    print("\nPress 'q' to quit")
    print("-" * 50)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
