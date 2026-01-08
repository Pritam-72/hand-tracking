"""
3D Hand Visualization Server
============================
WebSocket server that streams hand landmarks to a Three.js web client
for creative 3D visualization around the hand.

Usage:
    python 3d_hand_server.py
    Then open http://localhost:8000 in your browser

Press 'q' in the OpenCV window to quit.
"""

import asyncio
import json
import threading
import webbrowser
from pathlib import Path

import cv2
import mediapipe as mp
from http.server import HTTPServer, SimpleHTTPRequestHandler
import websockets

# ============== CONFIGURATION ==============
WEBSOCKET_PORT = 8765
HTTP_PORT = 8000
WEB_DIR = Path(__file__).parent / "web"

# ============== MEDIAPIPE SETUP ==============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# ============== GLOBAL STATE ==============
latest_landmarks = None
landmarks_lock = threading.Lock()
running = True


class WebHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve from web directory."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)
    
    def log_message(self, format, *args):
        # Suppress logging for cleaner output
        pass


def run_http_server():
    """Run HTTP server in background thread."""
    server = HTTPServer(('localhost', HTTP_PORT), WebHandler)
    print(f"[WEB] Server running at http://localhost:{HTTP_PORT}")
    while running:
        server.handle_request()


async def websocket_handler(websocket):
    """Handle WebSocket connections and stream landmarks."""
    global latest_landmarks
    print("[+] Web client connected!")
    
    try:
        while running:
            with landmarks_lock:
                if latest_landmarks is not None:
                    await websocket.send(json.dumps(latest_landmarks))
            await asyncio.sleep(0.016)  # ~60 FPS
    except websockets.exceptions.ConnectionClosed:
        print("[-] Web client disconnected")


async def run_websocket_server():
    """Run WebSocket server."""
    async with websockets.serve(websocket_handler, "localhost", WEBSOCKET_PORT):
        print(f"[WS] WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")
        while running:
            await asyncio.sleep(0.1)


def process_landmarks(hand_landmarks, frame_width, frame_height):
    """Convert MediaPipe landmarks to JSON-serializable format."""
    landmarks = []
    
    for idx, lm in enumerate(hand_landmarks.landmark):
        landmarks.append({
            "id": idx,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            # Pixel coordinates for reference
            "px": int(lm.x * frame_width),
            "py": int(lm.y * frame_height)
        })
    
    # Calculate palm center (average of wrist and finger bases)
    palm_indices = [0, 5, 9, 13, 17]  # Wrist and MCP joints
    palm_x = sum(landmarks[i]["x"] for i in palm_indices) / len(palm_indices)
    palm_y = sum(landmarks[i]["y"] for i in palm_indices) / len(palm_indices)
    palm_z = sum(landmarks[i]["z"] for i in palm_indices) / len(palm_indices)
    
    return {
        "landmarks": landmarks,
        "palm": {"x": palm_x, "y": palm_y, "z": palm_z},
        "fingertips": {
            "thumb": landmarks[4],
            "index": landmarks[8],
            "middle": landmarks[12],
            "ring": landmarks[16],
            "pinky": landmarks[20]
        },
        "frame": {"width": frame_width, "height": frame_height}
    }


def run_camera():
    """Main camera loop with hand tracking."""
    global latest_landmarks, running
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "=" * 50)
    print("[HAND] 3D Hand Visualization Server")
    print("=" * 50)
    print("\nOpen http://localhost:8000 in your browser")
    print("Show your hand to see the 3D effects!")
    print("\nPress 'q' to quit")
    print("-" * 50 + "\n")
    
    # Open browser automatically
    webbrowser.open(f"http://localhost:{HTTP_PORT}")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Update global landmarks for WebSocket
                with landmarks_lock:
                    latest_landmarks = process_landmarks(
                        hand_landmarks, frame_width, frame_height
                    )
        else:
            with landmarks_lock:
                latest_landmarks = None
        
        # Add status overlay
        cv2.putText(
            frame, "3D Visualization Active", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Open browser to see 3D effects", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        cv2.imshow("Hand Tracking - Camera View", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
    
    cap.release()
    cv2.destroyAllWindows()


async def main():
    """Main entry point."""
    global running
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()
    
    # Start WebSocket server in background task
    websocket_task = asyncio.create_task(run_websocket_server())
    
    # Run camera in executor to not block async loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_camera)
    
    running = False
    websocket_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
