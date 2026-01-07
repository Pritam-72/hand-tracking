# Hand-Tracking Mouse Controller 🖐️🖱️

Control your computer mouse with hand gestures using your webcam! This project uses MediaPipe for real-time hand tracking and translates natural hand movements into mouse actions.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

## ✨ Features

- **🖱️ Cursor Movement** - Move your hand to control the cursor
- **👆 Left Click** - Quick pinch gesture
- **🖐️ Right Click** - Open palm gesture
- **✊ Drag & Drop** - Pinch and hold, then move
- **📜 Scrolling** - Pinch + vertical movement
- **📸 Screenshot** - Three fingers + thumb gesture
- **⚙️ Configurable** - JSON config file for all settings
- **🔊 Audio Feedback** - Sound confirmation for actions
- **🎯 Cooldown Timers** - Prevents accidental repeated actions
- **💻 Cross-Platform** - Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Pritam-72/hand-tracking.git
cd hand-tracking

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Controller

```bash
python unified_controller.py
```

Press `q` to quit the application.

## 🎮 Gesture Reference

| Gesture | Action | Description |
|---------|--------|-------------|
| 👌👌 Double Pinch | **Activate** | Pinch twice quickly to start cursor control |
| ✋ Move Hand | **Move Cursor** | Move your hand to move the cursor |
| 👌 Single Pinch | **Left Click** | Quick pinch and release |
| 👌 Hold Pinch | **Drag** | Pinch and hold for 0.2s to start dragging |
| 🖐️ Open Palm | **Right Click** | Extend all fingers |
| 🤟 Three + Thumb | **Screenshot** | Index, middle, ring extended + thumb |
| ✊ Closed Fist | **Deactivate** | Close your hand to stop cursor control |

## ⚙️ Configuration

Edit `config.json` to customize behavior:

```json
{
    "camera": {
        "index": 0,           // Camera device index
        "width": 640,
        "height": 480
    },
    "cursor": {
        "smoothing_factor": 0.5,    // 0.0-1.0, higher = smoother
        "frame_reduction": 100      // Screen edge margin
    },
    "gestures": {
        "pinch_threshold": 40,      // Pinch detection sensitivity
        "drag_hold_time": 0.2       // Seconds to hold for drag
    },
    "cooldowns": {
        "click": 0.3,               // Seconds between clicks
        "screenshot": 1.0           // Seconds between screenshots
    },
    "feedback": {
        "enable_sound": true,
        "enable_visual": true
    }
}
```

## 📁 Project Structure

```
hand-tracking/
├── unified_controller.py   # 🎯 Main entry point (recommended)
├── main.py                  # Legacy controller (Deepakdj007 style)
├── hand_mouse_controller.py # Legacy controller (alternative)
├── util.py                  # Utility functions
├── config.json              # Configuration file
├── requirements.txt         # Dependencies
├── screenshots/             # Auto-saved screenshots
└── README.md
```

## 📜 Legacy Controllers

Two legacy controllers are still available:

### Original Controller (`main.py`)
```bash
python main.py
```
- Index finger bent → Left click
- Middle finger bent → Right click
- Both fingers bent → Double click
- Both bent + thumb close → Screenshot

### Alternative Controller (`hand_mouse_controller.py`)
```bash
python hand_mouse_controller.py
```
- Double pinch → Activate
- Single pinch → Left click
- Pinch + hold → Drag
- Closed fist → Deactivate

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Try changing `camera.index` in config.json |
| Cursor too sensitive | Increase `smoothing_factor` (max 0.9) |
| Clicks not registering | Decrease `pinch_threshold` |
| Accidental double-clicks | Increase `cooldowns.click` |
| No sound feedback | Check `feedback.enable_sound` in config |

## 📋 Requirements

- Python 3.8+
- Webcam
- Dependencies:
  - OpenCV (`opencv-python`)
  - MediaPipe
  - PyAutoGUI
  - pynput
  - NumPy

## 🙏 Credits

- Original concept inspired by [Deepakdj007/Computer-Vision](https://github.com/Deepakdj007/Computer-Vision)
- Hand tracking powered by [MediaPipe](https://mediapipe.dev/)

## 📄 License

MIT License - feel free to use and modify!

