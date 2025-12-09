# hand-tracking

Real-time hand gesture mouse controller using computer vision.

## Features

- **Mouse Movement**: Control cursor position with hand movements
- **Left Click**: Index finger bent gesture
- **Right Click**: Middle finger bent gesture
- **Double Click**: Both fingers bent
- **Screenshot**: Both fingers bent + thumb close to index
- **Drag & Drop**: Pinch and hold

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/hand-tracking.git
cd hand-tracking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Main Controller (Replicated from Deepakdj007)
```bash
python main.py
```

**Gestures:**
- Thumb+Index close + Index extended → Move cursor
- Index finger bent → Left click
- Middle finger bent → Right click
- Both fingers bent → Double click
- Both bent + Thumb close to index → Screenshot

### Alternative Controller
```bash
python hand_mouse_controller.py
```

**Gestures:**
- Double pinch (👌👌) → Activate cursor control
- Single pinch (👌) → Left click
- Pinch + hold + move → Click and drag
- Closed palm (✊) → Deactivate cursor

## Requirements

- Python 3.8+
- Webcam
- OpenCV
- MediaPipe
- PyAutoGUI
- pynput
- NumPy

## Controls

Press `q` to quit the application.

## License

MIT License
