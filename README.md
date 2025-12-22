# Eye Tracking Application

Instead of having to move your hand from the keyboard to the mouse, or marginally move your thumb from the spacebar to the trackpad to control the cursor, this program solves that issue.

This is a lightweight eye-tracking program using OpenCV. This application captures webcam input and performs real-time gaze detection. The user gaze is then used to move the cursor, while still allowing the user to take over the cursor for more precise cursor control. Everything can be done on the keyboard, effectively removing the need for a mouse or a trackpad.

If you feel like this is a project that would benefit you, or the wider community, please feel free to contribute, or reach out to me for code clarification (this repo is kinda sloppy, so I expect a lot of questions). Thanks!

---
# Instruction
Make sure you have installed all of the necessary libraries. Run eyetracking.py

If this is the first time you've run the program, you will be prompted with a calibration window. Please follow the instructions.

Keybinds: The program has 2 modes

```
    Insert mode: frees up the whole keyboard for typing and the eyetracker stops
```
- i : enter insert mode

```
    Normal mode: the eyetracker works again and start capturing keys for cursor control
```
- ` (tilde, next to "1") : enter normal mode
- q : quit
- r : reset calibration data, you will be prompted with a new calibration window.

- s, d, f, e: each of these corresponds to a movement of the cursor (s-left, d-down, f-right, e-up).
- j : left click (hold and combine with sdfe for mouse drag effect)
- k : right click
- l : scroll up
- ; : scroll down


# Installation
## For Windows:
Note for Windows Users: Due to the complexity of reliably exposing a webcam to Docker on Windows, I recommend installing the dependencies manually.

Install Python 3.11

Navigate to the repository folder

```bash
pip install -r requirements.txt
```

```bash
python eyetracking.py
```

## For Linux: Run with Docker (Work in progress)
### Build image
User must be in the root dir.
```bash
docker build -t eyetracking .
```

### Run program
The user must map their webcam device to the container using the --device flag. This is what enables the eyetracking.py to use OpenCV to find the camera.
```bash
docker run --device=/dev/video0:/dev/video0 -it eyetracking
```

