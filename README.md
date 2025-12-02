# Eye Tracking Application

A lightweight eye-tracking program using OpenCV. This application captures webcam input and performs real-time gaze detection. The user gaze is then used to move the cursor.

---
# Instruction
Make you have installed all of the necessary libraries. Run eyetracking.py

If this is the first time you've ran the program, you will be prompted with a calibartion window, please follow the instructions.

Keybinds:
- q : quit
- r : reset calibration data, you will be prompted with a new calibartion window.
- (more functionalities coming soon)

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
User must be in root dir.
```bash
docker build -t eyetracking .
```

### Run program
The user must map their webcam device to the container using the --device flag. This is what enables the eyetracking.py to use OpenCV to find the camera.
```bash
docker run --device=/dev/video0:/dev/video0 -it eyetracking
```

