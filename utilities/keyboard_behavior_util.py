import os 
import pyautogui
from Controller import Controller
from calibration import EyeTracker

def reset_calibration_data(controller: Controller, tracker:EyeTracker) -> None:
    controller.should_reset_calibration = False
    controller.enter_normal_mode()

    tracker.calibration_data = {}
    tracker.is_calibrated = False

    if os.path.exists(tracker.calibration_file):
        os.remove(tracker.calibration_file)
    print("Calibration reset!")

def manual_cursor_movement(controller: Controller, MOVE_SPEED) -> None:
    x_change = 0
    y_change = 0

    if controller.is_moving_left:
        x_change -= MOVE_SPEED
    if controller.is_moving_right:
        x_change += MOVE_SPEED
    if controller.is_moving_up:
        y_change -= MOVE_SPEED
    if controller.is_moving_down:
        y_change += MOVE_SPEED
        
    # Only call pyautogui.move() if a movement key is pressed
    if x_change != 0 or y_change != 0:
        pyautogui.move(x_change, y_change, duration=0)

def manual_cursor_scrolling(controller: Controller) -> None:
    # Handle scrolling
    if controller.is_scrolling_down:
        pyautogui.scroll(-controller.getScrollAmount())  # Negative scrolls down
    if controller.is_scrolling_up:
        pyautogui.scroll(controller.getScrollAmount())  # Positive scrolls up