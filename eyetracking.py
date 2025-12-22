import mediapipe as mp
import cv2
import time
import pyautogui
import keyboard
from calibration import EyeTracker
from KalmanFilter import KalmanFilter
from Controller import Controller
from utilities.calibration_and_image_processing_util import *
from utilities.keyboard_behavior_util import *


from scipy.spatial.distance import euclidean

pyautogui.FAILSAFE = False 

#global setup for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # face mesh 3
mp_drawing_styles = mp.solutions.drawing_styles
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Landmark Indices for Gaze Feature Calculation
LEFT_IRIS_IDX = 473
RIGHT_IRIS_IDX = 468
# Eye Midpoint/Reference Indices (outer and inner corner for center reference)
LEFT_EYE_INNER_CORNER_IDX = 133
RIGHT_EYE_INNER_CORNER_IDX = 362
LEFT_EYE_OUTER_CORNER_IDX = 33
RIGHT_EYE_OUTER_CORNER_IDX = 263

# sdfe control movement speed
MOVE_SPEED = 20

#initialize the FaceMesh model 
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False, # video stream
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
    
def main():
    capture = cv2.VideoCapture(0)
    tracker = EyeTracker()
    kf = KalmanFilter(process_variance=1e-5, measurement_variance=0.8) # Initialize Kalman Filter
    controller = Controller()
    
    if not capture.isOpened():
        print("cannot open camera")
        exit()

    #Set up screen calibration
    screen_calibration_points = get_calibration_points()
    current_point_index = 0

    #Set up keyboard controller
    keyboard.on_press_key('i', lambda e: controller.enter_insert_mode(), suppress=False)
    keyboard.on_press_key('`', lambda e: controller.enter_normal_mode(), suppress=True)
    controller.enter_normal_mode()

    # Flag to ensure the SPACE key press only registers once per press
    space_key_was_pressed = False 
    
    if not tracker.is_calibrated:
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Error: could not read frame")
            break

        # initialize gaze tracking functinality
        face_mesh_image = apply_face_mesh_model(frame, face_mesh_model)
        if controller.should_use_eye_tracking() and face_mesh_image.multi_face_landmarks:
            feature_vector = get_gaze_feature_vector(face_mesh_image.multi_face_landmarks[0])
            track_gaze_and_move_cursor(tracker, kf, feature_vector)

        elif not face_mesh_image.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Video', frame)

            
        cv2.waitKey(1)
        if controller.in_insert_mode: #help with optimization
            time.sleep(0.08)

        if controller.should_exit: # User hits q
            break
        elif controller.should_reset_calibration: # User hit r
            reset_calibration_data(controller, tracker)
            current_point_index = 0
            kf = KalmanFilter(process_variance=1e-5, measurement_variance=0.8) # Re-initialize Kalman Filter on reset

        elif in_calibration_process(tracker, feature_vector, current_point_index):    
            show_calibration_window(frame, current_point_index, screen_calibration_points)
            is_space_pressed = keyboard.is_pressed('space')
            if is_space_pressed and not space_key_was_pressed:
                # Capture the calibration point only on the DOWN stroke
                current_point = screen_calibration_points[current_point_index]
                tracker.add_calibration_point(feature_vector, current_point)
                current_point_index += 1
                # If all points done
                if current_point_index >= 9:
                    tracker.train_mapping_model()
                    tracker.save_calibration()
                    cv2.destroyWindow('Calibration')
            
            # Update the debounce flag for the next iteration
            space_key_was_pressed = is_space_pressed

        # Handle keyboard inputs for cursor movement and scrolling
        manual_cursor_movement(controller, MOVE_SPEED)
        manual_cursor_scrolling(controller)

        time.sleep(0.01) # Sleep briefly to prevent busy-waiting
        
    keyboard.unhook_all()   
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
