import mediapipe as mp
import cv2
import numpy as np
import os
import pyautogui
from calibration import EyeTracker
from KalmanFilter import KalmanFilter
from scipy.spatial.distance import euclidean

import keyboard

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


#initialize the FaceMesh model 
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False, # video stream
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_gaze_feature_vector(face_landmark) -> np.ndarray | None:
    '''
    Calculates the normalized gaze feature vector for the calibration process
    Args: 
        face_landmark: the face landmarks from MediaPipe
        
    Returns:
        [avg_norm_gaze_x, avg_norm_gaze_y]
    '''
    try:
        # Get Landmarks (normalized [0, 1])
        l_iris = face_landmark.landmark[LEFT_IRIS_IDX]
        r_iris = face_landmark.landmark[RIGHT_IRIS_IDX]
        l_outer = face_landmark.landmark[LEFT_EYE_OUTER_CORNER_IDX]
        r_outer = face_landmark.landmark[RIGHT_EYE_OUTER_CORNER_IDX]
        l_inner = face_landmark.landmark[LEFT_EYE_INNER_CORNER_IDX]
        r_inner = face_landmark.landmark[RIGHT_EYE_INNER_CORNER_IDX]

        # This distance is a proxy for the face's scale/distance from the camera.
        inter_ocular_dist = euclidean([l_outer.x, l_outer.y], [r_outer.x, r_outer.y])

        # Left Eye Center Midpoint
        l_mid_x = (l_inner.x + l_outer.x) / 2
        l_mid_y = (l_inner.y + l_outer.y) / 2
        # Right Eye Center Midpoint
        r_mid_x = (r_inner.x + r_outer.x) / 2
        r_mid_y = (r_inner.y + r_outer.y) / 2
        # Gaze offset/Iris center relative to eye midpoint
        l_gaze_x = l_iris.x - l_mid_x
        l_gaze_y = l_iris.y - l_mid_y
        r_gaze_x = r_iris.x - r_mid_x
        r_gaze_y = r_iris.y - r_mid_y

        # Normalize by inter-ocular distance
        l_gaze_x_norm = l_gaze_x / inter_ocular_dist
        l_gaze_y_norm = l_gaze_y / inter_ocular_dist
        r_gaze_x_norm = r_gaze_x / inter_ocular_dist
        r_gaze_y_norm = r_gaze_y / inter_ocular_dist
        
        # Average the left and right normalized gaze features for stability
        avg_norm_gaze_x = (l_gaze_x_norm + r_gaze_x_norm) / 2
        avg_norm_gaze_y = (l_gaze_y_norm + r_gaze_y_norm) / 2

        return np.array([avg_norm_gaze_x, avg_norm_gaze_y], dtype=np.float64)
    except Exception as e:
        print("System error at get_gaze_feature_vector()")
        return None
    

def show_calibration_window(image: np.ndarray, current_point_index: int, screen_calibration_points: list) -> None:
    '''
    Show the calibration window with the current point
    Args : image : openCV frame
        current_point_index: int representing the current point the eyes should look at
        screen_calibration_points: list of tuples representing the point the eyes should look at
    '''
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cal_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    if current_point_index < 9:
        point = screen_calibration_points[current_point_index]
        cv2.circle(cal_screen, point, 20, (0, 0, 255), -1)
        cv2.putText(cal_screen, f"Look at the RED dot and press SPACE ({current_point_index + 1}/9)\nPlease keep your head as steady as possible", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(cal_screen, "Please keep your head as stable as possible!", 
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Calibration', cal_screen)

def main():
    capture = cv2.VideoCapture(0)
    tracker = EyeTracker()
    kf = KalmanFilter(process_variance=1e-5, measurement_variance=0.8) # Initialize Kalman Filter
    
    if not capture.isOpened():
        print("cannot open camera")
        exit()

    screen_calibration_points = tracker.get_calibration_points()
    current_point_index = 0

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

        # opencv uses BGR scheme, mediapipe uses RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh_model.process(rgb_frame)

        if result.multi_face_landmarks:
            face_landmark = result.multi_face_landmarks[0]
            feature_vector = get_gaze_feature_vector(face_landmark)

            if feature_vector is not None:
                if tracker.is_calibrated:
                    screen_coords = tracker.iris_to_screen_coords(feature_vector)
                    if screen_coords is not None:
                        # Prediction: where the cursor *should* be
                        kf.predict() 
                        # Update: correct prediction with the new noisy gaze data
                        smoothed_x, smoothed_y = kf.update(screen_coords)

                        # Move the mouse cursor to the estimated screen coordinates
                        pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
                # if is_calibrated==false, then it is in calibrating phase
        else:
            cv2.putText(frame, "No face detected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Video', frame)

        
        if not tracker.is_calibrated:
            show_calibration_window(frame, current_point_index, screen_calibration_points)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('r'):  # Reset calibration
            tracker.calibration_data = {}
            tracker.is_calibrated = False
            current_point_index = 0
            if os.path.exists(tracker.calibration_file):
                os.remove(tracker.calibration_file)
            print("Calibration reset!")
            # Re-initialize Kalman Filter on reset
            kf = KalmanFilter(process_variance=1e-5, measurement_variance=0.8)

        elif not tracker.is_calibrated and feature_vector is not None and current_point_index < 9:
            
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

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
