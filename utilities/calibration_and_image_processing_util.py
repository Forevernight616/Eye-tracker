import numpy as np
import cv2
import mediapipe as mp
import pyautogui
from KalmanFilter import KalmanFilter
from calibration import EyeTracker
from scipy.spatial.distance import euclidean

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

def get_calibration_points() -> list[tuple]:
    # Calibration points (9-point calibration)
    points = [
        (SCREEN_WIDTH//6, 0),
        (SCREEN_WIDTH//2, 0),
        (5*SCREEN_WIDTH//6, 0),
        (SCREEN_WIDTH//6, SCREEN_HEIGHT//2),
        (SCREEN_WIDTH//2, SCREEN_HEIGHT//2),
        (5*SCREEN_WIDTH//6, SCREEN_HEIGHT//2),
        (SCREEN_WIDTH//6, 5*SCREEN_HEIGHT//6),
        (SCREEN_WIDTH//2, 5*SCREEN_HEIGHT//6),
        (5*SCREEN_WIDTH//6, 5*SCREEN_HEIGHT//6),
    ]
    return points


def apply_face_mesh_model(frame: np.ndarray, face_mesh_model: mp.solutions.face_mesh.FaceMesh) -> np.ndarray:
    '''Applies the face mesh model to the input frame and returns the processed result.'''
    # opencv uses BGR scheme, mediapipe uses RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return face_mesh_model.process(rgb_frame)

def track_gaze_and_move_cursor(tracker: EyeTracker, kf: KalmanFilter, feature_vector: np.ndarray) -> None:
    '''Attempts to move the cursor based on the processed result, tracker, and Kalman filter.'''
    if feature_vector is not None and tracker.is_calibrated:
        screen_coords = tracker.iris_to_screen_coords(feature_vector)
        if screen_coords is not None:
            # Prediction: where the cursor *should* be
            kf.predict() 
            # Update: correct prediction with the new noisy gaze data
            smoothed_x, smoothed_y = kf.update(screen_coords)

            # Move the mouse cursor to the estimated screen coordinates
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))

def in_calibration_process(tracker: EyeTracker, feature_vector: np.ndarray, current_point_index: int) -> bool:
    return not tracker.is_calibrated and feature_vector is not None and current_point_index < 9
