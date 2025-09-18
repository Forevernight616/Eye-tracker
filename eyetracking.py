import mediapipe as mp
import cv2
import numpy as np
import os
from calibration import EyeTracker



#global setup for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # face mesh 3
mp_drawing_styles = mp.solutions.drawing_styles
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

#initialize the FaceMesh model 
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False, # video stream
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def draw_eyes_landmarks(image: np.ndarray, face_landmark) -> None:
    image.flags.writeable = True
    frame_height, frame_width, _ = image.shape
    left_iris = face_landmark.landmark[473]
    right_iris = face_landmark.landmark[468]

    cv2.circle(image, (int(left_iris.x * frame_width),int(left_iris.y * frame_height)), 1, (255,100,0), thickness=4) # visuallize the iris center
    cv2.circle(image, (int(right_iris.x * frame_width),int(right_iris.y * frame_height)), 1, (255,100,0), thickness=4) # visuallize the iris center
    cv2.putText(image, "left x: {:.4}, y: {:.4}".format(left_iris.x, left_iris.y), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, "right x: {:.4}, y: {:.4}".format(right_iris.x, right_iris.y), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #draw the nose's line
    # between_brows= face_landmark.landmark[9]
    # lowest_nose = face_landmark.landmark[6]
    # nose_tip = face_landmark.landmark[4]
    # cv2.circle(image, (int(between_brows.x * frame_width),int(between_brows.y * frame_height)), 1, (0,255,0), thickness=4)
    # cv2.circle(image, (int(lowest_nose.x * frame_width),int(lowest_nose.y * frame_height)), 1, (0,255,0), thickness=4) 
    # cv2.circle(image, (int(nose_tip.x * frame_width),int(nose_tip.y * frame_height)), 1, (0,255,0), thickness=4)


def getIrisCoords(face_landmark) -> dict:
    '''
    Get the normalized iris coordinates from the face landmarks
    Args : face_landmark: the face landmarks from MediaPipe
    Returns : {'left': (float, float),   'right': (float, float)}
    '''
    left_iris = face_landmark.landmark[473]
    right_iris = face_landmark.landmark[468]
    return {
        'left': (left_iris.x, left_iris.y),
        'right': (right_iris.x, right_iris.y)
    }

def show_calibration_instructions():
    """Show calibration instructions"""
    print("\n" + "="*50)
    print("EYE TRACKING CALIBRATION")
    print("="*50)
    print("1. Look at the RED DOT on your screen")
    print("2. Press SPACE when you're looking directly at it")
    print("3. The dot will move to 9 different positions")
    print("4. After calibration, your gaze will be tracked!")
    print("\nControls:")
    print("- SPACE: Calibrate current point")
    print("- 'r': Reset calibration")
    print("- 'q': Quit")
    print("="*50)

def create_calibration_window() -> list[tuple]:
    '''
    Create a fullscreen calibration window
    '''
    # Calibration points (9-point calibration)
    points = [
        (SCREEN_WIDTH//4, SCREEN_HEIGHT//4),           # Top-left
        (SCREEN_WIDTH//2, SCREEN_HEIGHT//4),           # Top-center
        (3*SCREEN_WIDTH//4, SCREEN_HEIGHT//4),         # Top-right
        (SCREEN_WIDTH//4, SCREEN_HEIGHT//2),           # Middle-left
        (SCREEN_WIDTH//2, SCREEN_HEIGHT//2),           # Center
        (3*SCREEN_WIDTH//4, SCREEN_HEIGHT//2),         # Middle-right
        (SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4),         # Bottom-left
        (SCREEN_WIDTH//2, 3*SCREEN_HEIGHT//4),         # Bottom-center
        (3*SCREEN_WIDTH//4, 3*SCREEN_HEIGHT//4),       # Bottom-right
    ]
    return points



def main():
    capture = cv2.VideoCapture(0)
    tracker = EyeTracker()
    # skip every alternate frame to improve performance
    skip_this_frame = False

    #make sure the cam can be opened
    if not capture.isOpened():
        print("cannot open camera")
        exit()  

    #calibration setup
    calibration_points = create_calibration_window()
    current_point_index = 0
    calibrating = not tracker.is_calibrated

    show_calibration_instructions()
    #creating a calibration window
    if calibrating : 
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Error: could not read frame")
            break
        
        if skip_this_frame:
            skip_this_frame = False
            continue
        # Convert the frame to RGB for MediaPipe 
        skip_this_frame = True
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        result = face_mesh_model.process(rgb_frame)

        iris_coords = None
        screen_coords = None

        if result.multi_face_landmarks: # detected a face at index 0
            draw_eyes_landmarks(frame, result.multi_face_landmarks[0])
            iris_coords = getIrisCoords(result.multi_face_landmarks[0])

            if tracker.is_calibrated and iris_coords:
                screen_coords = tracker.iris_to_screen_coords(iris_coords['left'], iris_coords['right'])
                if screen_coords:
                    cv2.putText(frame, f"Estimated Screen Coords: x={int(screen_coords[0])}, y={int(screen_coords[1])}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else : 
            cv2.putText(frame, "No face detected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        if calibrating:
            
            cv2.putText(frame, f"Calibrating: Point {current_point_index + 1}/9", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show calibration screen
            cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cal_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            if current_point_index < len(calibration_points):
                point = calibration_points[current_point_index]
                cv2.circle(cal_screen, point, 20, (0, 0, 255), -1)
                cv2.putText(cal_screen, f"Look at the RED dot and press SPACE ({current_point_index + 1}/9)", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Calibration', cal_screen)

        elif tracker.is_calibrated:
            cv2.putText(frame, "Calibration complete! Tracking gaze...", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        cv2.imshow('Video', frame)
      # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset calibration
            tracker.calibration_data = {}
            tracker.is_calibrated = False
            calibrating = True
            current_point_index = 0
            if os.path.exists(tracker.calibration_file):
                os.remove(tracker.calibration_file)
            print("Calibration reset!")
        elif key == ord(' ') and calibrating and iris_coords and current_point_index < len(calibration_points):
            # Calibrate current point
            current_point = calibration_points[current_point_index]
            tracker.add_calibration_point(iris_coords, current_point)
            current_point_index += 1
            
            if current_point_index >= len(calibration_points):
                tracker.is_calibrated = True
                tracker.save_calibration()
                calibrating = False
                cv2.destroyWindow('Calibration')
                print("Calibration complete! Eye tracking is now active.")


    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  
