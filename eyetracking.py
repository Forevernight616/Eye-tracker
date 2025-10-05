import mediapipe as mp
import cv2
import numpy as np
import os
import pyautogui
from calibration import EyeTracker

pyautogui.FAILSAFE = False


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
    between_brows= face_landmark.landmark[9]
    lowest_nose = face_landmark.landmark[6]
    nose_tip = face_landmark.landmark[4]
    cv2.circle(image, (int(between_brows.x * frame_width),int(between_brows.y * frame_height)), 1, (0,255,0), thickness=4)
    cv2.circle(image, (int(lowest_nose.x * frame_width),int(lowest_nose.y * frame_height)), 1, (0,255,0), thickness=4) 
    cv2.circle(image, (int(nose_tip.x * frame_width),int(nose_tip.y * frame_height)), 1, (0,255,0), thickness=4)

def draw_calibration_data(image: np.ndarray, relative_iris_landmarks) -> None:
    '''
    Draw the calibration points extracted from the json file, values only 
    Args : image : openCV frame
        calibration_data_values : list of calibration data points extracted from the json file, values only'''
    frame_height, frame_width, _ = image.shape
    for point in relative_iris_landmarks.values():
         # Highlight current point in red
        cv2.circle(image, (int(point["left_iris"][0] * frame_width), int(point["left_iris"][1] * frame_height) ), 1, (0, 255, 0), thickness=4)
        cv2.circle(image, (int(point["right_iris"][0] * frame_width), int(point["right_iris"][1] * frame_height) ), 1, (0, 255, 0), thickness=4)

def show_calibration_window(image: np.ndarray, current_point_index: int, screen_calibration_points: list) -> None:
    '''
    Show the calibration window with the current point
    Args : image : openCV frame
        current_point_index: int representing the current point the eyes should look at
        screen_calibration_points: list of tuples representing the point the eyes should look at
    '''
    cv2.putText(image, f"Calibrating: Point {current_point_index + 1}/9", 
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cal_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8) # create a black screen 
    if current_point_index < 9:
        point = screen_calibration_points[current_point_index]
        cv2.circle(cal_screen, point, 20, (0, 0, 255), -1)
        cv2.putText(cal_screen, f"Look at the RED dot and press SPACE ({current_point_index + 1}/9)", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Calibration', cal_screen)

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

def getNoseCoords(face_landmark) -> dict:
    '''
    Get the normalized nose coordinates from the face landmarks
    Args : face_landmark: the face landmarks from MediaPipe
    Returns : {'between_brows': (float, float),   'lowest_nose': (float, float)}
    '''
    between_brows= face_landmark.landmark[9]
    lowest_nose = face_landmark.landmark[6] # fun fact, the dip on your nose between the eyes is called radix   
    return {
        'between_brows': (between_brows.x, between_brows.y),
        'lowest_nose': (lowest_nose.x, lowest_nose.y)
    }

def main():
    capture = cv2.VideoCapture(0)
    tracker = EyeTracker()

    #make sure the cam can be opened
    if not capture.isOpened():
        print("cannot open camera")
        exit()  

    #calibration setup
    screen_calibration_points = tracker.get_calibration_points()
    current_point_index = 0 

    #creating a calibration window  # potentially removable
    if tracker.is_calibrated == False : 
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Error: could not read frame")
            break

        # Convert the frame to RGB for MediaPipe 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        result = face_mesh_model.process(rgb_frame)

        if result.multi_face_landmarks: # detected a face at index 0
            draw_eyes_landmarks(frame, result.multi_face_landmarks[0])
            iris_coords = getIrisCoords(result.multi_face_landmarks[0])
            nose_coords = getNoseCoords(result.multi_face_landmarks[0]) 
            if tracker.is_calibrated and iris_coords:
                relative_iris_landmarks = tracker.converter(nose_coords)
                draw_calibration_data(frame, relative_iris_landmarks) #deletable
                screen_coords = tracker.iris_to_screen_coords(iris_coords['left'], iris_coords['right'],relative_iris_landmarks)
                cv2.putText(frame, f"Estimated Screen Coords: x={int(screen_coords[0])}, y={int(screen_coords[1])}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Move the mouse cursor to the estimated screen coordinates
                pyautogui.moveTo(int(screen_coords[0]), int(screen_coords[1]), duration=0.1)

        else : 
            cv2.putText(frame, "No face detected", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        if tracker.is_calibrated :
            cv2.putText(frame, "Calibration complete! Tracking gaze...", 
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            show_calibration_window(frame, current_point_index, screen_calibration_points)

        cv2.imshow('Video', frame)


    #   # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset calibration
            tracker.calibration_data = {}
            tracker.is_calibrated = False
            is_calibrating = True
            current_point_index = 0
            if os.path.exists(tracker.calibration_file):
                os.remove(tracker.calibration_file)
            print("Calibration reset!")
            print("NOT YET IMPLEMENTED")
        elif key == ord(' ') and not tracker.is_calibrated and iris_coords and current_point_index < 9:
            # Calibrate current point
            current_point = screen_calibration_points[current_point_index]
            tracker.add_calibration_point(iris_coords, nose_coords, current_point)
            current_point_index += 1
            
            #if all points done
            if current_point_index >= 9:
                tracker.is_calibrated = True
                tracker.save_calibration()
                cv2.destroyWindow('Calibration')

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  
