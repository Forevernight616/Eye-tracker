import mediapipe as mp
import cv2
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pupil import Pupil

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # face mesh 3
mp_drawing_styles = mp.solutions.drawing_styles
draw_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# The files used
script_dir = Path(__file__)
face = str(script_dir.parent / "Photos" / "lady_1.jpg")
face_long = str(script_dir.parent / "Videos" / "face_long.mp4")
haar_casade_data = str(script_dir.parent / "haar_face.xml")

def get_landmark(image):
    face_mesh=mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                    refine_landmarks=True, min_detection_confidence=0.5)
    image.flags.writeable = True
    # face_mesh.process(image)
    result = face_mesh.process(image)
    if result.multi_face_landmarks != None:
        mjr_landmarks = result.multi_face_landmarks[0].landmark
    else : 
        mjr_landmarks = None   
    return result, mjr_landmarks

def draw_landmarks(image, result, frame_width, frame_height):
    image.flags.writeable = True
    if result.multi_face_landmarks : 
        for face_landmark in result.multi_face_landmarks:
            left_iris = face_landmark.landmark[473]
            right_iris = face_landmark.landmark[468]
            cv2.circle(frame, (int(left_iris.x * frame_width),int(left_iris.y * frame_height)), 1, (255,100,0), thickness=4) # visuallize the iris center
            cv2.circle(frame, (int(right_iris.x * frame_width),int(right_iris.y * frame_height)), 1, (255,100,0), thickness=4) # visuallize the iris center
            cv2.putText(frame, "left x: {:.4}, y: {:.4}".format(left_iris.x, left_iris.y), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "right x: {:.4}, y: {:.4}".format(right_iris.x, right_iris.y), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

capture = cv2.VideoCapture(0)

#make sure the cam can be opened
if not capture.isOpened():
    print("cannot open camera")
    exit()  

while True:
    isTrue, frame = capture.read()
    results, mjr_landmarks = get_landmark(frame)
    dimensions = frame.shape # For visualization purpose
    frame_height = dimensions[0]
    frame_width = dimensions[1]
    if mjr_landmarks is not None:
        frame_v2= draw_landmarks(frame, results, frame_width, frame_height)
    else : 
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv2.destroyAllWindows()  
