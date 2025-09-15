import mediapipe as mp
import cv2

#global setup for MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh # face mesh 3
mp_drawing_styles = mp.solutions.drawing_styles

#initialize the FaceMesh model 
face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False, # video stream
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def draw_eyes_landmarks(image, face_landmark):
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
    
def main():
    capture = cv2.VideoCapture(0)

    # skip every alternate frame to improve performance
    skip_this_frame = False

    #make sure the cam can be opened
    if not capture.isOpened():
        print("cannot open camera")
        exit()  
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

        if result.multi_face_landmarks: # detected a face at index 0
            draw_eyes_landmarks(frame, result.multi_face_landmarks[0])
        else : 
            cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(20) & 0xFF==ord('d'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  
