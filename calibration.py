import math
import mediapipe as mp
import numpy as np
import json
import os

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

class EyeTracker:
    def __init__(self):
        self.calibration_points = []
        self.calibration_data = {}
        self.is_calibrated = False
        self.calibration_file = "eye_calibration.json"
        self.load_calibration()


    def add_calibration_point(self, iris_coords: dict, nose_coords: dict, screen_coords : tuple) -> None:
        '''
        Add a calibration point
        Args : iris_coords = {'left': (float, float),   'right': (float, float)}
            nose_coords = {'between_brows': (float, float),'lowest_nose': (float ,float)}
            screen_coords  = (x coords : int ,y coords : int)
        '''
        between_brows_x, between_brows_y = nose_coords['between_brows']
        lowest_nose_x, lowest_nose_y = nose_coords['lowest_nose']
        iris_left_x, iris_left_y = iris_coords['left']
        iris_right_x, iris_right_y = iris_coords['right']

        dis_irisLeft_betweenBrows = math.hypot(between_brows_x - iris_left_x, between_brows_y - iris_left_y)
        dis_irisRight_betweenBrows = math.hypot(between_brows_x - iris_right_x, between_brows_y - iris_right_y)
        dis_irisLeft_lowestNose = math.hypot(lowest_nose_x - iris_left_x, lowest_nose_y - iris_left_y)
        dis_irisRight_lowestNose = math.hypot(lowest_nose_x - iris_right_x, lowest_nose_y - iris_right_y)

        # Store iris distance to nose landmarks
        key = f"{screen_coords[0]}_{screen_coords[1]}"
        self.calibration_data[key] = {
            'screen': screen_coords,
            'left_iris': [dis_irisLeft_betweenBrows, dis_irisLeft_lowestNose],
            'right_iris': [dis_irisRight_betweenBrows, dis_irisRight_lowestNose]
        }

    def converter(self, nose_coords: dict) -> dict:
        '''
        Convert json data of relative distances to actual iris calibration landmark as normalized (x,y) points
        Args : nose_coords = {'between_brows': (float, float),'lowest_nose': (float ,float)}
        Returns : relative_iris_landmarks : dict that follows the same structure as self.calibration_data but with (x,y) coordinates instead of distances
        '''
        
        between_brows_x, between_brows_y = nose_coords['between_brows']
        lowest_nose_x, lowest_nose_y = nose_coords['lowest_nose']
        relative_iris_landmarks = {}
        # vector p1->p2
        dx, dy = lowest_nose_x - between_brows_x, lowest_nose_y - between_brows_y
        d = math.hypot(dx, dy)
        for key, data in self.calibration_data.items():

            # distances from nose landmarks to iris
            d1 = data['left_iris'][0]
            d2 = data['left_iris'][1]
            # distance from p1 to the base point along the line
            a = (d1**2 - d2**2 + d**2) / (2 * d)
            h = math.sqrt(max(d1**2 - a**2, 0))
            base_x = between_brows_x + a * dx / d
            base_y = between_brows_y + a * dy / d

            # perpendicular vector
            perp_x = -dy / d
            perp_y = dx / d

            # reconstruct left iris
            left_iris_X = base_x + (-1) * h * perp_x
            left_iris_y = base_y + (-1) * h * perp_y


            # distances from nose landmarks to right iris
            d1 = data['right_iris'][0]
            d2 = data['right_iris'][1]
            # distance from p1 to the base point along the line
            a = (d1**2 - d2**2 + d**2) / (2 * d)
            h = math.sqrt(max(d1**2 - a**2, 0))
            base_x = between_brows_x + a * dx / d
            base_y = between_brows_y + a * dy / d

            # perpendicular vector
            perp_x = -dy / d
            perp_y = dx / d

            # reconstruct p3
            right_iris_x = base_x + (1) * h * perp_x
            right_iris_y = base_y + (1) * h * perp_y

            relative_iris_landmarks[key] = {
                'screen': data['screen'],
                'left_iris': [left_iris_X, left_iris_y],
                'right_iris': [right_iris_x, right_iris_y]
            }

        return relative_iris_landmarks



    def iris_to_screen_coords(self, left_iris: tuple, right_iris: tuple, relative_iris_landmarks: dict) -> tuple | None:
        '''
        Translate the iris coordinates to screen coordinates using calibration data
        Args : left_iris = (float, float), live feeds from the webcam
            right_iris = (float, float), live feeds from the webcam
        Returns : (screen_x, screen_y) or None if not calibrated
        '''
        if not self.is_calibrated: # or len(self.calibration_data) < 9:
            print("Error: Not enough calibration points ( needs 9 in total)")
            return None
        
        # Use average of both eyes for more stability
        avg_iris_x = (left_iris[0] + right_iris[0]) / 2
        avg_iris_y = (left_iris[1] + right_iris[1]) / 2

        # Prepare data for polynomial regression
        iris_points = []
        screen_points = []
        
        for data in relative_iris_landmarks.values():
            # average the iris coordinates from calibration
            cal_avg_x = (data['left_iris'][0] + data['right_iris'][0]) / 2
            cal_avg_y = (data['left_iris'][1] + data['right_iris'][1]) / 2
            
            iris_points.append([cal_avg_x, cal_avg_y])
            screen_points.append(data['screen'])
        
        iris_points = np.array(iris_points)
        screen_points = np.array(screen_points)
        
        try:
            # Use polynomial fitting for more accurate mapping
            # Fit separate polynomials for X and Y coordinates
            
            # For X coordinate
            poly_x = np.polyfit(iris_points[:, 0], screen_points[:, 0], min(2, len(iris_points) - 1))
            screen_x = np.polyval(poly_x, avg_iris_x)
            
            # For Y coordinate 
            poly_y = np.polyfit(iris_points[:, 1], screen_points[:, 1], min(2, len(iris_points) - 1))
            screen_y = np.polyval(poly_y, avg_iris_y)
            
            # Handling out-of-bounds
            screen_x = max(0, min(SCREEN_WIDTH, screen_x))
            screen_y = max(0, min(SCREEN_HEIGHT, screen_y))
            
            return (int(screen_x), int(screen_y))
            
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return None

    def save_calibration(self) -> None:
        '''
        saving the data into eye_calibration.json"
        Args : calibration_data(dict)
        '''
        with open(self.calibration_file, "w") as file:
            json.dump(self.calibration_data, file)

    def load_calibration(self):
        '''
        Immediately load calibration data from file if it exists
        '''
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)
                    if len(self.calibration_data) >= 9:  # Need at least 9 points for good calibration
                        self.is_calibrated = True
                        print("Calibration data loaded successfully!")
            except:
                print("Error loading calibration data")



    @staticmethod
    def get_calibration_points() -> list[tuple]:
        '''
        return a list of 9 calibration points (x, y) tuples
        '''
        # Calibration points (9-point calibration)
        points = [
            (SCREEN_WIDTH//6, SCREEN_HEIGHT//6),           # Top-left
            (SCREEN_WIDTH//2, SCREEN_HEIGHT//6),           # Top-center
            (5*SCREEN_WIDTH//6, SCREEN_HEIGHT//6),         # Top-right
            (SCREEN_WIDTH//6, SCREEN_HEIGHT//2),           # Middle-left
            (SCREEN_WIDTH//2, SCREEN_HEIGHT//2),           # Center
            (5*SCREEN_WIDTH//6, SCREEN_HEIGHT//2),         # Middle-right
            (SCREEN_WIDTH//6, 5*SCREEN_HEIGHT//6),         # Bottom-left
            (SCREEN_WIDTH//2, 5*SCREEN_HEIGHT//6),         # Bottom-center
            (5*SCREEN_WIDTH//6, 5*SCREEN_HEIGHT//6),       # Bottom-right
        ]
        return points
    


    # unnecessary console print function used in eyetracking.py 
    @staticmethod 
    def show_calibration_instructions():
        '''Show calibration instructions'''
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


