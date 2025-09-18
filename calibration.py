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


    def add_calibration_point(self, iris_coords: dict, screen_coords : tuple) -> None:
        '''
        Add a calibration point
        Args : iris_coords = {'left': (float, float),   'right': (float, float)}
            screen_coords  = (screen_width,screen_height)
        '''
        key = f"{screen_coords[0]}_{screen_coords[1]}"
        self.calibration_data[key] = {
            'screen': screen_coords,
            'left_iris': iris_coords['left'],
            'right_iris': iris_coords['right']
        }

    def calibrate(self, iris_coords: dict, screen_coords: tuple) -> bool:
        '''
        Add calibration point and check if all 9 points calibration have been made
        Args : iris_coords = {'left': (float, float),   'right': (float, float)}
            screen_coords  = (screen_width,screen_height)
        '''
        self.add_calibration_point(iris_coords, screen_coords)
        if len(self.calibration_data) >= 9:
            self.is_calibrated = True
            self.save_calibration()
            print("Calibration complete")
            return True
        
        return False
    
    def iris_to_screen_coords(self, left_iris: tuple, right_iris: tuple) -> tuple | None:
        '''
        Translate the iris coordinates to screen coordinates using calibration data
        Args : left_iris, right_iris = (float, float)
        Returns : (screen_x, screen_y) or None if not calibrated
        '''

        if  not self.is_calibrated: # or len(self.calibration_data) < 9:
            print("Error: Not enough calibration points ( needs 9 in total)")
            return None
        
        # Use average of both eyes for more stability
        avg_iris_x = (left_iris[0] + right_iris[0]) / 2
        avg_iris_y = (left_iris[1] + right_iris[1]) / 2

        # prepare data for polunomial regression
        # Prepare data for polynomial regression
        iris_points = []
        screen_points = []
        
        for data in self.calibration_data.values():
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
        Load calibration data from file
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
    
        
    

