import numpy as np
import json
import os

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

class EyeTracker:
    def __init__(self):
        self.calibration_points = []
        self.calibration_data = {} # refers to the x_y of the eyes when stairing at the calibration points
        self.is_calibrated = False
        self.calibration_file = "eye_calibration.json"
        self.POLY_DEGREE=2
        # The coefficients for the polynomial mapping (X and Y separately). f(x)=ax^2 + bx + c
        self.poly_x_coeffs = None
        self.poly_y_coeffs = None
        self.load_calibration()
        
        

    def add_calibration_point(self, feature_vector: np.ndarray, screen_coords: tuple) -> None:
        '''
        Add a calibration point using a normalized feature vector (values from -1 to 1).

        Args:
            screen_coords  = (x coords : int ,y coords : int)
            feature_vector = [avg_norm_gaze_x, avg_norm_gaze_y]
        '''
        key = f"{screen_coords[0]}_{screen_coords[1]}"
        self.calibration_data[key] = {
            "screen" : screen_coords,
            "feature": feature_vector.tolist(),
        }
    
    def train_mapping_model(self):
        '''
        Trains the 2nd-degree polynomial regression model using all calibration data.
        Called once after calibration is complete.
        '''
        if len(self.calibration_data) < self.POLY_DEGREE + 1:
            print("Not enough data point for train_mapping_model()")
            self.is_calibrated = False
            return
        
        # set up arrays for polynomial regression
        feature_points = []
        screen_x_points = []
        screen_y_points = []
        
        for data in self.calibration_data.values():
            feature_points.append(data['feature'])
            screen_x_points.append(data['screen'][0])
            screen_y_points.append(data['screen'][1])
        
        feature_points = np.array(feature_points)
        screen_x_points = np.array(screen_x_points)
        screen_y_points = np.array(screen_y_points)

        try:
            # fit separate polynomials for X and Y coordinates (2nd degree)
            self.poly_x_coeffs = np.polyfit(feature_points[:, 0], screen_x_points, self.POLY_DEGREE)
            self.poly_y_coeffs = np.polyfit(feature_points[:, 1], screen_y_points, self.POLY_DEGREE)
            
            # update calibration
            self.is_calibrated = True
            print("Calibration model trained successfully with degree {} polynomial.".format(self.POLY_DEGREE))
        except Exception as e:
            print(f"Error training polynomial model: {e}")
            self.is_calibrated = False

    def iris_to_screen_coords(self, feature_vector: np.ndarray) -> tuple|None:
        '''
        Convert the normalized (from -1 to 1) feature_vectors into on screen coordinates using the trained model
        Args: 
            feature_vector = [avg_norm_gaze_x, avg_norm_gaze_y] (live feed)
        Returns:
            (float screen_x, float screen_y) or None if not calibrated  
            returned value as float for kalman filter
        '''
        if not self.is_calibrated or self.poly_x_coeffs is None:
            return None
        avg_norm_gaze_x, avg_norm_gaze_y = feature_vector

        try:
            # x and y predictions
            screen_x = np.polyval(self.poly_x_coeffs, avg_norm_gaze_x)
            screen_y = np.polyval(self.poly_y_coeffs, avg_norm_gaze_y)
            # Handling out-of-bounds
            screen_x = max(0, min(SCREEN_WIDTH, screen_x))
            screen_y = max(0, min(SCREEN_HEIGHT, screen_y))
            return (float(screen_x),float(screen_y))
        
        except Exception as e:
            print(f"Error iris to screen coordinates: {e}")
            return None

    def save_calibration(self) -> None:
        with open(self.calibration_file, 'w') as file:
            json.dump(self.calibration_data, file)
        
    def load_calibration(self) -> None:
        '''
        Immediately load calibration data from file, and train the model if successful.
        '''
        if os.path.exists(self.calibration_file):
            try:
                print("Loading calibration data...----------------------------------------------")
                with open(self.calibration_file, 'r') as f:
                    self.calibration_data = json.load(f)

                    for key in self.calibration_data:
                        self.calibration_data[key]['feature'] = np.array(self.calibration_data[key]['feature'])
                    
                    if len(self.calibration_data) >= self.POLY_DEGREE + 1:
                        self.is_calibrated=True
                        self.train_mapping_model() # Train on load
            except Exception as e:
                print(f"Error in load_calibration: {e}")

    @staticmethod
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
    
 
