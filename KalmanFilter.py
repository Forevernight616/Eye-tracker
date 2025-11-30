import numpy as np

class KalmanFilter:
    '''
    Kalman filtering is used to make the cursor less jittery
    '''
    def __init__(self, process_variance=1e-2, measurement_variance=0.1):
        # self.state: [x, y, dx, dy] -> position and velocity
        # self.A: Transition Matrix (A): Predicts next state
        # self.H: Measurement Matrix (H): Maps state to measurement (only observing position)
        # self.Q: Process Noise Covariance (Q): Uncertainty in the process model (motion)
        # self.R: Measurement Noise Covariance (R): Uncertainty in the measurement (gaze prediction)
        # self.P: Error Covariance Matrix (P): Initial guess of error
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        
        self.Q = np.identity(4) * process_variance
        self.R = np.identity(2) * measurement_variance
        self.P = np.identity(4)

    def predict(self) -> tuple:
        '''
        This is the Prediction Step of the filter. It uses the system's previous state to calculate the estimated current state, 
        based purely on a physics model (the assumption of constant velocity)
        
        Returns: 
            (float predicted_x_position, float predicted_y_position)
        '''
        # calculates the next predicted position of the cursor based on its last known position and velocity
        self.state = self.A @ self.state 
        # updates the uncertainty of the prediction
        self.P = self.A @ self.P @ self.A.T + self.Q 
        return (self.state[0,0], self.state[1,0])
    
    def update(self, measurement) -> tuple:
        '''
        Corrects the prediction made in predict() by incorporating the new noisy measurement just received from the camera model
        Args:
            measurement: (float, float), the noisy position captured from the webcam
        Returns: 
            (float predicted_x_position, float predicted_y_position)
        '''
        # Measurement: [x, y]
        measurement = np.array(measurement).reshape((2, 1))
        # S is uncertainty of Y
        S = self.H @ self.P @ self.H.T + self.R 
        # K is a weighting factor that determines how much the filter should adjust its prediction based on the innovation
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # Y is the difference between what the camera saw and what the filter expected to see
        Y = measurement - (self.H @ self.state)
        self.state = self.state + (K @ Y)
        # Update error covariance
        self.P = self.P - (K @ self.H @ self.P)
        return (self.state[0, 0], self.state[1, 0])
    


