import numpy as np
import cv2

class hand_detector:

    def __init__(self, average_color: np.ndarray, maximum_color: np.ndarray) -> None:
        '''
        Creates an object of hand_detector, a class that contains all needed functions, variables and tuning variables
        to detect a hand inside ROI
        ----------
        Parameters
        ----------
        average_color: numpy array
            The average of HSV values to be detected
        maximum_color: numpy array
            The maximum of HSV values to be detected
        '''
        # Initialize class variables
        self.average_color = average_color
        self.maximum_color = maximum_color
        self.h_sensibility = 100
        self.s_sensibility = 100
        self.v_sensibility = 100
        # Initalize GUI (This part might be removed)
        cv2.namedWindow('Hand Detection')
        cv2.createTrackbar('Hue Sensibility', 'Hand Detection', self.h_sensibility, 100, lambda: None)
        cv2.createTrackbar('Saturation Sensibility', 'Hand Detection', self.s_sensibility, 100, lambda: None)
        cv2.createTrackbar('Value Sensibility', 'Hand Detection', self.v_sensibility, 100, lambda: None)

    def set_sensibility(self, h_sensibility: int, s_sensibility: int, v_sensibility: int) -> np.ndarray:
        '''
        Sets the sensibility to adapt to light environment better
        ----------
        Parameters
        ----------
        h_sensibility: int, range(0, 100)
            PERCENTAGE sensibility of hue channel
        s_sensibility: int, range(0, 100)
            PERCENTAGE sensibility of saturation channel
        v_sensibility: int, range(0, 100)
            PERCENTAGE sensibility of value channel
        '''
        hSens = (h_sensibility * self.maximum_color[0]) / 100
        SSens = (s_sensibility * self.maximum_color[1]) / 100
        VSens = (v_sensibility * self.maximum_color[2]) / 100
        lower_bound_color = np.array([self.average_color[0] - hSens, self.average_color[1] - SSens, self.average_color[2] - VSens])
        upper_bound_color = np.array([self.average_color[0] + hSens, self.average_color[1] + SSens, self.average_color[2] + VSens])
        return np.array([lower_bound_color, upper_bound_color])

