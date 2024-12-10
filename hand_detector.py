import numpy as np
import cv2
import math

class hand_detector:

    ###################################### INITIALIZATION #######################################
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

    ################################### THRESHOLDING FUNCTIONS ##################################
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

    ################################ IMAGE PROCESSING ALGORITHMS ################################
    def detect(self, frame: np.ndarray, roi_position: str = 'left') -> None:
        '''
        main function to call to start the detection of the 2 finger process
        ----------
        Parameters
        ----------
        frame: numpy array
            The frame to process

        roi_position: str
            Position of the ROI block, left or right only
        '''
        if roi_position.lower() == 'left':
            roi = frame[100:300, 100:300]
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
        elif roi_position.lower() == 'right':
            roi = frame[50:300, 300:550]
            cv2.rectangle(frame, (300, 50), (550, 300), (0, 255, 0), 0)
        else:
            raise ValueError('roi_position must be left or right only')
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # get values from trackbar
        newHSens = cv2.getTrackbarPos('Hue Sensibility', 'Hand Detection')
        newSSens = cv2.getTrackbarPos('Saturation Sensibility', 'Hand Detection')
        newVSens = cv2.getTrackbarPos('Value Sensibility', 'Hand Detection')

        lower_bound_color, upper_bound_color = self.set_sensibility(newHSens, newSSens, newVSens)
        binary_mask, mask = self.segment_hand(hsv, roi, lower_bound_color, upper_bound_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            l = self.analyse_defects(cnt, roi)
            self.analyse_contours(frame, cnt, l + 1)
        except ValueError:
            pass
        self.show_results(binary_mask, mask, frame)

    def segment_hand(self, frame: np.ndarray, roi: np.ndarray, lower_bound_color: np.ndarray, upper_bound_color: np.ndarray) -> list[np.ndarray]:
        '''
        Segment the hand in ROI block.
        ----------
        Parameters
        ----------
        frame: numpy array
            The frame to process

        roi: numpy array
            part of the frame to detect on

        lower_bound_color: np array
            lower range of HSV

        upper_bound_color: np array
            upper range of HSV

        returns the segmented image frame and the ROI
        '''
        kernel = np.ones((3, 3), np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        binary_mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
        mask = cv2.dilate(binary_mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (5, 5), 90)
        return [binary_mask, mask]
    
    def analyse_defects(self, cnt, roi: np.ndarray) -> int:
        """
        Calculates how many convexity defects are on the image.
        A convexity defect is a area that is inside the convexity hull but does not belong to the object.
        Those defects in our case represent the division between fingers.
        ----------
        Parameters
        ----------
        cnt : array-like
          Contour of max area on the image, in this case, the contour of the hand

        roi : array-like
          Region of interest where should be drawn the found convexity defects
        """
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        l = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                d = (2 * ar) / a

                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)
                cv2.line(roi, start, end, [0, 255, 0], 2)
        return l

    def analyse_contours(self, frame: np.ndarray, cnt, l: int) -> None:
        """
        Writes to the image the signal of the hand.
        The hand signals can be the numbers from 0 to 5, the 'ok' signal, and the 'all right' symbol.
        The signals is first sorted by the number of convexity defects. Then, if the number of convexity defects is 1, 2, or 3, the area ratio is to be analysed.
        Parameters
        ----------
        frame : array-like
          The frame to be analysed
        cnt : array-like
          Contour of max area on the image, in this case, the contour of the hand
        l : int
          Number of convexity defects
        """
        hull = cv2.convexHull(cnt)

        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        arearatio = ((areahull - areacnt) / areacnt) * 100

        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Fixe', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 3:
            if arearatio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    def show_results(self, binary_mask: np.ndarray, mask: np.ndarray, frame: np.ndarray) -> None:
        """
        Shows the image with the results on it.
        The image is a result of a combination of the image with the result on it, the original captured ROI, and the ROI after optimizations.
        ----------
        Parameters
        ----------
        binary_mask : array-like
          ROI as it is captured

        mask : array-like
          ROI after optimizations

        frame : array-like
          Frame to be displayed
        """
        combine_masks = np.concatenate((binary_mask, mask), axis=0)
        height, _, _ = frame.shape
        _, width = combine_masks.shape
        masks_result = cv2.resize(combine_masks, dsize=(width, height))
        masks_result = cv2.cvtColor(masks_result, cv2.COLOR_GRAY2BGR)
        result_image = np.concatenate((frame, masks_result), axis=1)
        cv2.imshow('Hand Detection', result_image)

    # TODO: CONVEX HULL FROM SCRATCH
    def convex_hull():
        pass