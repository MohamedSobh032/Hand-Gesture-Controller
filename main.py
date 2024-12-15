import numpy as np
import cv2
from hand_detector import hand_detector

########################################## DEFINES ##########################################
AI = 0
IP = 1
ALGORITHM = IP
CAMERA_INDEX = 0    ## SET TO DIFFERENT VALUES AS PER YOUR LAPTOP
lower_color = np.array([0, 50, 120], dtype=np.uint8)
upper_color = np.array([180, 150, 250], dtype=np.uint8)
#############################################################################################


################################### MAIN APPLICATION CODE ###################################
def normal_processing():
    """Main function of the app"""
    hand = hand_detector(lower_color, upper_color)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    while True:
        try:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            hand.detect(frame, 'left')
        except Exception as e:
            print(e)
        ################################## INPUT BREAKS ##################################
        key = cv2.waitKey(10)
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
#############################################################################################


if __name__ == '__main__':
    if ALGORITHM == IP:
        normal_processing()
    else:
        pass