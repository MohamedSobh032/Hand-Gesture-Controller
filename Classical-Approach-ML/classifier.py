import os
import numpy as np
import cv2
import pickle
import util
import pyautogui

class HandGestureRecognizer:

    def __init__(self, model_path=os.path.join(util.script_dir, util.MODEL_NAME)) -> None:
        '''Initialize the HandGestureRecognizer with the trained model'''

        # Read the model and save it in a variable
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)['model']


    def recognize_gesture(self, binary_image: np.ndarray) -> str:
        '''
        Recognize the gesture from the binary image by extracting HOG features then predicting the gesture
        '''

        # Extract HOG features
        features = util.extract_hog_features(binary_image)
        if features is None:
            return "No hand detected"
        
        # Predict the gesture
        prediction = self.classifier.predict([np.asarray(features)])
        return prediction[0]


    def take_action(self, frame: np.ndarray, roi: np.ndarray, gesture: str) -> None:
        '''
        Take action based on the recognized gesture by the user
        '''
        if gesture == 'closed_fist':
            center = HandGestureRecognizer.find_hand_center(roi)
            cv2.circle(frame, center, 3, [255, 0, 0], -1)
            pyautogui.moveTo(center[0], center[1])

        elif gesture == 'thumbs_up':
            pyautogui.hotkey('ctrl', '+')

        elif gesture == 'thumbs_down':
            pyautogui.hotkey('ctrl', '-')

        elif gesture == 'i_love_you':
            pyautogui.rightClick()

        elif gesture == 'victory':
            pyautogui.leftClick()



def main():
    '''Main function of our project'''

    # Initialize the camera and the recognizer
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    while True:

        # read the frame and flip it
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # ROI based solution
        cv2.rectangle(frame, (util.x1 - 1, util.y1 - 1), (util.x2 + 1, util.y2 + 1), (0, 255, 0), 2)
        roi = frame[util.x1:util.x2, util.y1:util.y2]

        # segment the image using kmeans
        roi_kmeans, center = util.segment_hand_kmeans(roi, 3)
        cv2.imshow('ROI of K-Means', roi_kmeans)

        # recognize the gesture
        gesture = recognizer.recognize_gesture(roi_kmeans)
        
        # DEBUGGING: GET CENTER OF THE HAND AND PRINT A DOT INTO IT
        cv2.circle(frame, center, 3, [255, 0, 0], -1)

        # Take action based on the gesture
        # TODO: TAKE ACTION HERE WHEN THE PROJECT IS FINISHED

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()