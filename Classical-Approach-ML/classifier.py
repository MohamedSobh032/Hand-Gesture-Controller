import os
import numpy as np
import cv2
import pickle
import util
import pyautogui

class HandGestureRecognizer:
    def __init__(self, model_path=os.path.join(util.script_dir, util.MODEL_NAME)):
        '''
        Initialize the HandGestureRecognizer with the trained model
        '''

        # Read the model and save it in a variable
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)['model']

    def recognize_gesture(self, binary_image):
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
    
    def find_hand_center(image):
        '''
        Find the center of the hand in the binary image
        '''

        # Find all non-zero points (hand pixels)
        non_zero_points = cv2.findNonZero(image)

        # Calculate center using moments
        M = cv2.moments(image)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            # Fallback if moments fail
            mean_point = np.mean(non_zero_points, axis=0)
            center_x, center_y = mean_point[0]

        return (center_x, center_y)


def take_action(frame: np.ndarray, roi: np.ndarray, gesture: str) -> None:
    '''
    Take action based on the recognized gesture by the user
    '''
    if gesture == 'closed_fist':
        # Find the center of the hand
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
    else:
        pass

def main():
    '''
    Main function of our project
    '''

    # Initialize the camera and the recognizer
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    pyautogui.FAILSAFE = False

    while True:
        # read the frame and flip it
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # ROI based solution
        cv2.rectangle(frame, (util.x1 - 1, util.y1 - 1), (util.x2 + 1, util.y2 + 1), (0, 255, 0), 2)
        roi = frame[util.x1:util.x2, util.y1:util.y2]

        # segment the image using kmeans
        roi = util.segment_hand_kmeans(roi)
        cv2.imshow('roi', roi)

        # recognize the gesture
        gesture = recognizer.recognize_gesture(roi)
        
        # Take action based on the gesture
        #take_action(frame, roi, gesture)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()