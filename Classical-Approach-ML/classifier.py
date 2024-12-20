import os
import numpy as np
import cv2
import pickle
import util
import pyautogui

class HandGestureRecognizer:
    def __init__(self, model_path=os.path.join(util.script_dir, 'classifier.p')):
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)['model']
        self.x1, self.x2 = 50, 250
        self.y1, self.y2 = 50, 250

    def recognize_gesture(self, binary_image):
        
        features = util.extract_hog_features(binary_image)
        if features is None:
            return "No hand detected"
        prediction = self.classifier.predict([np.asarray(features)])
        return prediction[0]
    
    def find_hand_center(image):
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

def main():

    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    while True:
        # read the frame and flip it
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # ROI based solution
        cv2.rectangle(frame, (recognizer.x1, recognizer.y1), (recognizer.x2, recognizer.y2), (0, 255, 0), 2)
        roi = frame[recognizer.x1:recognizer.x2, recognizer.y1:recognizer.y2]

        # segment the image using kmeans
        roi = util.segment_hand_kmeans(roi)
        cv2.imshow('roi', roi)

        # recognize the gesture
        gesture = recognizer.recognize_gesture(roi)
        if gesture == 'closedfist':

            # Find the center of the hand
            center = HandGestureRecognizer.find_hand_center(roi)
            cv2.circle(frame, center, 3, [255, 0, 0], -1)
            print(center)

        elif gesture == 'thumbsup':
            pyautogui.hotkey('ctrl', '+')
        elif gesture == 'thumbsdown':
            pyautogui.hotkey('ctrl', '-')
        elif gesture == 'iloveyou':
            pyautogui.rightClick()
        elif gesture == 'victory':
            pyautogui.leftClick()
        else:
            pass

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()