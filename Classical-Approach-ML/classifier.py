import os
import numpy as np
import cv2
import pickle
import util

class HandGestureRecognizer:
    def __init__(self, model_path=os.path.join(util.script_dir, 'classifier.p')):
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)['model']
        self.x1, self.x2 = 50, 250
        self.y1, self.y2 = 50, 250

    def preprocess_frame(self, frame):
        roi = util.segment_hand_kmeans(frame)
        cv2.imshow('roi', roi)
        return roi

    def recognize_gesture(self, frame):
        binary_image = self.preprocess_frame(frame)
        features = util.extract_hog_features(binary_image)
        if features is None:
            return "No hand detected"
        prediction = self.classifier.predict([np.asarray(features)])
        return prediction[0]

def main():
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (recognizer.x1, recognizer.y1), (recognizer.x2, recognizer.y2), (0, 255, 0), 2)
        roi = frame[recognizer.x1:recognizer.x2, recognizer.y1:recognizer.y2]
        gesture = recognizer.recognize_gesture(roi)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(25) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()