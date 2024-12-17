import pickle
import os
import numpy as np
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL = 'model'

def get_sensibility(h_sensibility: int, s_sensibility: int, v_sensibility: int) -> np.ndarray:

    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    hSens = (h_sensibility * upper_color[0]) / 100
    SSens = (s_sensibility * upper_color[1]) / 100
    VSens = (v_sensibility * upper_color[2]) / 100

    lower_bound_color = np.array([lower_color[0] - hSens, lower_color[1] - SSens, lower_color[2] - VSens])
    upper_bound_color = np.array([lower_color[0] + hSens, lower_color[1] + SSens, lower_color[2] + VSens])

    return np.array([lower_bound_color, upper_bound_color])

def extract_features(thresh_img):

    features = []

    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    try:
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
    except ValueError:
        return None
    except cv2.error as e:
        return None
    
    finger_count = 0
    thumb_orientation = 0
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c)))
            if angle < 90:
                finger_count += 1
    
    x, y, w, h = cv2.boundingRect(max_contour)
    aspect_ratio = float(w) / h
    contour_area = cv2.contourArea(max_contour)

    # Thumb orientation estimation (simple assumption for thumb)
    moments = cv2.moments(max_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        for point in max_contour:
            px, py = point[0]
            if px < cx and py < cy:  # Thumb in upper-left quadrant
                thumb_orientation = 1  # Up
            elif px < cx and py > cy:  # Thumb in lower-left quadrant
                thumb_orientation = -1  # Down
    
    features = [aspect_ratio, contour_area, finger_count, thumb_orientation]
    return features

def segment_hand(roi: np.ndarray, lower_bound_color: np.ndarray, upper_bound_color: np.ndarray) -> list[np.ndarray]:

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    binary_mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
    return binary_mask

if __name__ == '__main__':

    with open(os.path.join(script_dir, MODEL), 'rb') as f:
        model = pickle.load(f)['model']

    lower_bound_color, upper_bound_color = get_sensibility(100, 49, 35)

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.rectangle(frame, (49, 49), (251, 251), (0, 255, 0), 0)

        roi = frame[50:250, 50:250]

        if roi is None:
            continue

        roi = segment_hand(roi, lower_bound_color, upper_bound_color)
        roi = cv2.inRange(roi, 200, 255)

        #features = extract_features(roi)
        #if features is not None:
        prediction = model.predict([roi.flatten()])
        cv2.putText(frame, f"Prediction: {prediction[0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        cv2.imshow('Threshold', roi)

        if cv2.waitKey(25) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()