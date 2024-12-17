import cv2
import numpy as np
import os
import pickle

def extract_features(thresh_img):
    features = []
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    
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


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'

if __name__ == '__main__':
    X = []
    y = []
    labels = os.listdir(os.path.join(script_dir, DATA_DIR))
    for label in labels:        
        label_path = os.path.join(script_dir, DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.inRange(img, 200, 255)
            if img is None:
                continue
            #features = extract_features(img)
            #if features is not None:
            X.append(img.flatten())
            y.append(label)
    with open(os.path.join(script_dir, 'data.pkl'), 'wb') as f:
        pickle.dump({'data': np.array(X), 'labels': np.array(y)}, f)