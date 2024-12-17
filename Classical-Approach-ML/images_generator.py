import os
import cv2
import numpy as np

def get_sensibility(h_sensibility: int, s_sensibility: int, v_sensibility: int) -> np.ndarray:

    lower_color = np.array([0, 50, 120], dtype=np.uint8)
    upper_color = np.array([180, 150, 250], dtype=np.uint8)

    hSens = (h_sensibility * upper_color[0]) / 100
    SSens = (s_sensibility * upper_color[1]) / 100
    VSens = (v_sensibility * upper_color[2]) / 100

    lower_bound_color = np.array([lower_color[0] - hSens, lower_color[1] - SSens, lower_color[2] - VSens])
    upper_bound_color = np.array([lower_color[0] + hSens, lower_color[1] + SSens, lower_color[2] + VSens])

    return np.array([lower_bound_color, upper_bound_color])

def segment_hand(roi: np.ndarray, lower_bound_color: np.ndarray, upper_bound_color: np.ndarray) -> list[np.ndarray]:

    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    binary_mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
    # mask = cv2.dilate(binary_mask, kernel, iterations=3)
    # mask = cv2.erode(mask, kernel, iterations=3)
    # mask = cv2.GaussianBlur(mask, (5, 5), 90)
    return binary_mask


script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
classes = ['thumbsup', 'thumbsdown', 'iloveyou', 'openpalm', 'closedfist', 'victory']
dataset_size = 1000

def generate_images():

    if not os.path.exists(os.path.join(script_dir, DATA_DIR)):
        os.makedirs(os.path.join(script_dir, DATA_DIR))

    lower_bound_color, upper_bound_color = get_sensibility(100, 49, 35)

    cap = cv2.VideoCapture(0)
    for j in classes:
        
        if not os.path.exists(os.path.join(script_dir, DATA_DIR, j)):
            os.makedirs(os.path.join(script_dir, DATA_DIR, j))

        print(f'Collecting data for class {j}')
        
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[50:250, 50:250]
            cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 0)
            cv2.putText(frame, f'Press "Q" {j}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            roi = segment_hand(roi, lower_bound_color, upper_bound_color)
            cv2.imshow('roi', roi)

            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[50:250, 50:250]
            cv2.rectangle(frame, (49, 49), (251, 251), (0, 255, 0), 0)
            cv2.imshow('frame', frame)

            # SEGMENT THE IMAGE
            roi = segment_hand(roi, lower_bound_color, upper_bound_color)
            cv2.imshow('roi', roi)
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(script_dir, DATA_DIR, j, '{}.jpg'.format(counter)), roi)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_images()