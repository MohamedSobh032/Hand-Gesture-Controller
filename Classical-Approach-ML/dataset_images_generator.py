import os
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
classes = ['thumbsup', 'thumbsdown', 'iloveyou', 'openpalm', 'closedfist', 'victory']
dataset_size = 100

def generate_images():

    if not os.path.exists(os.path.join(script_dir, DATA_DIR)):
        os.makedirs(os.path.join(script_dir, DATA_DIR))

    cap = cv2.VideoCapture(0)
    for j in classes:
        
        if not os.path.exists(os.path.join(script_dir, DATA_DIR, j)):
            os.makedirs(os.path.join(script_dir, DATA_DIR, j))

        print(f'Collecting data for class {j}')
        
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (50, 50), (250, 250), (0, 255, 0), 0)
            cv2.putText(frame, 'Press "Q"', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            roi = frame[50:250, 50:250]
            cv2.rectangle(frame, (49, 49), (251, 251), (0, 255, 0), 0)
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(script_dir, DATA_DIR, j, '{}.jpg'.format(counter)), roi)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_images()