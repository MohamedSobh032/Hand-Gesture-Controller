import os
import cv2
import numpy as np
import util

def generate_images(DELAY_TIME: int = 5) -> None:
    '''
    Generates images for each gesture in the gestures list
    How to use:
        - Position your hand in the ROI
        - Press "Q" to start taking images for the current gesture
        - Start moving your hand in the ROI to take different images
        - When the dataset size is reached, the program will automatically switch to the next gesture
    '''

    # if data folder does not exist, generate it
    if not os.path.exists(os.path.join(util.script_dir, util.DATA_DIR)):
        os.makedirs(os.path.join(util.script_dir, util.DATA_DIR))

    # Initialize CAM
    cap = cv2.VideoCapture(util.CAM_INDEX)

    # for each gesture, generate images
    for j in util.gestures:
        
        # if gesture folder does not exist, generate it
        if not os.path.exists(os.path.join(util.script_dir, util.DATA_DIR, j)):
            os.makedirs(os.path.join(util.script_dir, util.DATA_DIR, j))

        # logging
        print(f'Collecting data for class {j}')
        
        # waiting for the user to get ready
        while True:

            # read the frame and flip it
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # ROI based solution
            roi = frame[util.x1:util.x2, util.y1:util.y2]
            cv2.rectangle(frame, (util.x1 - 1, util.y1 - 1), (util.x2 + 1, util.y2 + 1), (0, 255, 0), 0)

            # print statement
            cv2.putText(frame, f'Press "Q" {j}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # segment the image using kmeans
            roi = util.segment_hand_kmeans(roi)
            cv2.imshow('Segmentation', roi)

            # if user pressed q, break the waiting and start taking snippets
            if cv2.waitKey(10) == ord('q'):
                break

        # iterator of the current gesture dataset
        counter = 0
        while counter < util.dataset_size:

            # read the frame and flip it
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # ROI based solution
            roi = frame[util.x1:util.x2, util.y1:util.y2]
            cv2.rectangle(frame, (util.x1 - 1, util.y1 - 1), (util.x2 + 1, util.y2 + 1), (0, 255, 0), 0)

            # print statement
            cv2.imshow('frame', frame)

            # segment the image using kmeans
            roi = util.segment_hand_kmeans(roi)
            cv2.imshow('Segmentation', roi)

            # await between each snippet
            cv2.waitKey(DELAY_TIME)

            # save the image in the directory and increment to take the next one
            cv2.imwrite(os.path.join(util.script_dir, util.DATA_DIR, j, '{}.png'.format(counter)), roi)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate_images(5)