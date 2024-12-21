import os
import pickle
import cv2
import numpy as np
import util

def create_hand_gesture_dataset():
    """
    Create dataset from hand gesture images by HOG
    """

    # two lists for the dataset
    all_features = []       # features
    all_labels = []         # label of the feature

    # get all the directories in the data directory
    labels = os.listdir(os.path.join(util.script_dir, util.DATA_DIR))

    # iterate on each directory in the data directory
    for label in labels:
        # iterate on each image in the directory
        label_path = os.path.join(util.script_dir, util.DATA_DIR, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            # read the image and get the HOG
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.inRange(img, 200, 255, cv2.THRESH_BINARY)
            features = util.extract_hog_features(img)
            if len(features) == 0:
                continue

            # append the feature and label to the lists
            all_features.append(features)
            all_labels.append(label)

    # save the dataset
    with open(os.path.join(util.script_dir, util.DATASET_NAME), 'wb') as f:
        pickle.dump({'features': np.array(all_features), 'labels': np.array(all_labels)}, f)
    
    # display the dataset information
    print("Dataset created and saved:")
    print(f"Total samples: {len(all_labels)}")
    print("Samples per gesture:")
    for gesture in util.gestures:
        print(f"{gesture}: {all_labels.count(gesture)}")

if __name__ == '__main__':
    create_hand_gesture_dataset()