import os
import pickle
import util
import cv2
import numpy as np

def create_hand_gesture_dataset():
    """
    Create dataset from hand gesture images
    
    Args:
        dataset_dir (str): Directory containing gesture image folders
        gestures (list): List of gesture names
    
    Returns:
        tuple: (features, labels)
    """
    # two lists for the dataset
    all_features = []
    all_labels = []

    labels = os.listdir(os.path.join(util.script_dir, util.DATA_DIR))

    # iterate on each directory in the data directory
    for label in labels:
        label_path = os.path.join(util.script_dir, util.DATA_DIR, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.inRange(img, 200, 255, cv2.THRESH_BINARY)
            features = util.extract_hog_features(img)
            if len(features) == 0:
                continue
            all_features.append(features)
            all_labels.append(label)

    with open(os.path.join(util.script_dir, util.DATASET_NAME), 'wb') as f:
        pickle.dump({'features': np.array(all_features), 'labels': np.array(all_labels)}, f)
    
    print("Dataset created and saved:")
    print(f"Total samples: {len(all_labels)}")
    print("Samples per gesture:")
    for gesture in util.gestures:
        print(f"{gesture}: {all_labels.count(gesture)}")

if __name__ == '__main__':
    create_hand_gesture_dataset()