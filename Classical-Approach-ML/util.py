import os
import cv2
import numpy as np
from skimage.feature import hog
import pyclesperanto_prototype as cle

########################################## GLOBAL DEFINES AND VARIABLES ##########################################

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
DATASET_NAME = 'data.pkl'
CAM_INDEX = 0
MODEL_NAME = 'classifier.pkl'

gestures = ['thumbs_up', 'thumbs_down', 'i_love_you', 'open_palm', 'closed_fist', 'victory']
dataset_size = 1000
random_seed = 42

x1, x2, y1, y2 = 50, 250, 50, 250

############################################ GENERAL FUNCTIONS NEEDED ############################################

def segment_hand_kmeans(ROI, k=3, reference_color=(170, 120, 90)):
    """
    Segment the hand using K-means clustering and return the segmented image and hand center.
    
    Args:
        ROI (numpy.ndarray): Region of interest containing the hand.
        k (int): Number of clusters for K-means.
        reference_color (tuple): Approximate RGB color of the hand for cluster identification.
    
    Returns:
        tuple: Segmented image (grayscale) and center coordinates of the hand cluster (x, y).
    """
    roi_rgb = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = roi_rgb.reshape((-1, 3))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    # Define criteria for the K-means algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert centers to 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((roi_rgb.shape))
    segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    
    # Find the cluster corresponding to the hand based on color proximity to reference
    closest_cluster_idx = np.argmin(
        [np.linalg.norm(center - np.array(reference_color)) for center in centers]
    )
    
    # Mask for the hand cluster
    hand_mask = (labels.flatten() == closest_cluster_idx).reshape((ROI.shape[0], ROI.shape[1]))
    hand_mask = hand_mask.astype(np.uint8) * 255
    
    # Find the center of the hand cluster using moments
    moments = cv2.moments(hand_mask)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x, center_y = None, None  # No center found if no mass is detected

    return segmented_image_gray * 255, (center_x, center_y)


def extract_hog_features(img):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, (64, 128))
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[0]
