import os
import cv2
import numpy as np
from skimage.feature import hog

########################################## GLOBAL DEFINES AND VARIABLES ##########################################

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
DATASET_NAME = 'data.pkl'
gestures = ['thumbs_left', 'thumbs_right', 'i_love_you', 'open_palm', 'closed_fist', 'victory', 'no_hand_dark', 'no_hand_light']
dataset_size = 1000
random_seed = 42

############################################ GENERAL FUNCTIONS NEEDED ############################################

def segment_hand_kmeans(ROI):

    sobh = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = sobh.reshape((-1,3))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    # the below line of code defines the criteria for the algorithm to stop running, 
    # which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
    # becomes 85%
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # then perform k-means clustering with number of clusters defined as 3
    #also random centres are initially choosed for k-means clustering
    k = 2
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((sobh.shape))
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    masker = (segmented_image <= 150) & (segmented_image >= 0)
    segmented_image[masker] = 255
    segmented_image[~masker] = 0
    return segmented_image

def extract_hog_features(img):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, (64, 128))
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[0]

