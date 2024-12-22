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

def segment_hand_kmeans(ROI, k = 2):

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
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((sobh.shape))
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    return segmented_image

def thresholding_otsu(ROI):

    img_gaussian2 = cle.gaussian_blur(ROI, sigma_x=1, sigma_y=1, sigma_z=1)
    img_thresh = cle.threshold_otsu(img_gaussian2) * 255
    return np.array(img_thresh)

def extract_hog_features(img):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, (64, 128))
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[0]
