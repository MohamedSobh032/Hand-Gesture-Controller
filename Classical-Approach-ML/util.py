import os
import cv2
import numpy as np
from skimage.feature import hog
import pyclesperanto_prototype as cle

########################################## GLOBAL DEFINES AND VARIABLES ##########################################

# To be able to run the any code from any directory, we need to get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Data directory to put all the images
DATA_DIR = 'data'

# Dataset file name
DATASET_NAME = 'data.pkl'

# Generalization if different laptops
CAM_INDEX = 0

# Classifier file name
MODEL_NAME = 'classifier.pkl'

# All the gestures in our program, can be changed or added to
gestures = ['thumbs_up', 'thumbs_down', 'i_love_you', 'open_palm', 'closed_fist', 'victory']

# Number of frames to capture for each gesture
dataset_size = 1000

# Range of interest box dimensions
x1, x2, y1, y2 = 50, 250, 50, 250

##################################################################################################################

############################################ GENERAL FUNCTIONS NEEDED ############################################

def segment_hand_kmeans(ROI: np.ndarray, k: int = 3, reference_color: tuple = (170, 120, 90)) -> tuple:
    """
    Segment the hand using K-means clustering and return the segmented image and hand center.
    
    Args:
        ROI (numpy.ndarray): Region of interest containing the hand.
        k (int): Number of clusters for K-means.
        reference_color (tuple): Approximate RGB color of the hand for cluster identification.
    
    Returns:
        tuple: Segmented image (grayscale) and center coordinates of the hand cluster (x, y).
    """

    # Convert the image to RGB
    roi_rgb = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)

    # Reshaping the 3D image into a 2D array of pixels and 3 color values (RGB)
    # -1 for automatically calculate the size of this dimension based on array size
    pixel_vals = np.float32(roi_rgb.reshape((-1, 3)))

    # Define the termination criteria for the K-means algorithm. This specifies when the algorithm should stop
    # - cv2.TERM_CRITERIA_EPS: Stop when the SPECIFIED ACCURAY (epsilon) is reached
    #   OR  OR  OR  OR  OR  OR  OR  OR
    # - cv2.TERM_CRITERIA_MAX_ITER: Stop after a maximum number of iterations (100 in this case)
    # The minimum accuray will be 0.85 epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # Apply K-means clustering on the pixel values to segment the image into K clusters
    # - pixel_val ---> input data
    # - k number of clusters
    # - None for initial randomization of the cluster centers
    # - criteria of the stopping conditions for the algorithm
    # - 10 is the number of times the algorithm is executed with different initializations
    # - cv2.KMEANS_RANDOM_CENTERS initializes the cluster centers randomly.
    # The function returns
    # - labels: An array where each element corresponds to the cluster index for a pixel
    # - centers: The RGB values of the cluster centers
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the cluster centers from float32 to uint8 (0-255)
    centers = np.uint8(centers)

    # Map each pixel to the RGB value of its cluster center
    # - labels.flatten() to converts the 2D labels array into a 1D array for indexing
    # - centers[labels.flatten()] replaces each pixel's cluster index with the corresponding cluster center's RGB value
    segmented_data = centers[labels.flatten()]

    # Reshape the segmented data back into the original image's dimensions.
    # This restores the 2D image structure, with each pixel now having the RGB value of its cluster center
    # then converted back to gray-scale
    segmented_image = segmented_data.reshape((roi_rgb.shape))
    segmented_image_gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    

    # Determine the cluster that best corresponds to the hand region by comparing the cluster centers RGB values with a predefined reference color
    # - `reference_color` is the RGB value representing the approximate color of the hand.
    # - For each cluster center in `centers`, compute the Euclidean distance to `reference_color`
    #   using `np.linalg.norm`, which calculates the vector magnitude.
    # - `np.argmin(...)` finds the index of the cluster center with the smallest distance to `reference_color`
    closest_cluster_idx = np.argmin(
        [np.linalg.norm(center - np.array(reference_color)) for center in centers]
    )
    
    # Create a binary mask corresponding to the cluster identified as the hand.
    # - `labels.flatten()` converts the 2D labels array to 1D for comparison.
    # - The mask is True for pixels belonging to the hand cluster and False otherwise.
    # - Reshape the mask to match the original ROI dimensions (height and width).
    hand_mask = (labels.flatten() == closest_cluster_idx).reshape((ROI.shape[0], ROI.shape[1]))
    hand_mask = hand_mask.astype(np.uint8) * 255
    
    # Calculate the spatial center (centroid) of the hand cluster using image moments
    # - cv2.moments computes spatial moments of the binary mask.
    # - moments["m00"] is the zeroth-order moment, representing the total mass of the cluster (area of the mask).
    # - If `moments["m00"]` is non-zero, calculate the centroid coordinates:
    #   - `center_x` = m10/m00 (weighted average of x-coordinates).
    #   - `center_y` = m01/m00 (weighted average of y-coordinates).
    # - If `moments["m00"]` is zero (no detected cluster), set the center coordinates to `None`.
    moments = cv2.moments(hand_mask)
    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x, center_y = None, None  # No center found if no mass is detected

    return segmented_image_gray * 255, (center_x, center_y)


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) features from the input image.

    Parameters:
    img (np.ndarray): Input image as a NumPy array. It can be grayscale or color.

    Returns:
    np.ndarray: A 1D array of HOG feature descriptors extracted from the image.
    """

    # Resize the input image to a fixed size of 64x128 pixels
    # This is a standard size used for HOG feature extraction, especially in object detection
    img = cv2.resize(img, (64, 128))

    # Extract HOG features from the resized image.
    # - `orientations=9`: Divide the gradient orientation into 9 bins (0-180 degrees).
    # - `pixels_per_cell=(8, 8)`: Each cell is an 8x8 pixel region where gradients are aggregated.
    # - `cells_per_block=(2, 2)`: A block contains 2x2 cells, providing spatial normalization.
    # - `visualize=True`: Returns both the HOG descriptor and a visualization image (el mafrood tet4al bas kman shwaya)
    # - The HOG descriptor is a 1D feature vector representing the local shape information.
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[0]
