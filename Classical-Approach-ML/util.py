import os
import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
import imutils

########################################## GLOBAL DEFINES AND VARIABLES ##########################################

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
DATASET_NAME = 'data.pkl'
gestures = ['thumbsup', 'thumbsdown', 'iloveyou', 'openpalm', 'closedfist', 'victory', 'nohand']
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

def extract_hand_features(binary_image):
    """
    Extract comprehensive features for hand gesture classification
    
    Args:
        binary_image (numpy.ndarray): Binary segmented hand image
    
    Returns:
        dict: Dictionary of extracted features
    """
    # Feature extraction dictionary
    features = {}
    
    # 1. Contour-based features
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Contour area and perimeter
    features['contour_area'] = cv2.contourArea(hand_contour)
    features['contour_perimeter'] = cv2.arcLength(hand_contour, True)
    
    # 2. Convex Hull and Defects Analysis
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    
    # Safer convexity defects handling
    try:
        defects = cv2.convexityDefects(hand_contour, hull)
        
        # Convexity defects
        if defects is not None and len(defects.shape) > 1:
            features['num_defects'] = len(defects)
            
            # Safely extract defect depths
            defect_depths = []
            for defect in defects:
                # Check if the defect has at least 4 elements
                if len(defect) >= 4:
                    defect_depths.append(defect[3]/256.0)
            
            features['avg_defect_depth'] = np.mean(defect_depths) if defect_depths else 0
        else:
            features['num_defects'] = 0
            features['avg_defect_depth'] = 0
    
    except cv2.error:
        # Fallback if convexity defects can't be computed
        features['num_defects'] = 0
        features['avg_defect_depth'] = 0
    
    # 3. Bounding Box Features
    x, y, w, h = cv2.boundingRect(hand_contour)
    features['aspect_ratio'] = float(w) / h if h > 0 else 1
    features['extent'] = features['contour_area'] / (w * h) if w * h > 0 else 1
    
    # 4. Finger Detection and Spread
    finger_detect = binary_image.copy()
    finger_contours, _ = cv2.findContours(finger_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    potential_fingers = [cnt for cnt in finger_contours if cv2.contourArea(cnt) > 50]
    features['num_fingers'] = len(potential_fingers)
    
    # 5. Pixel Distribution and Texture
    # Ensure image has enough variation for GLCM
    if binary_image.size > 0:
        # Ensure unique pixel values
        glcm = graycomatrix(binary_image, [1], [0], levels=256)
        features['contrast'] = graycoprops(glcm, 'contrast')[0, 0]
        features['dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
        features['homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    else:
        features['contrast'] = 0
        features['dissimilarity'] = 0
        features['homogeneity'] = 0
    
    # 6. Moment-based features
    moments = cv2.moments(hand_contour)
    # Add small epsilon to avoid log(0)
    hu_moments = cv2.HuMoments(moments).flatten()
    features['hu_moments'] = np.log(np.abs(hu_moments) + 1e-10)
    
    return features

def extract_features(thresh_img):
    features = []
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    try:
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
    except cv2.error as e:
        print("Error in convexityDefects:", e)
        return None
    
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

def extract_hog_features(img):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, (64, 128))
    return hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[0]

def extract_hsv_histogram(img):
    """
    TODO
    1. Resize the image to target_img_size using cv2.resize
    2. Convert the image from BGR representation (cv2 is BGR not RGB) to HSV using cv2.cvtColor
    3. Acquire the histogram using the cv2.calcHist. Apply the functions on the 3 channels. For the bins 
        parameter pass (8, 8, 8). For the ranges parameter pass ([0, 180, 0, 256, 0, 256]). Name the histogram
        <hist>.
    """
    resized = cv2.resize(img, (200, 200))
    hsv_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_resized], [0,1,2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()
